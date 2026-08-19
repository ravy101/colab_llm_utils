import copy
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from transformers.optimization import Adafactor
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time

from . import misc


def cascade_scored_samples(df, col, metric, ml_suffix='_13b'):
    thresh = np.linspace(0 - .001, 1 + 0.0011, 200)
    accs = []
    n_deferred = []
    p_deferred = []
    accept_acc = []
    deferred_acc = []
    deferred_correct = []
    coverage = []
    gains = []
    small_correct = df[metric].mean()
    ranks = df[col].rank(method='first') / len(df[col])
    for t in thresh:
        defer_idx = ranks > (1 - t)
        correct_7 = df[metric][~defer_idx].sum()
        correct_70 = df[metric + ml_suffix][defer_idx].sum()
        accept_acc.append(df[metric][~defer_idx].mean())
        deferred_acc.append(df[metric + ml_suffix][defer_idx].mean())
        deferred_correct.append(df[metric + ml_suffix][defer_idx].sum())
        n_deferred.append(defer_idx.sum())
        p_deferred.append((defer_idx.sum() / len(df)))
        coverage.append((~defer_idx).mean())
        accs.append((correct_7 + correct_70) / len(df))
        gains.append(accs[-1] - small_correct)
        if len(p_deferred) > 1:
            if p_deferred[-2] < .2 and p_deferred[-1] >= .2:
                p_del_20 = p_deferred.copy()
                accs_20 = accs.copy()
                p_del_20, accs_20 = misc.cap_interp_curve(p_del_20, accs_20, .2)
                auc_20 = np.trapezoid(accs_20, x=p_del_20)
            if p_deferred[-2] < .4 and p_deferred[-1] >= .4:
                p_del_40 = p_deferred.copy()
                accs_40 = accs.copy()
                p_del_40, accs_40 = misc.cap_interp_curve(p_del_40, accs_40, .4)
                auc_40 = np.trapezoid(accs_40, x=p_del_40)

    return {"p_deferred": p_deferred, "n_deferred": n_deferred, "deferred_correct": deferred_correct,
            "accepted_acc": accept_acc, "deferred_acc": deferred_acc, "accs": accs, "gains": gains,
            "auc": np.trapezoid(accs, x=p_deferred), "auc_20": auc_20, "auc_40": auc_40,
            "accs_20": accs_20, "accs_40": accs_40,
            "aurc": np.trapezoid(accept_acc, x=p_deferred), "p_del_20": p_del_20, "p_del_40": p_del_40}


def standardize_train_val(feats_train, feats_val, eps=1e-6):
    """Fold-safe z-scoring: statistics computed on TRAIN only, applied to both."""
    if feats_train is None:
        return None, None
    mean = feats_train.mean(axis=0, keepdims=True)
    std = feats_train.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    ftr = (feats_train - mean) / std
    fva = (feats_val - mean) / std if feats_val is not None else None
    return ftr.astype(np.float32), (fva.astype(np.float32) if fva is not None else None)


class TextClassificationDataset(Dataset):
    """Dataset for text classification / multilabel deferral with DeBERTa."""
    def __init__(self, texts, labels, tokenizer, max_length=512, multilabel=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.multilabel = multilabel

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        if self.multilabel:
            label_tensor = torch.tensor(label, dtype=torch.float)
        else:
            label_tensor = torch.tensor(label, dtype=torch.long)
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': label_tensor
        }


class FeatureFusionDataset(Dataset):
    """Text dataset that also emits a per-row numeric feature vector.

    feats may be None (text-only behaviour) or a (n, d) float array.
    """
    def __init__(self, texts, labels, tokenizer, feats=None, max_length=512, multilabel=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.feats = feats
        self.max_length = max_length
        self.multilabel = multilabel

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        label = self.labels[idx]
        if self.multilabel:
            label_tensor = torch.tensor(label, dtype=torch.float)
        else:
            label_tensor = torch.tensor(label, dtype=torch.long)
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': label_tensor,
        }
        if self.feats is not None:
            item['feats'] = torch.tensor(self.feats[idx], dtype=torch.float)
        return item


class DeBERTaClassificationHead(nn.Module):
    """DeBERTa model with classification head.

    forward accepts an ignored `feats` kwarg for API-compat with the fusion head.
    """
    def __init__(self, model_name, num_classes, dropout_rate=0.1):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.deberta.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, feats=None):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if torch.isnan(outputs).any():
            print("NaNs detected AFTER DeBERTa encoder!")
            print(f"inputs: {input_ids}")
            print(f"attention_mask: {attention_mask}")
            print(f"output shape: {outputs.shape}")
            print(f"outputs: {outputs}")
        mask = attention_mask.unsqueeze(-1).float()          # (B, T, 1)
        pooled = (outputs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        pooled = pooled.float()
        if torch.isnan(pooled).any():
            print("NaNs detected AFTER pooling!")
            print(f"min: {pooled.nanmin()}")
            print(f"max: {pooled.nanmax()}")
            nan_idx = torch.nonzero(torch.isnan(pooled))
            print("First pooled NaN index:", nan_idx[0])
        pooled = pooled.float()
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


class DeBERTaFusionHead(nn.Module):
    """DeBERTa encoder with optional late-fusion of numeric features.

    If num_features == 0 this reduces to a text-only head (safe drop-in).
    """
    def __init__(self, model_name, num_classes, num_features=0,
                 dropout_rate=0.1, text_proj_dim=128, feat_hidden=32):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(model_name)
        self.num_features = int(num_features)
        hidden = self.deberta.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        if self.num_features > 0:
            self.text_proj = nn.Linear(hidden, text_proj_dim)
            self.feat_mlp = nn.Sequential(
                nn.Linear(self.num_features, feat_hidden), nn.ReLU(),
                nn.Linear(feat_hidden, feat_hidden), nn.ReLU(),
            )
            self.classifier = nn.Linear(text_proj_dim + feat_hidden, num_classes)
        else:
            self.text_proj = None
            self.feat_mlp = None
            self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, input_ids, attention_mask, feats=None):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if torch.isnan(outputs).any():
            print("NaNs detected AFTER DeBERTa encoder!")
        mask = attention_mask.unsqueeze(-1).float()          # (B, T, 1)
        pooled = (outputs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        pooled = pooled.float()
        if torch.isnan(pooled).any():
            print("NaNs detected AFTER pooling!")
        if self.num_features > 0:
            if feats is None:
                raise ValueError("DeBERTaFusionHead built with num_features>0 but no feats passed.")
            t = self.dropout(torch.relu(self.text_proj(pooled)))
            f = self.feat_mlp(feats.float())
            fused = torch.cat([t, f], dim=1)
            return self.classifier(fused)
        return self.classifier(self.dropout(pooled))


def train_deberta_model(model, train_loader, val_loader, num_epochs=3, learning_rate=2e-5,
                        device='cpu', multilabel=False, pos_weight=None, threshold=0.5):
    """Train a DeBERTa head in multiclass or multilabel mode.

    Batches may optionally carry a 'feats' tensor; forwarded when present.
    """
    optimizer = Adafactor(model.parameters(), lr=learning_rate,
                          scale_parameter=False, relative_step=False)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    if multilabel:
        if pos_weight is not None and not torch.is_tensor(pos_weight):
            pos_weight = torch.tensor(pos_weight, dtype=torch.float)
        if pos_weight is not None:
            pos_weight = pos_weight.to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.CrossEntropyLoss()

    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        start = time.perf_counter()
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            feats = batch['feats'].to(device) if 'feats' in batch else None

            mask_sums = attention_mask.sum(dim=1)
            if torch.any(mask_sums == 0):
                print("Mask sum zero.")
                bad_rows = (mask_sums == 0)
                attention_mask[bad_rows, 0] = 1
            if torch.any(input_ids < 0):
                print("Warning: Negative input_ids detected (possible NaN text).")
            if torch.any(attention_mask < 0):
                print("Warning: Negative attention_mask detected.")

            if not multilabel:
                if torch.any(labels < 0):
                    print("Warning: Negative labels detected (possible NaN in targets).")
                if hasattr(model, 'classifier') and hasattr(model.classifier, 'out_features'):
                    if torch.any(labels >= model.classifier.out_features):
                        print(f"Warning: Labels out of bounds! Max label: {labels.max().item()}, "
                              f"Num classes: {model.classifier.out_features}")

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, feats=feats)
            if multilabel:
                loss = criterion(logits, labels.float())
            else:
                loss = criterion(logits, labels)

            if torch.isnan(loss):
                print("Warning: NaN loss detected!")
                print(f"labels {labels}")
                print(f"logits {logits}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

        scheduler.step()
        print(f"Finished epoch {epoch+1}")
        end = time.perf_counter()
        print(f"Iteration {epoch+1} took {end - start:0.4f} seconds")
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                feats = batch['feats'].to(device) if 'feats' in batch else None

                logits = model(input_ids, attention_mask, feats=feats)
                if multilabel:
                    loss = criterion(logits, labels.float())
                    preds = (torch.sigmoid(logits) > threshold).int()
                else:
                    loss = criterion(logits, labels)
                    preds = torch.argmax(logits, dim=-1)
                total_val_loss += loss.item()
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        end = time.perf_counter()
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        if multilabel:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='macro', zero_division=0)
            micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='micro', zero_division=0)
            exact_match = (all_preds == all_labels).all(axis=1).mean()
            print(f"\n================ Epoch {epoch+1}/{num_epochs} (multilabel) ================")
            print(f"Time Elapsed     : {end - start:0.2f} seconds")
            print(f"Train Loss       : {avg_train_loss:.4f}")
            print(f"Validation Loss  : {avg_val_loss:.4f}")
            print(f"Exact-match Acc  : {exact_match:.4f}")
            print(f"Macro P/R/F1     : {precision:.4f} / {recall:.4f} / {f1:.4f}")
            print(f"Micro P/R/F1     : {micro_p:.4f} / {micro_r:.4f} / {micro_f1:.4f}")
            print("=============================================")
        else:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='macro', zero_division=0)
            accuracy = accuracy_score(all_labels, all_preds)
            print(f"\n================ Epoch {epoch+1}/{num_epochs} ================")
            print(f"Time Elapsed   : {end - start:0.2f} seconds")
            print(f"Train Loss     : {avg_train_loss:.4f}")
            print(f"Validation Loss: {avg_val_loss:.4f}")
            print(f"Accuracy       : {accuracy:.4f}")
            print(f"Precision (Mac): {precision:.4f}")
            print(f"Recall (Macro) : {recall:.4f}")
            print(f"F1-Score (Mac) : {f1:.4f}")
            print("=============================================")
    return model


def predict_deberta_proba(model, val_loader, num_classes, device='cpu', multilabel=False):
    """Probability predictions from a DeBERTa head. Forwards 'feats' when present."""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            feats = batch['feats'].to(device) if 'feats' in batch else None
            logits = model(input_ids, attention_mask, feats=feats)
            if multilabel:
                probs = torch.sigmoid(logits)
            else:
                probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
    return np.vstack(all_probs)


def post_hoc_oof(df, feature_cols, target_col, n_splits=5, random_state=42,
                 rf_kwargs=None, model_type=LogisticRegression):
    """Out-of-fold predictions for each row using k-fold CV."""
    for c in feature_cols:
        if c not in list(df.columns):
            print(f"feature {c} not found in dataframe.")
            feature_cols.remove(c)
    if rf_kwargs is None:
        rf_kwargs = {}
    X = df[feature_cols].values
    y = df[target_col].values
    oof_preds = np.zeros(len(df))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]
        model = model_type(random_state=random_state, **rf_kwargs)
        model.fit(X_train, y_train)
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 0]
    return pd.Series(oof_preds, index=df.index, name="oof_prediction")


def post_hoc_oof_cont(df, feature_cols, target_col, n_splits=5, random_state=42,
                      rf_kwargs=None, model_type=RandomForestRegressor):
    """Out-of-fold predictions (regression) for each row using k-fold CV."""
    for c in feature_cols:
        if c not in list(df.columns):
            print(f"feature {c} not found in dataframe.")
            feature_cols.remove(c)
    if rf_kwargs is None:
        rf_kwargs = {}
    X = df[feature_cols].values
    y = df[target_col].values
    oof_preds = np.zeros(len(df))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]
        model = model_type(random_state=random_state, **rf_kwargs)
        model.fit(X_train, y_train)
        oof_preds[val_idx] = model.predict(X_val)
    return pd.Series(oof_preds, index=df.index, name="oof_prediction")


class MultiaxialCascade:
    def __init__(self, origin_df, axes_names, metric_col="gpt_score", fill_undefined=True, k=4, seed=42):
        self.metric_col = metric_col
        self.axes_names = axes_names
        self.origin = (0,) * len(axes_names)
        self.registry = {self.origin: origin_df}
        self.cost_registry = {self.origin: 1}
        self.model_registry = {}
        self.stage_of = {}
        self.pref_def_registry = {}
        self.kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    def register_axis_data(self, df, position, cost, stage = None):
        """Adds a dataframe for a specific point in the cascade grid."""
        if len(position) != len(self.axes_names):
            raise ValueError(f"Position invalid, expexted {len(self.axes_names)} dimensions.")
        self.registry[position] = df
        self.cost_registry[position] = cost
        df['inf_cost'] = df['prompt_len'] * cost[0] + df['output_len'] * cost[1]
        if stage is not None:
            self.stage_of[tuple(position)] = int(stage)
        print(f"Registered {[ax + ': ' + str(position[i]) for i, ax in enumerate(self.axes_names)]} | Shape: {df.shape} | Cost: {cost}")
        self.normalize_dfs()

    def _stage_members(self):
        stages = {}
        for pos, k in self.stage_of.items():
            stages.setdefault(k, []).append(pos)
        return [stages[k] for k in sorted(stages)]   # -> [[(0,0)], [(0,1),(1,0)], [(2,2)]]

    def _assert_legal_deferral(self, from_pos, to_pos):
        if self.stage_of[to_pos] != self.stage_of[from_pos] + 1:
            raise ValueError(f"{from_pos} (stage {self.stage_of[from_pos]}) may only "
                             f"defer to stage {self.stage_of[from_pos]+1}; "
                             f"{to_pos} is stage {self.stage_of[to_pos]}.")

    def normalize_dfs(self):
        min_len = min([len(df) for df in self.registry.values()])
        for df in self.registry.values():
            if len(df) > min_len:
                print(f"Dropping {len(df) - min_len} rows for consistency.")
                df.drop(df.index[min_len:], inplace=True)

    def compute_cv_splits(self):
        """Generate fold indeces and fix them across all dataframes."""
        df = self.registry[self.origin]
        df['fold'] = -1
        for fold_idx, (_, test_indices) in enumerate(self.kf.split(df)):
            df.iloc[test_indices, df.columns.get_loc('fold')] = fold_idx
        origin_fold = df['fold']
        for other_df in self.registry.values():
            other_df['fold'] = origin_fold

    def _build_feature_matrix(self, df, feature_cols, normalize_features=False):
        """Build a (n, d) float32 feature matrix from df[feature_cols].

        Returns (feats, n_features). feats is None (n_features 0) when no
        feature_cols supplied, preserving text-only behaviour. Normalization is
        NOT applied here even if normalize_features=True: z-scoring must be
        fold-safe (train stats only), so it is deferred to standardize_train_val
        inside each fold; the flag is threaded through by the callers.
        """
        if not feature_cols:
            return None, 0
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"feature_cols not found in dataframe: {missing}")
        feats = df[feature_cols].values.astype(np.float32)
        return feats, feats.shape[1]

    def set_pref_deferral_at(self, position, column, offset=-1):
        self.registry[position]["preferred_deferral"] = self.registry[position][column] + offset

    def evaluate_cascade_over_rates(self, deferral_column, from_position=None,
                                    cost_col='inf_cost', rates=None,
                                    metric_override=None):
        """Sweep the position-normalized deferral rate and evaluate the whole
        recursive cascade at each rate (via resolve_full_deferred).

        Return shape mirrors full_threshold_sim_temp: the x-axis is p_deferred
        (fraction of entry rows that deferred at least one stage) and spend is
        the mean accumulated path cost (sum of cost_stage_*).

        deferral_column : per-position WHETHER-to-defer signal (high = defer).
        from_position   : entry position (default origin).
        rates           : iterable of deferral rates in [0, 1]
                          (default np.linspace(0, 1, 51)).
        """
        metric = metric_override or self.metric_col
        if from_position is None:
            from_position = self.origin
        from_position = tuple(from_position)
        if rates is None:
            rates = np.linspace(0.0, 1.0, 51)

        entry_df = self.registry[from_position]
        n = len(entry_df)
        entry_idx = entry_df.index

        accs, p_deferred, n_deferred, coverage = [], [], [], []
        accept_acc, deferred_acc, deferred_correct, spend = [], [], [], []

        # sentinels so the capped-AUC blocks always resolve, even if the rate
        # grid never crosses a boundary (e.g. resolution too coarse).
        p_del_20 = accs_20 = p_del_40 = accs_40 = None
        auc_20 = auc_40 = np.nan

        for r in rates:
            resolved = self.resolve_full_deferred(
                from_position, deferral_column, float(r), cost_col=cost_col)
            m = resolved[metric].values
            deferred = (resolved['final_position'].values != from_position)
            # object-array elementwise compare guard (tuples don't broadcast)
            deferred = np.fromiter(
                (fp != from_position for fp in resolved['final_position'].values),
                bool, n)
            kept = ~deferred

            accs.append(float(m.mean()))
            p_deferred.append(float(deferred.mean()))
            n_deferred.append(int(deferred.sum()))
            coverage.append(float(kept.mean()))
            accept_acc.append(float(m[kept].mean()) if kept.any() else np.nan)
            deferred_acc.append(float(m[deferred].mean()) if deferred.any() else np.nan)
            deferred_correct.append(int(m[deferred].sum()))
            spend.append(float(resolved['resolved_cost'].mean()))

            if len(p_deferred) > 1:
                if p_deferred[-2] < .2 and p_deferred[-1] >= .2:
                    p_del_20, accs_20 = misc.cap_interp_curve(
                        p_deferred.copy(), accs.copy(), .2)
                    auc_20 = np.trapezoid(accs_20, x=p_del_20)
                if p_deferred[-2] < .4 and p_deferred[-1] >= .4:
                    p_del_40, accs_40 = misc.cap_interp_curve(
                        p_deferred.copy(), accs.copy(), .4)
                    auc_40 = np.trapezoid(accs_40, x=p_del_40)

        spend_arr = np.asarray(spend, dtype=float)
        span = (spend_arr.max() - spend_arr.min()) if spend_arr.max() > spend_arr.min() else 1.0
        spend_norm = ((spend_arr - spend_arr.min()) / span).tolist()
        order = np.argsort(spend_arr)
        cost_auc = np.trapezoid(np.asarray(accs)[order], x=spend_arr[order])
        cost_auc_norm = np.trapezoid(np.asarray(accs)[order], x=np.asarray(spend_norm)[order])

        return {"p_deferred": p_deferred, "n_deferred": n_deferred,
                "deferred_correct": deferred_correct, "accepted_acc": accept_acc,
                "deferred_acc": deferred_acc, "accs": accs, "acc": accs,
                "auc": np.trapezoid(accs, x=p_deferred), "auc_20": auc_20, "auc_40": auc_40,
                "accs_20": accs_20, "accs_40": accs_40,
                "aurc": np.trapezoid(accept_acc, x=p_deferred),
                "p_del_20": p_del_20, "p_del_40": p_del_40,
                "rates": list(map(float, rates)),
                "spend": spend, "spend_norm": spend_norm,
                "cost_auc": cost_auc, "cost_auc_norm": cost_auc_norm}

    def _defer_mask_by_rate(self, deferral_values, arrived_mask, deferral_rate):
        """Position-normalized deferral flag.

        Among rows in arrived_mask, select the top `deferral_rate` fraction by
        deferral_values (HIGH = defer), using the same rank rule as the original
        single-step simulator (rank(method='first')/n > 1 - rate).
        """
        mask = np.zeros(len(deferral_values), dtype=bool)
        if deferral_rate <= 0 or not arrived_mask.any():
            return mask
        sub = pd.Series(deferral_values[arrived_mask])
        ranks = sub.rank(method='first').values / len(sub)
        deferred = ranks > (1 - deferral_rate)
        mask[np.where(arrived_mask)[0][deferred]] = True
        return mask

    def resolve_full_deferred(self, position, deferral_column, deferral_rate,
                              cost_col='inf_cost'):
        """Recursively resolve a position-normalized multi-stage cascade.

        At each position, among the rows that REACHED it, the top
        `deferral_rate` fraction (ranked by `deferral_column`, high = defer) are
        routed onward via the per-position destination column named in
        self.pref_def_registry[position]. A row is RESOLVED at a position when
        it is under the deferral threshold OR the position has no successor
        (not in pref_def_registry).

        Returns a DataFrame indexed like `position`'s dataframe, carrying the
        fields of each row's RESOLVED (deepest-reached) position, plus:
          cost_stage_1 .. cost_stage_D : `cost_col` paid at each hop depth
            (stage_1 always populated; deeper stages 0 unless the row reached
            that depth),
          final_position : the resolved position tuple per row,
          resolved_cost  : sum of cost_stage_* (total path cost).
        """
        entry = tuple(position)
        idx = self.registry[entry].index
        n = len(idx)

        final_pos = np.empty(n, dtype=object)
        for i in range(n):
            final_pos[i] = entry
        stage_cost = {1: self.registry[entry][cost_col].values.astype(float).copy()}

        def recurse(pos, arrived_mask, depth):
            pos = tuple(pos)
            df = self.registry[pos]
            # settle everyone currently here (default resolved position)
            for j in np.where(arrived_mask)[0]:
                final_pos[j] = pos
            # cost paid at this hop depth by the rows that reached it
            if depth > 1:
                arr = stage_cost.setdefault(depth, np.zeros(n, dtype=float))
                arr[arrived_mask] = df[cost_col].values[arrived_mask]
            # terminal: no successor stage
            if pos not in self.pref_def_registry:
                return
            defer = self._defer_mask_by_rate(
                df[deferral_column].values, arrived_mask, deferral_rate)
            if not defer.any():
                return
            dests = df[self.pref_def_registry[pos]].values
            by_dest = {}
            for j in np.where(defer)[0]:
                d = tuple(dests[j])
                by_dest.setdefault(d, np.zeros(n, dtype=bool))[j] = True
            for d, mask_d in by_dest.items():
                recurse(d, mask_d, depth + 1)

        recurse(entry, np.ones(n, dtype=bool), 1)

        rows = [self.registry[final_pos[k]].iloc[k] for k in range(n)]
        out = pd.DataFrame(rows).reset_index(drop=True)
        out.index = idx
        max_depth_seen = max(stage_cost)
        for k in range(1, max_depth_seen + 1):
            out[f'cost_stage_{k}'] = stage_cost.get(k, np.zeros(n))
        out['final_position'] = [final_pos[k] for k in range(n)]
        out['resolved_cost'] = sum(out[f'cost_stage_{k}']
                                   for k in range(1, max_depth_seen + 1))
        return out

    def fit_post_hoc_at(self, position, feature_cols, rf_kwargs=None,
                        model_type=RandomForestClassifier):
        """Returns out-of-fold predictions for each row using k-fold CV."""
        df = self.registry[position]
        for c in feature_cols:
            if c not in list(df.columns):
                print(f"feature {c} not found in dataframe.")
                feature_cols.remove(c)
        if rf_kwargs is None:
            rf_kwargs = {}
        X = df[feature_cols].values
        target_dict = {0: df[self.metric_col]}
        deferral_options = {0: position}
        for i, _ in enumerate(self.axes_names):
            pos = copy.deepcopy(position)
            pos = pos[:i] + (pos[i] + 1,) + pos[i+1:]
            deferral_options[i+1] = pos
            target_dict[i+1] = self.registry[pos][self.metric_col]
        target_df = pd.DataFrame(target_dict)
        print(f"target dict shape {target_df.shape}")
        targets = target_df.apply(misc.biased_idxmax, axis=1)
        print(f"targets:{targets.describe()}")
        y = targets
        all_classes = np.arange(len(self.axes_names) + 1)
        n_classes = len(all_classes)
        oof_preds = np.zeros((len(df), n_classes))
        for i in range(self.kf.get_n_splits()):
            train_idx = df['fold'] != i
            val_idx = df['fold'] == i
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]
            model = model_type(**rf_kwargs)
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_val)
            present_classes = model.classes_
            aligned = np.zeros((len(X_val), n_classes))
            aligned[:, present_classes] = probs
            oof_preds[val_idx] = aligned
        df['post_hoc'] = oof_preds[:, 0]
        df['post_hoc'] = 1 - df['post_hoc']
        def_destinations = []
        for idx in oof_preds[:, 1:].argmax(axis=1):
            l = list(position)
            l[idx] = l[idx] + 1
            def_destinations.append(tuple(l))
        df['preferred_deferral'] = def_destinations
        return pd.DataFrame(oof_preds, index=df.index)

    def set_oracle_pref_deferral_at(self, position, tie_breaker=misc.biased_idxmax, allow_keep=True):
        """Sets the oracle (ground-truth) preferred deferral destination per row."""
        if position not in self.registry:
            raise KeyError(f"Position {position} not registered.")
        df = self.registry[position]
        if allow_keep:
            deferral_options = {0: tuple(position)}
            target_dict = {0: df[self.metric_col].values}
        else:
            deferral_options = {}
            target_dict = {}
        for i, _ in enumerate(self.axes_names):
            pos = position[:i] + (position[i] + 1,) + position[i+1:]
            if pos not in self.registry:
                raise KeyError(
                    f"Required neighbour {pos} (axis '{self.axes_names[i]}') "
                    f"not registered; cannot compute oracle preference.")
            target_dict[i + 1] = self.registry[pos][self.metric_col].values
            deferral_options[i + 1] = pos
        target_df = pd.DataFrame(target_dict, index=df.index)
        oracle_idx = target_df.apply(tie_breaker, axis=1).astype(int)
        df['oracle_pref_idx'] = oracle_idx.values
        df['oracle_preferred_deferral'] = oracle_idx.map(deferral_options).values
        return df[['oracle_pref_idx', 'oracle_preferred_deferral']]

    def fit_post_hoc_lm_at(self, position, input_text_col, output_text_col,
                           model_name="microsoft/deberta-v3-small",
                           num_epochs=3, batch_size=8, learning_rate=2e-5, max_length=512,
                           multilabel=False, multilabel_keep=False, recovery_fn=None,
                           use_pos_weight=True, threshold=0.5, simple_def_col=None,
                           target_func=misc.basic_idxmax, feature_cols=None,
                           normalize_features=False, device=None, verbose=True):
        """Fit a DeBERTa head for post-hoc deferral prediction (multiclass or multilabel).

        feature_cols : optional numeric columns to fuse via DeBERTaFusionHead.
        normalize_features : fold-safe z-scoring of fused features (train stats only).
            Default False (raw scales).
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        df = self.registry[position]
        for c in [input_text_col, output_text_col]:
            if c not in list(df.columns):
                print(f"feature {c} not found in dataframe.")
                raise ValueError(f"Column {c} not found")

        combined_texts = (df[input_text_col].astype(str) + " [SEP] "
                          + df[output_text_col].astype(str)).values
        feats_all, n_features = self._build_feature_matrix(df, feature_cols, normalize_features)

        axis_positions = []
        for i, _ in enumerate(self.axes_names):
            pos = copy.deepcopy(position)
            pos = pos[:i] + (pos[i] + 1,) + pos[i+1:]
            if pos not in self.registry:
                raise KeyError(f"Required neighbour {pos} (axis '{self.axes_names[i]}') not registered.")
            axis_positions.append(pos)

        if multilabel:
            if not multilabel_keep:
                if recovery_fn is None:
                    recovery_fn = lambda base, axis: (axis > base).astype(np.float32)
                base_score = df[self.metric_col].values
                label_cols = []
                for pos in axis_positions:
                    axis_score = self.registry[pos][self.metric_col].values
                    label_cols.append(recovery_fn(base_score, axis_score).astype(np.float32))
                targets = np.stack(label_cols, axis=1)
                # after targets are built (multilabel, destination-correctness semantics)
                informative = ~np.all(targets == targets[:, [0]], axis=1)
                n_outputs = len(self.axes_names)
                axis_offset = 0
            else:
                base_score = df[self.metric_col].values
                label_cols = [base_score]
                for pos in axis_positions:
                    label_cols.append(self.registry[pos][self.metric_col].values)
                targets = np.stack(label_cols, axis=1)
                n_outputs = len(self.axes_names) + 1
                axis_offset = 1
            pos_weight = None
            if use_pos_weight:
                t_inf = targets[informative]
                pos = t_inf.sum(axis=0)
                neg = t_inf.shape[0] - pos
                pos_weight = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0)
        else:
            target_dict = {}
            option_count = 0
            if simple_def_col is None:
                target_dict[0] = df[self.metric_col]
                option_count += 1
            for i, pos in enumerate(axis_positions):
                target_dict[i + option_count] = self.registry[pos][self.metric_col]
            target_df = pd.DataFrame(target_dict)
            targets = target_df.apply(target_func, axis=1).values
            n_outputs = len(target_dict)
            pos_weight = None
            axis_offset = 0

        oof_preds = np.zeros((len(df), n_outputs))

        if verbose:
            print(f"mode: {'multilabel' if multilabel else 'multiclass'}")
            print(f"n_outputs: {n_outputs} | n_features: {n_features} "
                  f"| normalize_features: {normalize_features} | device: {device}")
            if multilabel:
                rates = targets[:, axis_offset:].mean(axis=0)
                for ax, r in zip(self.axes_names, rates):
                    print(f"  recovery rate [{ax}]: {r:.4f}")
                if pos_weight is not None:
                    print(f"  pos_weight: {np.round(pos_weight, 3)}")
            else:
                print(f"targets: {pd.Series(targets).describe()}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            raise

        for fold_idx in range(self.kf.get_n_splits()):
            if verbose:
                print(f"Training fold {fold_idx + 1}/{self.kf.get_n_splits()}")

            train_mask = (df['fold'] != fold_idx).values & informative
            val_mask = (df['fold'] == fold_idx).values
            X_train_texts = combined_texts[train_mask]
            X_val_texts = combined_texts[val_mask]
            y_train = targets[train_mask]
            y_val = targets[val_mask]

            if feats_all is not None:
                f_train = feats_all[train_mask]
                f_val = feats_all[val_mask]
                if normalize_features:
                    f_train, f_val = standardize_train_val(f_train, f_val)
            else:
                f_train = f_val = None

            train_dataset = FeatureFusionDataset(
                X_train_texts, y_train, tokenizer, feats=f_train,
                max_length=max_length, multilabel=multilabel)
            val_dataset = FeatureFusionDataset(
                X_val_texts, y_val, tokenizer, feats=f_val,
                max_length=max_length, multilabel=multilabel)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            model = DeBERTaFusionHead(model_name, n_outputs, num_features=n_features, dropout_rate=0.1)
            self.model_registry[position] = model
            try:
                model = train_deberta_model(
                    model, train_loader, val_loader,
                    num_epochs=num_epochs, learning_rate=learning_rate, device=device,
                    multilabel=multilabel, pos_weight=pos_weight, threshold=threshold)
            except Exception as e:
                print(f"Error during training on fold {fold_idx}: {e}")
                raise

            probs = predict_deberta_proba(model, val_loader, n_outputs,
                                          device=device, multilabel=multilabel)
            if multilabel:
                oof_preds[val_mask] = probs
            else:
                present_classes = np.arange(n_outputs)
                aligned = np.zeros((len(X_val_texts), n_outputs))
                aligned[:, present_classes] = probs
                oof_preds[val_mask.values] = aligned

            del model, train_dataset, val_dataset, train_loader, val_loader
            torch.cuda.empty_cache()

        if multilabel:
            if multilabel_keep:
                df['post_hoc_lm'] = oof_preds[:, 0]
                offset_ax = 1
            else:
                offset_ax = 0
            for i, ax in enumerate(self.axes_names):
                df[f'post_hoc_lm_{ax}'] = oof_preds[:, i + offset_ax]
            best_axis = oof_preds[:, offset_ax:].argmax(axis=1)
            def_destinations = []
            for idx in best_axis:
                l = list(position)
                l[idx] = l[idx] + 1
                def_destinations.append(tuple(l))
            df['preferred_deferral_lm'] = def_destinations
        else:
            offset = 0
            if simple_def_col is None:
                df['post_hoc_lm'] = oof_preds[:, 0]
                df['post_hoc_lm'] = 1 - df['post_hoc_lm']
                offset = 1
            else:
                df['post_hoc_lm'] = df[simple_def_col]
            def_destinations = []
            for idx in oof_preds[:, offset:].argmax(axis=1):
                l = list(position)
                l[idx] = l[idx] + 1
                def_destinations.append(tuple(l))
            df['preferred_deferral_lm'] = def_destinations

        if verbose:
            print(f"Completed fit_post_hoc_lm_at at position {position} "
                  f"(multilabel={multilabel}, n_features={n_features})")
        return pd.DataFrame(oof_preds, index=df.index)

    def fit_correctness_lm_at(self, position=None, input_text_col="prompts",
                              output_text_col="responses",
                              model_name="microsoft/deberta-v3-small",
                              num_epochs=3, batch_size=8, learning_rate=2e-5, max_length=512,
                              use_pos_weight=True, threshold=0.5, score_col="post_hoc_conf",
                              feature_cols=None, normalize_features=False,
                              device=None, verbose=True):
        """Fit a DeBERTa head to predict correctness of the model at `position`,
        independent of any downstream axis. Plain post-hoc confidence model.

        Binary target: y = 1 if metric_col > 0 else 0. 2-logit softmax head.
        Writes score_col = P(incorrect) (HIGH = defer).

        feature_cols / normalize_features : optional late-fusion of numeric
            features with optional fold-safe z-scoring (default off).
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if position is None:
            position = self.origin
        df = self.registry[position]
        if 'fold' not in df.columns:
            raise RuntimeError("No 'fold' column found. Call compute_cv_splits() first.")
        for c in [input_text_col, output_text_col]:
            if c not in list(df.columns):
                raise ValueError(f"Column {c} not found in dataframe at {position}.")

        combined_texts = (df[input_text_col].astype(str) + " [SEP] "
                          + df[output_text_col].astype(str)).values
        feats_all, n_features = self._build_feature_matrix(df, feature_cols, normalize_features)

        targets = (df[self.metric_col].values > 0).astype(np.int64)
        n_outputs = 2

        class_weight = None
        if use_pos_weight:
            counts = np.bincount(targets, minlength=n_outputs).astype(float)
            counts = np.maximum(counts, 1.0)
            class_weight = targets.shape[0] / (n_outputs * counts)

        oof_preds = np.zeros((len(df), n_outputs))

        if verbose:
            acc_rate = targets.mean()
            print(f"[fit_correctness_lm_at] position={position} | device={device}")
            print(f"  correctness rate: {acc_rate:.4f}  (n_correct={int(targets.sum())}/{len(targets)})")
            print(f"  n_features: {n_features} | normalize_features: {normalize_features}")
            if class_weight is not None:
                print(f"  class_weight [incorrect, correct]: {np.round(class_weight, 3)}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            raise

        for fold_idx in range(self.kf.get_n_splits()):
            if verbose:
                print(f"  fold {fold_idx + 1}/{self.kf.get_n_splits()}")
            train_mask = df['fold'] != fold_idx
            val_mask = df['fold'] == fold_idx
            X_train_texts = combined_texts[train_mask.values]
            X_val_texts = combined_texts[val_mask.values]
            y_train = targets[train_mask.values]
            y_val = targets[val_mask.values]

            if feats_all is not None:
                f_train = feats_all[train_mask.values]
                f_val = feats_all[val_mask.values]
                if normalize_features:
                    f_train, f_val = standardize_train_val(f_train, f_val)
            else:
                f_train = f_val = None

            train_dataset = FeatureFusionDataset(
                X_train_texts, y_train, tokenizer, feats=f_train,
                max_length=max_length, multilabel=False)
            val_dataset = FeatureFusionDataset(
                X_val_texts, y_val, tokenizer, feats=f_val,
                max_length=max_length, multilabel=False)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            model = DeBERTaFusionHead(model_name, n_outputs, num_features=n_features, dropout_rate=0.1)

            if class_weight is not None:
                cw = torch.tensor(class_weight, dtype=torch.float, device=device)
                _orig_ce = nn.CrossEntropyLoss
                nn.CrossEntropyLoss = lambda *a, **k: _orig_ce(weight=cw)
            try:
                model = train_deberta_model(
                    model, train_loader, val_loader,
                    num_epochs=num_epochs, learning_rate=learning_rate, device=device,
                    multilabel=False, threshold=threshold)
            finally:
                if class_weight is not None:
                    nn.CrossEntropyLoss = _orig_ce

            probs = predict_deberta_proba(model, val_loader, n_outputs,
                                          device=device, multilabel=False)
            oof_preds[val_mask.values] = probs
            del model, train_dataset, val_dataset, train_loader, val_loader
            torch.cuda.empty_cache()

        df[score_col] = oof_preds[:, 0]
        if verbose:
            self._sanity_check_correctness_lm(position, score_col)
        return pd.DataFrame(oof_preds, index=df.index, columns=["p_correct", "p_incorrect"])

    def _sanity_check_correctness_lm(self, position, score_col):
        """Diagnostics for a post-hoc correctness/confidence head."""
        from sklearn.metrics import roc_auc_score
        from scipy import stats
        df = self.registry[position]
        y = (df[self.metric_col].values > 0).astype(int)
        s = df[score_col].values
        print("---- correctness-LM sanity check ----")
        print(f"  score '{score_col}': min={s.min():.3f} max={s.max():.3f} "
              f"mean={s.mean():.3f} std={s.std():.3f}")
        if s.std() < 1e-3:
            print("  !! WARNING: score has ~zero variance -> head collapsed. "
                  "Check class balance / LR / epochs.")
            return
        try:
            auroc = roc_auc_score(1 - y, s)
            print(f"  detector AUROC (P(incorrect) vs is-incorrect): {auroc:.4f}  (want > 0.5)")
        except ValueError:
            print("  AUROC undefined (only one class present).")
        tau = stats.kendalltau(s, y).statistic
        print(f"  Kendall tau (score vs correct): {tau:.4f}  (want negative)")
        print(f"  mean score | correct   samples: {s[y == 1].mean():.4f}")
        print(f"  mean score | incorrect samples: {s[y == 0].mean():.4f}")
        if len(self.axes_names) >= 1 and 'preferred_deferral_lm' not in df.columns:
            l = list(position); l[0] += 1
            df['preferred_deferral_lm'] = [tuple(l)] * len(df)
        try:
            res = self.full_threshold_sim_temp(score_col, pref_def_column='preferred_deferral_lm')
            print(f"  deferral AUC (full)={res['auc']:.4f} | "
                  f"AUC@20={res['auc_20']:.4f} | AUC@40={res['auc_40']:.4f}")
        except Exception as e:
            print(f"  (skipped deferral AUC probe: {e})")
        print("--------------------------------------")

    def resolve_full_deferred_old(self, from_position, pref_def_column='preferred_deferral'):
        idx = self.registry[from_position].index
        rows = []
        for i in idx:
            target_position = self.registry[from_position].loc[i][pref_def_column]
            rows.append(self.registry[target_position].loc[i])
        return pd.DataFrame(rows)

    def _axis_costs(self, position):
        """Full invocation cost of each one-step escalation destination."""
        costs = []
        for i, _ in enumerate(self.axes_names):
            pos = position[:i] + (position[i] + 1,) + position[i+1:]
            costs.append(self.cost_registry[pos])
        return np.array(costs, dtype=float)

    def set_cost_adjusted_pref(self, position, lam, gain=1.0, prob_cols=None,
                               keep_prob_col=None, cost_norm="max"):
        """Cost-adjusted deferral preference from per-axis recovery probabilities.

            utility_a = gain * (p_a - p_keep) - lam * c_a      (keep utility = 0)
        Route to argmax_a utility_a; defer only if that max > 0.
        """
        df = self.registry[position]
        if prob_cols is None:
            prob_cols = [f'post_hoc_lm_{ax}' for ax in self.axes_names]
        P = df[prob_cols].values
        p_keep = 0.0 if keep_prob_col is None else df[keep_prob_col].values[:, None]
        c = self._axis_costs(position)
        if cost_norm == "max" and c.max() > 0:
            c = c / c.max()
        elif cost_norm == "origin":
            c = c / self.cost_registry[self.origin]
        util = gain * (P - p_keep) - lam * c[None, :]
        best_axis = util.argmax(axis=1)
        best_util = util[np.arange(len(df)), best_axis]
        df['post_hoc_lm'] = best_util
        df['pref_axis_utility'] = best_util
        dests = []
        for row_i, ax_i in enumerate(best_axis):
            if best_util[row_i] <= 0:
                dests.append(tuple(position))
            else:
                l = list(position); l[ax_i] += 1
                dests.append(tuple(l))
        df['preferred_deferral_lm'] = dests
        return df[['preferred_deferral_lm', 'post_hoc_lm', 'pref_axis_utility']]

    def cost_adjusted_frontier(self, position=None, lams=None, gain=1.0,
                               keep_prob_col=None, cost_norm="max"):
        """Sweep lam to trace the cost-accuracy Pareto frontier and integrate AUC."""
        if position is None:
            position = self.origin
        if lams is None:
            lams = np.linspace(0, 5, 60)
        df0 = self.registry[self.origin]
        base_c = self.cost_registry[self.origin]
        pos_tuple = tuple(position)
        pts = []
        for lam in lams:
            self.set_cost_adjusted_pref(position, lam, gain=gain,
                                        keep_prob_col=keep_prob_col, cost_norm=cost_norm)
            resolved = self.resolve_full_deferred(position, 'preferred_deferral_lm')
            acc = resolved[self.metric_col].mean()
            dests = df0['preferred_deferral_lm']
            extra = np.array([0.0 if d == pos_tuple else self.cost_registry[d] for d in dests])
            spend = base_c + extra.mean()
            defer_frac = float((dests != pos_tuple).mean())
            pts.append((spend, acc, float(lam), defer_frac))
        pts.sort(key=lambda t: t[0])
        spend = [p[0] for p in pts]
        acc = [p[1] for p in pts]
        lam_out = [p[2] for p in pts]
        defer = [p[3] for p in pts]
        span = (spend[-1] - spend[0]) if spend[-1] > spend[0] else 1.0
        spend_norm = [(s - spend[0]) / span for s in spend]
        return {"spend": spend, "spend_norm": spend_norm, "acc": acc,
                "lams": lam_out, "defer_frac": defer,
                "cost_auc": np.trapezoid(acc, x=spend_norm)}

    def single_axis_frontier(self, axis_index, def_col, position=None):
        """(spend, acc) curve for a 1-axis cascade escalating only along axis_index."""
        if position is None:
            position = self.origin
        dest = position[:axis_index] + (position[axis_index] + 1,) + position[axis_index+1:]
        df = self.registry[self.origin]
        df_dest = self.registry[dest]
        base_c = self.cost_registry[self.origin]
        dest_c = self.cost_registry[dest]
        metric = self.metric_col
        thresh = np.linspace(-0.001, 1.0011, 200)
        ranks = df[def_col].rank(method='first') / len(df[def_col])
        spend, acc = [], []
        for t in thresh:
            defer_idx = ranks > (1 - t)
            p = defer_idx.mean()
            correct = df[metric][~defer_idx].sum() + df_dest[metric][defer_idx].sum()
            acc.append(correct / len(df))
            spend.append(base_c + p * dest_c)
        return {"spend": spend, "acc": acc}

    def compare_frontiers_audc(self, frontier_a, frontier_b, n_grid=200):
        """Compare two (spend, acc) frontiers on a common normalised-spend grid."""
        def envelope(front):
            s = np.asarray(front["spend"], float)
            a = np.asarray(front["acc"], float)
            order = np.argsort(s)
            s, a = s[order], a[order]
            a = np.maximum.accumulate(a)
            us, idx = np.unique(s, return_index=True)
            return us, a[idx]
        sa, aa = envelope(frontier_a)
        sb, ab = envelope(frontier_b)
        s_lo = max(sa.min(), sb.min())
        s_hi = min(sa.max(), sb.max())
        if s_hi <= s_lo:
            raise ValueError(f"No overlapping spend range (a:[{sa.min()},{sa.max()}] "
                             f"b:[{sb.min()},{sb.max()}]).")
        grid = np.linspace(s_lo, s_hi, n_grid)
        ga = np.interp(grid, sa, aa)
        gb = np.interp(grid, sb, ab)
        grid_norm = (grid - s_lo) / (s_hi - s_lo)
        audc_a = np.trapezoid(ga, x=grid_norm)
        audc_b = np.trapezoid(gb, x=grid_norm)
        gap = gb - ga
        return {"audc_a": audc_a, "audc_b": audc_b, "audc_delta": audc_b - audc_a,
                "frac_b_wins": float((gap >= 0).mean()),
                "mean_gap": float(gap.mean()), "max_gap": float(gap.max()),
                "min_gap": float(gap.min()), "s_lo": s_lo, "s_hi": s_hi,
                "grid": grid, "acc_a": ga, "acc_b": gb}

    def full_threshold_sim_temp(self, def_col, from_position=None,
                                pref_def_column='preferred_deferral',
                                metric_override=None, axis_fn=None):
        if metric_override:
            metric = metric_override
        else:
            metric = self.metric_col
        if from_position is None:
            from_position = self.origin

        df = self.registry[self.origin]
        df_large = self.resolve_full_deferred(from_position, pref_def_column=pref_def_column)

        # Per-row full destination cost for each row's preferred deferral target.
        # Post-hoc cost model: everyone pays origin; a deferred row additionally
        # pays the FULL cost of wherever it resolves to.
        base_c = self.cost_registry[from_position]
        row_dest_cost = np.array(
            [self.cost_registry[d] for d in df[pref_def_column].values], dtype=float)

        thresh = np.linspace(0 - .001, 1 + 0.0011, 200)
        accs = []
        n_deferred = []
        p_deferred = []
        accept_acc = []
        deferred_acc = []
        deferred_correct = []
        coverage = []
        spend = []
        ranks = df[def_col].rank(method='first') / len(df[def_col])
        for t in thresh:
            defer_idx = ranks > (1 - t)
            correct_current = df[metric][~defer_idx].sum()
            correct_large = df_large[metric][defer_idx].sum()
            accept_acc.append(df[metric][~defer_idx].mean())
            deferred_acc.append(df_large[metric][defer_idx].mean())
            deferred_correct.append(df_large[metric][defer_idx].sum())
            n_deferred.append(defer_idx.sum())
            p_deferred.append((defer_idx.sum() / len(df)))
            coverage.append((~defer_idx).mean())
            accs.append((correct_current + correct_large) / len(df))
            total_spend = base_c + (row_dest_cost * defer_idx.values).sum() / len(df)
            spend.append(total_spend)
            if len(p_deferred) > 1:
                if p_deferred[-2] < .2 and p_deferred[-1] >= .2:
                    p_del_20 = p_deferred.copy()
                    accs_20 = accs.copy()
                    p_del_20, accs_20 = misc.cap_interp_curve(p_del_20, accs_20, .2)
                    auc_20 = np.trapezoid(accs_20, x=p_del_20)
                if p_deferred[-2] < .4 and p_deferred[-1] >= .4:
                    p_del_40 = p_deferred.copy()
                    accs_40 = accs.copy()
                    p_del_40, accs_40 = misc.cap_interp_curve(p_del_40, accs_40, .4)
                    auc_40 = np.trapezoid(accs_40, x=p_del_40)

        spend_arr = np.asarray(spend, dtype=float)
        span = (spend_arr.max() - spend_arr.min()) if spend_arr.max() > spend_arr.min() else 1.0
        spend_norm = ((spend_arr - spend_arr.min()) / span).tolist()
        order = np.argsort(spend_arr)
        cost_auc = np.trapezoid(np.asarray(accs)[order], x=spend_arr[order])
        cost_auc_norm = np.trapezoid(np.asarray(accs)[order], x=np.asarray(spend_norm)[order])

        return {"p_deferred": p_deferred, "n_deferred": n_deferred,
                "deferred_correct": deferred_correct, "accepted_acc": accept_acc,
                "deferred_acc": deferred_acc, "accs": accs, "acc": accs,
                "auc": np.trapezoid(accs, x=p_deferred), "auc_20": auc_20, "auc_40": auc_40,
                "accs_20": accs_20, "accs_40": accs_40,
                "aurc": np.trapezoid(accept_acc, x=p_deferred),
                "p_del_20": p_del_20, "p_del_40": p_del_40,
                "spend": spend, "spend_norm": spend_norm,
                "cost_auc": cost_auc, "cost_auc_norm": cost_auc_norm}

    def run_simulation(self, router_fn):
        """router_fn: inspects current state and history -> (next_axis, next_pos) or None."""
        sim_results = []
        for prompt_id in self.origin.index:
            current_axis = 'origin'
            current_pos = 0
            history = [(current_axis, current_pos)]
            while True:
                current_row = self.registry[current_axis][current_pos].loc[prompt_id]
                decision = router_fn(current_row, history)
                if decision is None:
                    break
                next_axis, next_pos = decision
                if next_axis not in self.registry or next_pos not in self.registry[next_axis]:
                    break
                current_axis, current_pos = next_axis, next_pos
                history.append((current_axis, current_pos))
            final_row = self.registry[current_axis][current_pos].loc[prompt_id]
            sim_results.append({
                'prompt_id': prompt_id,
                'final_axis': current_axis,
                'final_score': final_row[self.metric_col],
                'path': " -> ".join([f"{a}({p})" for a, p in history]),
                'total_hops': len(history) - 1
            })
        return pd.DataFrame(sim_results)

    def find_optimal_path(self, threshold=0.9):
        """Diagnostic: for each prompt, the cheapest axis meeting the success threshold."""
        optimals = []
        for prompt_id in self.origin.index:
            found = False
            for axis in ['origin'] + self.axes_names:
                for pos in sorted(self.registry[axis].keys()):
                    score = self.registry[axis][pos].loc[prompt_id][self.metric_col]
                    if score >= threshold:
                        optimals.append({'prompt_id': prompt_id, 'cheapest_win': axis, 'score': score})
                        found = True
                        break
                if found:
                    break
            if not found:
                optimals.append({'prompt_id': prompt_id, 'cheapest_win': 'failure', 'score': 0})
        return pd.DataFrame(optimals)


def dummy_switch(row):
    return 0
