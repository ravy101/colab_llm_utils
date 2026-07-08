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
    #print(f"deferring by {col}")
    thresh = np.linspace(0- .001, 1 +0.0011,200)
    accs = []
    n_deferred = []
    p_deferred = []
    accept_acc = []
    deferred_acc = []
    deferred_correct = []
    coverage = []
    gains = []
    small_correct = df[metric].mean()
    ranks = df[col].rank(method='first')/len(df[col])
    for t in thresh:
        defer_idx = ranks > (1-t)
        #defer_idx = df_full[defer_COLUMN] <= t
        correct_7 = df[metric][~defer_idx].sum()
        correct_70 = df[metric + ml_suffix][defer_idx].sum()
        accept_acc.append(df[metric][~defer_idx].mean())
        deferred_acc.append(df[metric + ml_suffix][defer_idx].mean())
        deferred_correct.append(df[metric + ml_suffix][defer_idx].sum())
        n_deferred.append(defer_idx.sum())
        p_deferred.append((defer_idx.sum()/len(df)))
        coverage.append((~defer_idx).mean())
        accs.append((correct_7 + correct_70)/len(df))
        gains.append(accs[-1] - small_correct)
        if len(p_deferred) > 1:
            if p_deferred[-2] < .2 and p_deferred[-1] >= .2:
                p_del_20 = p_deferred.copy()
                accs_20 = accs.copy()
                p_del_20, accs_20 = misc.cap_interp_curve(p_del_20, accs_20, .2)
                auc_20 = np.trapezoid(accs_20, x= p_del_20)
            if p_deferred[-2] < .4 and p_deferred[-1] >= .4:
                p_del_40 = p_deferred.copy()
                accs_40 = accs.copy()
                p_del_40, accs_40 = misc.cap_interp_curve(p_del_40, accs_40, .4)
                auc_40 = np.trapezoid(accs_40, x= p_del_40)

    return {"p_deferred": p_deferred, "n_deferred":n_deferred, "deferred_correct": deferred_correct, "accepted_acc": accept_acc, "deferred_acc":deferred_acc, "accs": accs, "gains":gains, 
            "auc": np.trapezoid(accs, x= p_deferred), "auc_20": auc_20, "auc_40": auc_40, "accs_20": accs_20, "accs_40": accs_40, 
            "aurc": np.trapezoid(accept_acc, x = p_deferred), "p_del_20": p_del_20, "p_del_40": p_del_40}



class TextClassificationDataset(Dataset):
    """Dataset for text classification / multilabel deferral with DeBERTa.

    labels may be either:
      - a scalar class index (multiclass mode), or
      - a 1-D vector of 0/1 floats with length == n_axes (multilabel mode).
    The dtype of the emitted label tensor is controlled by `multilabel`.
    """
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
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
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

class DeBERTaClassificationHead(nn.Module):
    """DeBERTa model with classification head."""
    def __init__(self, model_name, num_classes, dropout_rate=0.1):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.deberta.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        #pooled = outputs.last_hidden_state[:, 0, :]
        if torch.isnan(outputs).any():
            print("NaNs detected AFTER DeBERTa encoder!")
            print(f"inputs: {input_ids}")
            print(f"attention_mask: {attention_mask}")
            print("hidden state:")
            print(f"output shape: {outputs.shape}")
            print(f"outputs: {outputs}")

        pooled = outputs.mean(dim=1)
        if torch.isnan(pooled).any():
            print("NaNs detected AFTER pooling!")

            print("pooled stats:")
            print(f"min: {pooled.nanmin()}")
            print(f"max: {pooled.nanmax()}")

            nan_idx = torch.nonzero(torch.isnan(pooled))
            print("First pooled NaN index:", nan_idx[0])
        pooled = pooled.float()
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


def train_deberta_model(model, train_loader, val_loader, num_epochs=3, learning_rate=2e-5,
                        device='cpu', multilabel=False, pos_weight=None, threshold=0.5):
    """Train a DeBERTa head in either multiclass or multilabel mode.

    multilabel=False -> CrossEntropyLoss, argmax metrics (original behaviour).
    multilabel=True  -> BCEWithLogitsLoss (optional per-axis pos_weight),
                        sigmoid>threshold metrics (per-axis + micro/macro F1).
    The model returns raw logits in BOTH modes (no activation in forward).
    """
    optimizer = Adafactor(
        model.parameters(),
        lr=learning_rate,
        scale_parameter=False,
        relative_step=False
    )
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
            logits = model(input_ids, attention_mask)
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

                logits = model(input_ids, attention_mask)
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
                all_labels, all_preds, average='macro', zero_division=0
            )
            micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='micro', zero_division=0
            )
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
                all_labels, all_preds, average='macro', zero_division=0
            )
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
    """Get probability predictions from a DeBERTa head.

    multilabel=False -> softmax across classes (rows sum to 1).
    multilabel=True  -> independent sigmoid per axis (columns independent).
    Output shape is (n, num_classes) in both cases.
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            logits = model(input_ids, attention_mask)
            if multilabel:
                probs = torch.sigmoid(logits)
            else:
                probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    return np.vstack(all_probs)


def post_hoc_oof(
    df,
    feature_cols,
    target_col,
    n_splits=5,
    random_state=42,
    rf_kwargs=None,
    model_type = LogisticRegression
):
    """
    Returns out-of-fold predictions for each row in df using 5-fold CV.
    """

    for c in feature_cols:
        if c not in list(df.columns):
            print(f"feature {c} not found in dataframe.")
            feature_cols.remove(c)

    if rf_kwargs is None:
        rf_kwargs = {}

    X = df[feature_cols].values
    y = df[target_col].values

    oof_preds = np.zeros(len(df))

    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]

        model = model_type(
            random_state=random_state,
            **rf_kwargs
        )

        model.fit(X_train, y_train)
        oof_preds[val_idx] = model.predict_proba(X_val)[:,0]

    return pd.Series(oof_preds, index=df.index, name="oof_prediction")

def post_hoc_oof_cont(
    df,
    feature_cols,
    target_col,
    n_splits=5,
    random_state=42,
    rf_kwargs=None,
    model_type = RandomForestRegressor
):
    """
    Returns out-of-fold predictions for each row in df using 5-fold CV.
    """

    for c in feature_cols:
        if c not in list(df.columns):
            print(f"feature {c} not found in dataframe.")
            feature_cols.remove(c)

    if rf_kwargs is None:
        rf_kwargs = {}

    X = df[feature_cols].values
    y = df[target_col].values

    oof_preds = np.zeros(len(df))

    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]

        model = model_type(
            random_state=random_state,
            **rf_kwargs
        )

        model.fit(X_train, y_train)
        oof_preds[val_idx] = model.predict(X_val)

    return pd.Series(oof_preds, index=df.index, name="oof_prediction")


class MultiaxialCascade:
    def __init__(self, origin_df, axes_names, metric_col="gpt_score", fill_undefined = True, k=4, seed=42):

        self.metric_col = metric_col
        self.axes_names = axes_names
            
        self.origin = (0,) * len(axes_names)
        
        # Internal registry: {tuple(int, int): dataframe}
        
        self.registry = {self.origin: origin_df}
        self.cost_registry = {self.origin: 1}

        # model registry {tuple(int, int): List<Model>}
        self.model_registry = {}
        self.kf = KFold(
        n_splits=k,
        shuffle=True,
        random_state=seed
    )

    def register_axis_data(self, df, position, cost):
        """Adds a dataframe for a specific point in the cascade grid."""
        if len(position) != len(self.axes_names):
            raise ValueError(f"Position invalid, expexted {len(self.axes_names)} dimensions.")
            
        self.registry[position] = df
        self.cost_registry[position] = cost
        print(f"Registered {[ax + ": " + str(position[i]) for i, ax in enumerate(self.axes_names)]} | Shape: {df.shape} | Cost: {cost}")
        self.normalize_dfs()


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

    
    def set_pref_deferral_at(self, position, column, offset=-1):
        self.registry[position]["preferred_deferral"] = self.registry[position][column] + offset

    def fit_post_hoc_at(self,
    position,
    feature_cols,
    rf_kwargs=None,
    model_type = RandomForestClassifier
):
        """
        Returns out-of-fold predictions for each row in df using 5-fold CV.
        """
        df = self.registry[position]

        for c in feature_cols:
            if c not in list(df.columns):
                print(f"feature {c} not found in dataframe.")
                feature_cols.remove(c)

        if rf_kwargs is None:
            rf_kwargs = {}

        X = df[feature_cols].values

        target_dict = {0:df[self.metric_col]}
        # setup targets
        deferral_options = {0:position} # 0 index is keep for this model
        for i, _ in enumerate(self.axes_names):
            pos = copy.deepcopy(position)
            pos = pos[:i] + (pos[i] + 1,) + pos[i+1:]

            #pos[i] = pos[i] + 1
            deferral_options[i+1] = pos
            target_dict[i+1] = self.registry[pos][self.metric_col]


        target_df = pd.DataFrame(target_dict)

        print(f"target dict shape {target_df.shape}")
        targets = target_df.apply(misc.biased_idxmax, axis=1)
        #targets = target_df.idxmax(axis=1)
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

            model = model_type(
                **rf_kwargs
            )

            model.fit(X_train, y_train)
            probs = model.predict_proba(X_val)
            present_classes = model.classes_

            aligned = np.zeros((len(X_val), n_classes))
            aligned[:, present_classes] = probs

            oof_preds[val_idx] = aligned
        df['post_hoc'] = oof_preds[:,0]
        df['post_hoc'] = 1 - df['post_hoc']
        def_destinations = []
        for idx in oof_preds[:, 1:].argmax(axis=1):
          l = list(position)
          l[idx] = l[idx] + 1
          def_destinations.append(tuple(l))
        df['preferred_deferral']  = def_destinations
        return pd.DataFrame(oof_preds, index=df.index)

    def set_oracle_pref_deferral_at(self, position, tie_breaker=misc.biased_idxmax, allow_keep=True):
        """
        Sets the oracle (ground-truth) preferred deferral destination for each row
        at `position`, based on the actual metric outcomes at each single-axis
        escalation from `position`.

        For each row, considers:
            - keeping at `position` (class 0)
            - escalating one step along each registered axis (classes 1..N)
        and picks the option with the highest realised metric, breaking ties via
        `tie_breaker` (defaults to misc.biased_idxmax, consistent with
        fit_post_hoc_at).

        Writes three columns to self.registry[position]:
            - 'oracle_pref_idx'        : int in [0, len(axes_names)],
                                        0 = keep, i = escalate along axis i-1
            - 'oracle_pref_axis'       : 'keep' or self.axes_names[i-1]
            - 'oracle_preferred_deferral' : tuple position to defer to
                                            (equals `position` when keep is best)
        """
        if position not in self.registry:
            raise KeyError(f"Position {position} not registered.")

        df = self.registry[position]

        # Build target frame: column 0 = stay, columns 1..N = escalate along each axis
        
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
                    f"not registered; cannot compute oracle preference."
                )
            target_dict[i + 1] = self.registry[pos][self.metric_col].values
            deferral_options[i + 1] = pos

        target_df = pd.DataFrame(target_dict, index=df.index)

        # Pick the best option per row (with tie-breaking consistent with the rest
        # of the class). Result is an int label in {0, 1, ..., len(axes_names)}.
        oracle_idx = target_df.apply(tie_breaker, axis=1).astype(int)

        df['oracle_pref_idx'] = oracle_idx.values
        df['oracle_preferred_deferral'] = oracle_idx.map(deferral_options).values

        return df[['oracle_pref_idx', 'oracle_preferred_deferral']]

    def fit_post_hoc_lm_at(self,
        position,
        input_text_col,
        output_text_col,
        model_name="microsoft/deberta-v3-small",
        num_epochs=3,
        batch_size=8,
        learning_rate=2e-5,
        max_length=512,
        multilabel=False,
        recovery_fn=None,
        use_pos_weight=True,
        threshold=0.5,
        simple_def_col=None,
        target_func=misc.basic_idxmax,
        device=None,
        verbose=True
    ):
        """
        Fit a DeBERTa head for post-hoc deferral prediction, in either
        multiclass (single best axis) or multilabel (per-axis recovery) mode.

        multilabel=False (default): preserves original behaviour exactly.
            target = target_func over [keep, axis_1..N] (or axes only if
            simple_def_col is set); softmax + CrossEntropy; argmax routing.

        multilabel=True: one independent binary target per axis = "does
            escalating one step along this axis recover the sample?".
            sigmoid + BCEWithLogitsLoss; per-axis probabilities written back.
            `recovery_fn(base_score, axis_score) -> {0,1} array` defines
            recovery; defaults to (axis_score > base_score).
            simple_def_col / target_func are ignored in this mode.

        Returns a DataFrame of OOF predictions:
            multiclass  -> columns [keep, axis_1..N]  (simplex rows)
            multilabel  -> columns [axis_1..N]        (independent sigmoids)
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

        # ---- neighbour positions (one step along each axis) ----
        axis_positions = []
        for i, _ in enumerate(self.axes_names):
            pos = copy.deepcopy(position)
            pos = pos[:i] + (pos[i] + 1,) + pos[i+1:]
            if pos not in self.registry:
                raise KeyError(f"Required neighbour {pos} (axis '{self.axes_names[i]}') "
                               f"not registered.")
            axis_positions.append(pos)

        # ============================ TARGET BUILD ============================
        if multilabel:
            if recovery_fn is None:
                recovery_fn = lambda base, axis: (axis > base).astype(np.float32)

            base_score = df[self.metric_col].values
            label_cols = []
            for pos in axis_positions:
                axis_score = self.registry[pos][self.metric_col].values
                label_cols.append(recovery_fn(base_score, axis_score).astype(np.float32))
            targets = np.stack(label_cols, axis=1)          # (n, n_axes) float
            n_outputs = len(self.axes_names)

            # per-axis pos_weight = n_neg / n_pos (clamped to avoid inf)
            pos_weight = None
            if use_pos_weight:
                pos = targets.sum(axis=0)
                neg = targets.shape[0] - pos
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

        oof_preds = np.zeros((len(df), n_outputs))

        if verbose:
            print(f"mode: {'multilabel' if multilabel else 'multiclass'}")
            print(f"n_outputs: {n_outputs} | device: {device}")
            if multilabel:
                rates = targets.mean(axis=0)
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

        # ============================ K-FOLD ============================
        for fold_idx in range(self.kf.get_n_splits()):
            if verbose:
                print(f"Training fold {fold_idx + 1}/{self.kf.get_n_splits()}")

            train_mask = df['fold'] != fold_idx
            val_mask = df['fold'] == fold_idx

            X_train_texts = combined_texts[train_mask.values]
            X_val_texts = combined_texts[val_mask.values]
            y_train = targets[train_mask.values]
            y_val = targets[val_mask.values]

            train_dataset = TextClassificationDataset(
                X_train_texts, y_train, tokenizer, max_length=max_length, multilabel=multilabel
            )
            val_dataset = TextClassificationDataset(
                X_val_texts, y_val, tokenizer, max_length=max_length, multilabel=multilabel
            )

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            model = DeBERTaClassificationHead(model_name, n_outputs, dropout_rate=0.1)
            self.model_registry[position] = model
            try:
                model = train_deberta_model(
                    model, train_loader, val_loader,
                    num_epochs=num_epochs,
                    learning_rate=learning_rate,
                    device=device,
                    multilabel=multilabel,
                    pos_weight=pos_weight,
                    threshold=threshold
                )
            except Exception as e:
                print(f"Error during training on fold {fold_idx}: {e}")
                raise

            probs = predict_deberta_proba(model, val_loader, n_outputs,
                                          device=device, multilabel=multilabel)

            if multilabel:
                # all axes always emitted; no class alignment needed
                oof_preds[val_mask.values] = probs
            else:
                present_classes = np.arange(n_outputs)
                aligned = np.zeros((len(X_val_texts), n_outputs))
                aligned[:, present_classes] = probs
                oof_preds[val_mask.values] = aligned

            del model, train_dataset, val_dataset, train_loader, val_loader
            torch.cuda.empty_cache()

        # ============================ WRITE-BACK ============================
        if multilabel:
            # per-axis recovery probabilities
            for i, ax in enumerate(self.axes_names):
                df[f'post_hoc_lm_{ax}'] = oof_preds[:, i]
            # defer-or-not score = best available axis recovers it
            df['post_hoc_lm'] = oof_preds.max(axis=1)
            # preferred axis = most-likely-to-recover axis
            best_axis = oof_preds.argmax(axis=1)
            def_destinations = []
            for idx in best_axis:
                l = list(position); l[idx] = l[idx] + 1
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
                l = list(position); l[idx] = l[idx] + 1
                def_destinations.append(tuple(l))
            df['preferred_deferral_lm'] = def_destinations

        if verbose:
            print(f"Completed fit_post_hoc_lm_at at position {position} "
                  f"(multilabel={multilabel})")

        return pd.DataFrame(oof_preds, index=df.index)

        

    def resolve_full_deferred(self, from_position, pref_def_column='preferred_deferral',):
        idx = self.registry[from_position].index
        rows = []
        for i in idx:
          target_position = self.registry[from_position].loc[i][pref_def_column]
          rows.append(self.registry[target_position].loc[i])
        return pd.DataFrame(rows)

    def _axis_costs(self, position):
        """Full invocation cost of each one-step escalation destination.

        Post-hoc cascade: the origin has already run, so its cost is sunk and
        each escalation is charged the FULL destination cost (not dest - origin).
        Returns array shape (n_axes,).
        """
        costs = []
        for i, _ in enumerate(self.axes_names):
            pos = position[:i] + (position[i] + 1,) + position[i+1:]
            costs.append(self.cost_registry[pos])
        return np.array(costs, dtype=float)


    def set_cost_adjusted_pref(self, position, lam, gain=1.0,
                               prob_cols=None, keep_prob_col=None,
                               cost_norm="max"):
        """Cost-adjusted deferral preference from per-axis recovery probabilities.

            utility_a = gain * (p_a - p_keep) - lam * c_a      (keep utility = 0)

        Route to argmax_a utility_a; defer only if that max > 0.

        Args:
            position     : cascade position to decide at.
            lam          : cost-aversion / budget dual (0 = pure accuracy argmax).
            gain         : value of a unit of accuracy (1.0 for 0/1 metric).
            prob_cols    : per-axis recovery-prob columns
                           (default: post_hoc_lm_<axis>).
            keep_prob_col: column giving p_keep = P(origin answer correct).
                           If None, p_keep is treated as 0 (base-failure-conditioned
                           form; correct when probs were trained on "recovers a
                           base failure").
            cost_norm    : "max" -> divide costs by max axis cost;
                           "origin" -> divide by origin cost (reads as x-origin);
                           None -> raw costs.

        Writes to self.registry[position]:
            preferred_deferral_lm : tuple destination (== position when keep wins)
            post_hoc_lm           : max utility margin (ranking score for sweeps)
            pref_axis_utility     : chosen utility value
        Returns those three columns.
        """
        df = self.registry[position]

        if prob_cols is None:
            prob_cols = [f'post_hoc_lm_{ax}' for ax in self.axes_names]
        P = df[prob_cols].values                       # (n, n_axes)

        p_keep = 0.0 if keep_prob_col is None else df[keep_prob_col].values[:, None]

        c = self._axis_costs(position)                 # (n_axes,)
        if cost_norm == "max" and c.max() > 0:
            c = c / c.max()
        elif cost_norm == "origin":
            c = c / self.cost_registry[self.origin]

        util = gain * (P - p_keep) - lam * c[None, :]  # (n, n_axes)
        best_axis = util.argmax(axis=1)
        best_util = util[np.arange(len(df)), best_axis]

        df['post_hoc_lm'] = best_util
        df['pref_axis_utility'] = best_util

        dests = []
        for row_i, ax_i in enumerate(best_axis):
            if best_util[row_i] <= 0:                  # keep dominates -> no defer
                dests.append(tuple(position))
            else:
                l = list(position); l[ax_i] += 1
                dests.append(tuple(l))
        df['preferred_deferral_lm'] = dests

        return df[['preferred_deferral_lm', 'post_hoc_lm', 'pref_axis_utility']]


    def cost_adjusted_frontier(self, position=None, lams=None, gain=1.0,
                               keep_prob_col=None, cost_norm="max"):
        """Sweep lam to trace the cost-accuracy Pareto frontier and integrate AUC.

        For each lam: set cost-adjusted preferences, resolve deferrals, and record
            acc   = mean metric of the resolved (post-routing) answers
            spend = origin cost (paid by all) + full destination cost (deferred only)

        Returns dict with per-lam spend/acc/defer_frac and the cost-AUC
        (accuracy integrated over normalised spend).
        """
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
                                        keep_prob_col=keep_prob_col,
                                        cost_norm=cost_norm)
            resolved = self.resolve_full_deferred(position, 'preferred_deferral_lm')
            acc = resolved[self.metric_col].mean()

            dests = df0['preferred_deferral_lm']
            extra = np.array([0.0 if d == pos_tuple else self.cost_registry[d]
                              for d in dests])
            spend = base_c + extra.mean()
            defer_frac = float((dests != pos_tuple).mean())
            pts.append((spend, acc, float(lam), defer_frac))

        pts.sort(key=lambda t: t[0])                   # order by spend
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
        """(spend, acc) curve for a 1-axis cascade escalating only along `axis_index`.

        Sweeps the deferral rank-threshold on `def_col` (e.g. an origin-confidence /
        post_hoc score). Post-hoc cascade cost model: everyone pays origin; deferred
        samples additionally pay the full single-destination cost.
        """
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
        """Compare two (spend, acc) frontiers on a common normalised-spend grid.

        frontier_a = reference (e.g. best single-axis cascade)
        frontier_b = candidate (e.g. two-axis cost_adjusted_frontier)

        Both dicts must have 'spend' and 'acc'. Curves are made monotone-in-spend
        by upper-envelope (best accuracy achievable at <= that spend), interpolated
        onto the OVERLAPPING spend range, then compared.

        Returns:
            audc_a, audc_b : area under each curve over the common grid
            audc_delta     : audc_b - audc_a  (>0 => B dominates in area)
            frac_b_wins    : fraction of grid where acc_b >= acc_a (1.0 => strict Pareto)
            mean_gap, max_gap : acc_b - acc_a summary over the grid
            s_lo, s_hi     : common spend range compared
        """
        def envelope(front):
            s = np.asarray(front["spend"], float)
            a = np.asarray(front["acc"], float)
            order = np.argsort(s)
            s, a = s[order], a[order]
            a = np.maximum.accumulate(a)          # Pareto upper envelope
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

    def full_threshold_sim_temp(self, def_col, from_position = None, pref_def_column='preferred_deferral', metric_override = None, axis_fn = None):
        #print(f"deferring by {col}")
        if metric_override:
            metric = metric_override
        else:
            metric = self.metric_col

        if from_position is None:
          from_position = self.origin
          
        df = self.registry[self.origin]
        df_large = self.resolve_full_deferred(from_position, pref_def_column=pref_def_column)
        thresh = np.linspace(0- .001, 1 +0.0011,200)
        accs = []
        n_deferred = []
        p_deferred = []
        accept_acc = []
        deferred_acc = []
        deferred_correct = []
        coverage = []
        ranks = df[def_col].rank(method='first')/len(df[def_col])
        for t in thresh:
            defer_idx = ranks > (1-t)
            correct_current = df[metric][~defer_idx].sum()
            correct_large = df_large[metric][defer_idx].sum()
            accept_acc.append(df[metric][~defer_idx].mean())
            deferred_acc.append(df_large[metric][defer_idx].mean())
            deferred_correct.append(df_large[metric][defer_idx].sum())
            n_deferred.append(defer_idx.sum())
            p_deferred.append((defer_idx.sum()/len(df)))
            coverage.append((~defer_idx).mean())
            accs.append((correct_current + correct_large)/len(df))

            # switch fn = return 0??

            if len(p_deferred) > 1:
                if p_deferred[-2] < .2 and p_deferred[-1] >= .2:
                    p_del_20 = p_deferred.copy()
                    accs_20 = accs.copy()
                    p_del_20, accs_20 = misc.cap_interp_curve(p_del_20, accs_20, .2)
                    auc_20 = np.trapezoid(accs_20, x= p_del_20)
                if p_deferred[-2] < .4 and p_deferred[-1] >= .4:
                    p_del_40 = p_deferred.copy()
                    accs_40 = accs.copy()
                    p_del_40, accs_40 = misc.cap_interp_curve(p_del_40, accs_40, .4)
                    auc_40 = np.trapezoid(accs_40, x= p_del_40)

        return {"p_deferred": p_deferred, "n_deferred":n_deferred, "deferred_correct": deferred_correct, "accepted_acc": accept_acc, "deferred_acc":deferred_acc, "accs": accs, 
                "auc": np.trapezoid(accs, x= p_deferred), "auc_20": auc_20, "auc_40": auc_40, "accs_20": accs_20, "accs_40": accs_40, 
                "aurc": np.trapezoid(accept_acc, x = p_deferred), "p_del_20": p_del_20, "p_del_40": p_del_40}



    def run_simulation(self, router_fn):
        """
        router_fn: Logic that inspects current state and history to return 
                   (next_axis_name, next_position) or None.
        """
        sim_results = []

        for prompt_id in self.origin.index:
            current_axis = 'origin'
            current_pos = 0
            history = [(current_axis, current_pos)]
            
            while True:
                # Get the row from the current active dataframe
                current_row = self.registry[current_axis][current_pos].loc[prompt_id]
                
                # Decision point: where to go next?
                decision = router_fn(current_row, history)
                
                if decision is None:
                    break
                
                next_axis, next_pos = decision
                
                # Safety check for exhaustive data availability
                if next_axis not in self.registry or next_pos not in self.registry[next_axis]:
                    # This happens if the router tries to escalate to a data tier you didn't load
                    break
                
                current_axis, current_pos = next_axis, next_pos
                history.append((current_axis, current_pos))
            
            # Record final outcome of the walk
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
        """
        Diagnostic tool: For each prompt, identifies the cheapest axis 
        that meets the success threshold.
        """
        optimals = []
        for prompt_id in self.origin.index:
            found = False
            # We check in order of expected cost: Origin -> Knowledge -> Reasoning -> Large
            for axis in ['origin'] + self.axes_names:
                for pos in sorted(self.registry[axis].keys()):
                    score = self.registry[axis][pos].loc[prompt_id][self.metric_col]
                    if score >= threshold:
                        optimals.append({'prompt_id': prompt_id, 'cheapest_win': axis, 'score': score})
                        found = True
                        break
                if found: break
            if not found:
                optimals.append({'prompt_id': prompt_id, 'cheapest_win': 'failure', 'score': 0})
        return pd.DataFrame(optimals)

def dummy_switch(row):
    return 0