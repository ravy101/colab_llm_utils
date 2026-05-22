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
from torch.optim import AdamW
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
    """Dataset for text classification with DeBERTa."""
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
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
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class DeBERTaClassificationHead(nn.Module):
    """DeBERTa model with classification head."""
    def __init__(self, model_name, num_classes, dropout_rate=0.1):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.deberta.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = pooled.float()
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


def train_deberta_model(model, train_loader, val_loader, num_epochs=3, learning_rate=2e-5, device='cpu'):
    """Train a DeBERTa classification model."""
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
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
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        scheduler.step()
        print(f"Finished epoch {epoch+1}")
        end = time.perf_counter()
        print(f"Iteration {epoch+1} took {end - start:0.4f} seconds")
        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- VALIDATION PHASE ---
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
                loss = criterion(logits, labels)
                total_val_loss += loss.item()
                
                # Get predicted class indices (highest logit)
                preds = torch.argmax(logits, dim=-1)
                
                # Move to CPU and convert to list for sklearn metric evaluation
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_loader)
        end = time.perf_counter()
        
        # --- METRIC CALCULATION ---
        # "macro" averaging works well for multi-class; change to "binary" if doing 2-class classification
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        accuracy = accuracy_score(all_labels, all_preds)
        
        # --- PERFORMANCE REPORT ---
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


def predict_deberta_proba(model, val_loader, num_classes, device='cpu'):
    """Get probability predictions from DeBERTa model."""
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            logits = model(input_ids, attention_mask)
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

    def fit_post_hoc_lm_at(self,
        position,
        input_text_col,
        output_text_col,
        model_name="microsoft/deberta-v3-small",
        num_epochs=3,
        batch_size=8,
        learning_rate=2e-5,
        max_length=512,
        device=None,
        verbose=True
    ):
        """
        Fits a DeBERTa model with classification head for post-hoc deferral prediction.
        Uses input and output text columns to predict the best model to defer to.
        Returns out-of-fold predictions for each row using K-fold CV.
        
        Args:
            position: Position in the cascade to fit at
            input_text_col: Column name for input text
            output_text_col: Column name for output text
            model_name: HuggingFace model identifier (default: microsoft/deberta-v3-small)
            num_epochs: Number of training epochs per fold
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            max_length: Max sequence length for tokenizer
            device: torch device (auto-detects GPU if available)
            verbose: Whether to print progress
            
        Returns:
            DataFrame with predicted probabilities for each class
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        df = self.registry[position]
        
        for c in [input_text_col, output_text_col]:
            if c not in list(df.columns):
                print(f"feature {c} not found in dataframe.")
                raise ValueError(f"Column {c} not found")
        
        # Combine input and output text
        combined_texts = (df[input_text_col].astype(str) + " [SEP] " + df[output_text_col].astype(str)).values
        
        # Generate targets
        target_dict = {0: df[self.metric_col]}
        deferral_options = {0: position}
        for i, _ in enumerate(self.axes_names):
            pos = copy.deepcopy(position)
            pos = pos[:i] + (pos[i] + 1,) + pos[i+1:]
            deferral_options[i+1] = pos
            target_dict[i+1] = self.registry[pos][self.metric_col]
        
        target_df = pd.DataFrame(target_dict)
        targets = target_df.apply(misc.biased_idxmax, axis=1).values
        #targets = target_df.idxmax(axis=1).values
        
        all_classes = np.arange(len(self.axes_names) + 1)
        n_classes = len(all_classes)
        oof_preds = np.zeros((len(df), n_classes))
        
        if verbose:
            print(f"target dict shape {target_df.shape}")
            print(f"targets: {pd.Series(targets).describe()}")
            print(f"Number of classes: {n_classes}")
            print(f"Device: {device}")
        
        # Initialize tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            raise
        
        # K-fold cross-validation
        for fold_idx in range(self.kf.get_n_splits()):
            if verbose:
                print(f"Training fold {fold_idx + 1}/{self.kf.get_n_splits()}")
            
            train_mask = df['fold'] != fold_idx
            val_mask = df['fold'] == fold_idx
            
            X_train_texts = combined_texts[train_mask]
            X_val_texts = combined_texts[val_mask]
            y_train = targets[train_mask]
            
            # Create datasets
            train_dataset = TextClassificationDataset(
                X_train_texts, y_train, tokenizer, max_length=max_length
            )
            val_dataset = TextClassificationDataset(
                X_val_texts, targets[val_mask], tokenizer, max_length=max_length
            )
            
            # Create dataloaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Initialize model for this fold
            model = DeBERTaClassificationHead(model_name, n_classes, dropout_rate=0.1)
            
            # Train model
            try:
                model = train_deberta_model(
                    model, train_loader, val_loader,
                    num_epochs=num_epochs,
                    learning_rate=learning_rate,
                    device=device
                )
            except Exception as e:
                print(f"Error during training on fold {fold_idx}: {e}")
                raise
            
            # Get predictions
            probs = predict_deberta_proba(model, val_loader, n_classes, device=device)
            
            # Align with all classes
            present_classes = np.arange(n_classes)
            aligned = np.zeros((len(X_val_texts), n_classes))
            aligned[:, present_classes] = probs
            
            oof_preds[val_mask] = aligned
            
            # Clean up to free memory
            del model, train_dataset, val_dataset, train_loader, val_loader
            torch.cuda.empty_cache()
        
        # Post-process predictions
        df['post_hoc_lm'] = oof_preds[:, 0]
        df['post_hoc_lm'] = 1 - df['post_hoc_lm']
        
        def_destinations = []
        for idx in oof_preds[:, 1:].argmax(axis=1):
            l = list(position)
            l[idx] = l[idx] + 1
            def_destinations.append(tuple(l))
        df['preferred_deferral_lm'] = def_destinations
        
        if verbose:
            print(f"Completed fit_post_hoc_lm_at at position {position}")
        
        return pd.DataFrame(oof_preds, index=df.index)

        

    def resolve_full_deferred(self, from_position, pref_def_column='preferred_deferral',):
        idx = self.registry[from_position].index
        rows = []
        for i in idx:
          target_position = self.registry[from_position].loc[i][pref_def_column]
          rows.append(self.registry[target_position].loc[i])
        return pd.DataFrame(rows)




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