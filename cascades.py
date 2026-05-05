import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

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
    def __init__(self, origin_df, axes_names, metric_col="gpt_score"):
        """
        Args:
            origin_df (pd.DataFrame): The small model's base inference.
            axes_names (list): Names of escalation paths (e.g., ['knowledge', 'reasoning', 'large']).
            metric_col (str): The column used to determine "success" (e.g., 'gpt_score').
        """
        self.metric_col = metric_col
        self.axes_names = axes_names
        
        # Ensure prompt_id is our alignment key
        if 'prompt_id' not in origin_df.columns and origin_df.index.name != 'prompt_id':
            raise ValueError("Dataframe must contain 'prompt_id' column or index.")
            
        self.origin = origin_df.set_index('prompt_id') if 'prompt_id' in origin_df.columns else origin_df
        
        # Internal registry: {axis_name: {position_int: dataframe}}
        self.registry = {name: {} for name in axes_names}
        self.registry['origin'] = {0: self.origin}

    def register_axis_data(self, df, axis_name, position):
        """Adds a dataframe for a specific point in the cascade grid."""
        if axis_name not in self.registry:
            raise ValueError(f"Axis '{axis_name}' not defined in constructor.")
            
        indexed_df = df.set_index('prompt_id') if 'prompt_id' in df.columns else df
        self.registry[axis_name][position] = indexed_df
        print(f"Registered {axis_name} (Pos: {position}) | Rows: {len(indexed_df)}")

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
