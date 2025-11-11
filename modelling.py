import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score

def auc_kfold(X_tok, y, n_splits=5, random_state=0, model_constructor = LogisticRegression, model_args = {"max_iter": 1000}):
    """
    Compute AUC using k-fold cross-validation.

    X_tok: [n_samples, feature_dim]
    y: [n_samples] binary labels
    n_splits: number of folds
    Returns: aucs [max_len] — mean out-of-training AUC per token
    """
    n_samples, hidden_dim = X_tok.shape
    auc_matrix = np.full((n_splits), np.nan)
    acc_matrix = np.full((n_splits), np.nan)

    skf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    preds = y.copy()
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_tok, y)):
        X_train, X_test = X_tok[train_idx], X_tok[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = model_constructor(**model_args)
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:,1]
        acc_matrix[fold_idx] = (y_test == clf.predict(X_test)).mean()
        auc_matrix[fold_idx] = roc_auc_score(y_test, probs)
        preds[test_idx] = probs

    # mean over folds
    mean_acc = np.nanmean(acc_matrix, axis=0)
    mean_auc = np.nanmean(auc_matrix, axis=0)

    return mean_auc, mean_acc, preds

def regress_kfold(X_tok, y, n_splits=5, random_state=0, model_constructor = RandomForestRegressor, model_args = {"max_depth": 4, "n_estimators": 100}):
    """
    Compute MSE and R2 using k-fold cross-validation.

    X_tok: [n_samples, feature_dim]
    y: [n_samples] binary labels
    n_splits: number of folds
    Returns: aucs [max_len] — mean out-of-training AUC per token
    """

    r2_matrix = np.full((n_splits), np.nan)
    mse_matrix = np.full((n_splits), np.nan)

    skf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    preds = y.copy()
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_tok, y)):
        X_train, X_test = X_tok[train_idx], X_tok[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = model_constructor(**model_args)
        clf.fit(X_train, y_train)
        fold_preds = clf.predict(X_test)
        mse_matrix[fold_idx] = mean_squared_error(y_test, fold_preds)
        r2_matrix[fold_idx] = r2_score(y_test, fold_preds)
        preds[test_idx] = fold_preds

    # mean over folds
    mean_mse = np.nanmean(mse_matrix, axis=0)
    mean_r2 = np.nanmean(r2_matrix, axis=0)

    return mean_r2, mean_mse, preds



