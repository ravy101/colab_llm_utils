import numpy as np
from . import misc

def cascade_scored_samples(df, col, metric, ml_suffix='_13b'):
    print(f"deferring by {col}")
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
    for t in thresh:
        defer_idx = df[col] > (1-t)
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

    return {"p_deferred": p_deferred, "n_deferred":n_deferred, "deferred_correct": deferred_correct, "deferred_acc":deferred_acc, "accs": accs, "gains":gains, "auc": np.trapezoid(accs, x= p_deferred),
            "auc_20": auc_20, "auc_40": auc_40, "accs_20": accs_20, "accs_40": accs_40, "p_del_20": p_del_20, "p_del_40": p_del_40}

