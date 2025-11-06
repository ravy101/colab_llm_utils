import numpy as np
from . import misc

def cascade_scored_samples(df, col, metric, ml_suffix='_13b', ):
    print(f"delegating by {col}")
    thresh = np.linspace(0- .001, 1 +0.001,100)
    accs = []
    n_delegated = []
    p_delegated = []
    accept_acc = []
    delegated_acc = []
    delegated_correct = []
    coverage = []
    gains = []
    small_correct = df[metric].mean()
    for t in thresh:
        delegate_idx = df[col] > (1-t)
        #delegate_idx = df_full[DELEGATE_COLUMN] <= t
        correct_7 = df[metric][~delegate_idx].sum()
        correct_70 = df[metric + ml_suffix][delegate_idx].sum()
        accept_acc.append(df[metric][~delegate_idx].mean())
        delegated_acc.append(df[metric + ml_suffix][delegate_idx].mean())
        delegated_correct.append(df[metric + ml_suffix][delegate_idx].sum())
        n_delegated.append(delegate_idx.sum())
        p_delegated.append((delegate_idx.sum()/len(df)))
        coverage.append((~delegate_idx).mean())
        accs.append((correct_7 + correct_70)/len(df))
        gains.append(accs[-1] - small_correct)
        if len(p_delegated) > 1:
            if p_delegated[-2] < .2 and p_delegated[-1] >= .2:
                p_del_20 = p_delegated.copy()
                accs_20 = accs.copy()
                p_del_20, accs_20 = misc.cap_interp_curve(p_del_20, accs_20, .2)
                auc_20 = np.trapezoid(accs_20, x= p_del_20)
            if p_delegated[-2] < .4 and p_delegated[-1] >= .4:
                p_del_40 = p_delegated.copy()
                accs_40 = accs.copy()
                p_del_40, accs_40 = misc.cap_interp_curve(p_del_40, accs_40, .4)
                auc_40 = np.trapezoid(accs_40, x= p_del_40)

    return {"p_delegated": p_delegated, "n_delegated":n_delegated, "delegated_correct": delegated_correct, "delegated_acc":delegated_acc, "accs": accs, "gains":gains, "auc": np.trapezoid(accs, x= p_delegated),
            "auc_20": auc_20, "auc_40": auc_40, "accs_20": accs_20, "accs_40": accs_40, "p_del_20": p_del_20, "p_del_40": p_del_40}

