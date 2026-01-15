import numpy as np

def b_entropy(p1, p2):
  return -(p1 * np.log(p1) + p2 * np.log(p2))

def gen_entropy(log_likes):
  likelihoods = np.exp(log_likes)
  probas = likelihoods / np.sum(likelihoods)
  sum = 0
  for p in probas:
    sum = sum - p * np.log(p)
  return sum

def embedding_entropy(probabilities, embeddings):
  sum = 0
  for i in range(len(probabilities)):
    sum = sum - probabilities[i] * b_entropy(embeddings)
  return sum

def softmax_from_loglik(logliks):
    logliks = np.array(logliks, dtype=np.float64)
    max_ll = np.max(logliks)              # for numerical stability
    exp_shifted = np.exp(logliks - max_ll)
    probs = exp_shifted / np.sum(exp_shifted)
    return probs

def chow_av(likelihoods):
    result = likelihoods.mean()
    if np.isnan(result) or np.isinf(result):
       result = 0
    return result

def log_chow_av2(probas):
    result = np.log(probas).mean()
    if np.isnan(result) or np.isinf(result):
       result = 0
    return result

def log_chow_av(probas, eps = 1e-12):
    probas = np.asarray(probas)
    probas = np.clip(probas, eps, 1.0)
    return np.mean(np.log(probas))

def chow_prod_av(probas):
    result =  np.prod(probas)
    if np.isnan(result) or np.isinf(result):
       result = 0
    return result

def chow_sum(likelihoods):
    result =  likelihoods.sum()
    if np.isnan(result) or np.isinf(result):
       result = 0
    return result

def chow_quantile(likelihoods, alpha = .5):
    result =  np.quantile(likelihoods, alpha)
    if np.isnan(result) or np.isinf(result):
       result = 0
    return result

def ll_to_proba(likelihoods):
    return np.exp(likelihoods)/np.sum(np.exp(likelihoods))

def chow_cvar_uncertainty(
    probas,
    eps=1e-12,
    min_alpha=0.05,
    max_alpha=0.5,
):
    """
    Best-of-both-worlds uncertainty score for open generation.

    Args:
        probas (array-like): token probabilities for the generated sequence
        eps (float): numerical stability for log
        min_alpha (float): minimum CVaR tail fraction
        max_alpha (float): maximum CVaR tail fraction

    Returns:
        float: uncertainty score (larger = worse / more uncertain)
    """

    # --- 1. Convert probabilities to NLLs ---
    probas = np.asarray(probas, dtype=np.float64)
    probas = np.clip(probas, eps, 1.0)
    nll = -np.log(probas)

    T = nll.size
    if T == 0:
        return 0.0

    # --- 2. Chow average (mean NLL) ---
    chow = np.mean(nll)

    # --- 3. Adaptive alpha via tail gap (knee) detection ---
    sorted_nll = np.sort(nll)

    # compute gaps in upper half only
    start = T // 2
    gaps = sorted_nll[start+1:] - sorted_nll[start:-1]

    if gaps.size == 0:
        alpha = min_alpha
    else:
        i_rel = np.argmax(gaps)
        i_star = start + i_rel
        tail_len = T - i_star - 1
        alpha = tail_len / T
        alpha = np.clip(alpha, min_alpha, max_alpha)

    # --- 4. CVaR computation ---
    k = max(1, int(np.ceil(alpha * T)))
    cvar = np.mean(sorted_nll[-k:])

    # --- 5. Adaptive lambda based on dispersion ---
    # normalize variance to (0,1) using a soft saturation
    var = np.var(nll)
    lambda_ = var / (var + 1.0)

    # --- 6. Final blended score ---
    uncertainty = (1.0 - lambda_) * chow + lambda_ * cvar

    return float(uncertainty)