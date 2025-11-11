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
    if np.isnan(result):
       result = 0
    return result

def log_chow_av(likelihoods):
    result = np.log(likelihoods).mean()
    if np.isnan(result):
       result = 0
    return result

def chow_prod_av(probas):
    result =  np.prod(probas)
    if np.isnan(result):
       result = 0
    return result

def chow_sum(likelihoods):
    result =  likelihoods.sum()
    if np.isnan(result):
       result = 0
    return result

def chow_quantile(likelihoods, alpha = .5):
    result =  np.quantile(likelihoods, alpha)
    if np.isnan(result):
       result = 0
    return result

def ll_to_proba(likelihoods):
    return np.exp(likelihoods)/np.sum(np.exp(likelihoods))

