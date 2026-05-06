import gc
import numpy as np
import pandas as pd
from . import likelihood
from . import scorers
from . import text


def process_dataframe(df, dataset_config, metric_dict, self_conf = False, p_true = False, thinking = False):
  df = df[df['responses'].str.len() >0]
  likes = []
  all_probas = []
  top_probs = []
  for l, t in zip(df['logit_outs'], df['token_outs']):
    gen_tokens = t[-len(l):]
    candidate_tokens = [list(ll.keys()) for ll in l]
    like_values = [list(ll.values()) for ll in l]
    all_prob = [likelihood.ll_to_proba(np.array(list(ll.values()))) for ll in l]
    all_probas.append(all_prob)

    token_likes = []
    token_probs = []
    # get the likelihood and proba for each SELECTED token (not top)
    for i, token in enumerate(gen_tokens):
      try:
        idx = candidate_tokens[i].index(token)
      except:
        print(f"missing index {token}")
        print(f"in candidates {candidate_tokens[i]}")
        idx = -1
      token_likes.append(like_values[i][idx])
      token_probs.append(all_prob[i][idx])
    likes.append(np.array(token_likes))
    top_probs.append(np.array(token_probs))
  df['likes'] = likes
  df['all_probas'] = all_probas
  df['top_probas'] = top_probs
  df['chow_av'] = [likelihood.chow_av(l) for l in df['top_probas']]
  df['chow_sum'] = [likelihood.chow_sum(l) for l in df['top_probas']]
  df['chow_quantile'] = [likelihood.chow_quantile(l) for l in df['top_probas']]
  df['log_chow_av'] = [likelihood.log_chow_av(l) for l in df['top_probas']]

  if thinking:
    splits = [text.split_tagged_text(a[0]) for a in df['responses']]
    df['thinking_text'] = [s[1] for s in splits]
    df['responses'] = [[s[0]] for s in splits]
  else:
    df['thinking_text'] = None

  if self_conf:
    self_conf_series = [m['self_conf'] for m in df['meta']]
    df['self_conf'] = self_conf_series

  if p_true:
    p_true_series = [m['p_true'] for m in df['meta']]
    df['p_true'] = p_true_series

  results = []
  results_bl = []
  results_em = []
  results_f1 = []
  if dataset_config['task_type'] == 'translation':
    for out, ans, question in zip(df['responses'], df['ans'], df['prompts']):
      results.append(metric_dict['meteor'].compute(predictions=out, references=[ans]))
      results_bl.append(metric_dict['bleurt'].compute(predictions=out, references=[ans]))
    df['meteor'] = [r['meteor'] for r in results]
    df['bleurt'] = [r['scores'][0] for r in results_bl]
  elif dataset_config['task_type'] == 'qa':
    for out, ans, question in zip(df['responses'], df['ans'], df['prompts']):
      if dataset_config['dict_ans']:
        if dataset_config['clean_name'] == 'TruthfulQA' or dataset_config['clean_name'] == 'HotpotQA':
          targets = ans
        else:
          targets = ans['normalized_aliases']
      elif dataset_config['clean_name'] == 'StrategyQA':
        if ans == True:
          targets = ["True", "true", "Yes", "yes"]
        elif ans == False:
          targets = ["False", "false", "No", "no"]
      else:
        targets = ans
      response = out

      results.append(scorers.best_rouge_l(response, targets))
      results_em.append(scorers.best_em(response, targets))
      results_f1.append(scorers.best_f1(response, targets))

    df['rouge'] = results
    df['em'] = results_em
    df['f1'] = results_f1
  elif dataset_config['task_type'] == 'summarization':
    for out, ans in zip(df['responses'], df['ans']):
      results.append(scorers.best_rouge_l(out, ans))
    df['rouge'] = results

  gc.collect()
  return df

def combine_dataframe(dfs):
  df =  pd.concat(dfs).reset_index(drop=True)
  return df
