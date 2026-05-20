import gc
import numpy as np
import pandas as pd
import re
from . import likelihood
from . import scorers
from . import text


def clean_mcq_strict(output_text, options_list):
    """
    Cleans LLM output strictly against the allowed option structure tokens.
    If the model outputs the right semantic answer but wrong format structure,
    it returns None (marking it incorrect for F1/Exact Match).
    
    Args:
        output_text (str): The raw string output from the local model.
        options_list (list): The valid tokens for this dataset (e.g., ['A', 'B', 'C', 'D'], 
                             ['1', '2', '3'], ['yes', 'no'], ['(0)', '(1)']).
    """
    if not output_text or not options_list:
        return "none"
        
    # Standardize input text
    text = output_text.strip()
    
    # 1. Escape options in case they contain regex characters like parentheses: (0), (1)
    escaped_options = [re.escape(str(opt)) for opt in options_list]
    options_pattern = "|".join(escaped_options)
    
    # 2. Look for "Answer: <valid_option>" (case-insensitive unless your options are case-sensitive)
    # Pattern matches: "Answer: A", "Answer: (0)", "Answer: yes"
    answer_prefix_pattern = rf'(?:answer\s*:\s*)({options_pattern})\b'
    prefix_match = re.search(answer_prefix_pattern, text, re.IGNORECASE)
    
    if prefix_match:
        # Return the exact matching token from your options list to preserve original casing/type
        matched_text = prefix_match.group(1).lower()
        for original_opt in options_list:
            if str(original_opt).lower() == matched_text:
                return original_opt

    # 3. Look for standalone option tokens at the start of lines or surrounded by boundaries
    # This catches "D\n\n", "A.\n\n", "D Supporter(s)..."
    for original_opt in options_list:
        opt_str = str(original_opt)
        # Create a strict boundary pattern for this specific token
        # Handles plain letters/numbers or tokens wrapped in punctuation
        opt_pattern = rf'^\s*{re.escape(opt_str)}(?:\b|\.|\s|\n|$)'
        if re.match(opt_pattern, text, re.IGNORECASE):
            return original_opt
            
    # If the model outputted "3 * 125" instead of 'C', it falls through here and returns "none"
    return "none"

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

  elif dataset_config['task_type'] == 'multiple_choice':
    for out, ans, question in zip(df['responses'], df['ans'], df['prompts']):
      targets = ans
      response = [clean_mcq_strict(out[0], dataset_config["options"])]
      print(response)

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

def columnize_meta_field(df, meta_field):
  metadata = df['meta'][meta_field]

def combine_dataframe(dfs):
  df =  pd.concat(dfs).reset_index(drop=True)
  return df
