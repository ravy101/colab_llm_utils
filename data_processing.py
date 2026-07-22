import gc
import numpy as np
import pandas as pd
import re
from . import likelihood
from . import scorers
from . import text

def clean_mcq_strict(output_text, options_list):
    """
    Parse LLM MCQ outputs into one of the allowed option tokens.

    Preference:
      1. Explicit answer statements ("Answer:", "Final answer:", etc.)
      2. Last standalone option token in the text.
      3. "none" if nothing valid is found.
    """
    if not output_text or not options_list:
        return "none"

    txt = str(output_text).strip()

    # lookup preserving original option capitalisation
    lookup = {str(opt).lower(): opt for opt in options_list}

    escaped = [re.escape(str(opt)) for opt in options_list]
    options_pattern = "|".join(escaped)

    # standalone token boundary
    # allows: A, A., "A", (A), [A], A:, etc.
    token = rf'(?<![A-Za-z0-9])({options_pattern})(?![A-Za-z0-9])'

    # Explicit answer forms
    answer_pattern = (
        rf'(?i)'
        rf'(?:'
        rf'final\s+answer|'
        rf'correct\s+answer|'
        rf'answer'
        rf')'
        rf'\s*(?:is|should\s+be|=|:|-)?\s*["\']?'
        rf'({options_pattern})'
        rf'["\']?'
        rf'(?![A-Za-z0-9])'
    )

    matches = list(re.finditer(answer_pattern, txt))
    if matches:
        return lookup[matches[-1].group(1).lower()]

    # Otherwise return the LAST standalone option token
    matches = list(re.finditer(token, txt, flags=re.IGNORECASE))
    if matches:
        return lookup[matches[-1].group(1).lower()]

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

  graded_responses = []
  results = []
  results_bl = []
  results_em = []
  results_f1 = []
  if dataset_config['task_type'] == 'translation':
    for out, ans, question in zip(df['responses'], df['ans'], df['prompts']):
      graded_responses.append(out)
      results.append(metric_dict['meteor'].compute(predictions=out, references=[ans]))
      results_bl.append(metric_dict['bleurt'].compute(predictions=out, references=[ans]))
    df['scored_responses'] = graded_responses
    df['meteor'] = [r['meteor'] for r in results]
    df['bleurt'] = [r['scores'][0] for r in results_bl]
  elif dataset_config['task_type'] == 'qa':
    for out, ans, question in zip(df['responses'], df['ans'], df['prompts']):
      if dataset_config['dict_ans']:
        if dataset_config['clean_name'] == 'TruthfulQA' or dataset_config['clean_name'] == 'HotpotQA':
          targets = ans
        elif isinstance(ans, list):
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
      graded_responses.append(response)
      results.append(scorers.best_rouge_l(response, targets))
      results_em.append(scorers.best_em(response, targets))
      results_f1.append(scorers.best_f1(response, targets))
    df['scored_responses'] = graded_responses
    df['rouge'] = results
    df['em'] = results_em
    df['f1'] = results_f1

  elif dataset_config['task_type'] == 'multiple_choice':
    for out, ans, question in zip(df['responses'], df['ans'], df['prompts']):
      targets = ans
      response = [clean_mcq_strict(out[0], dataset_config["options"])]
      #print(response)
      graded_responses.append(response)
      results.append(scorers.best_rouge_l(response, targets))
      results_em.append(scorers.best_em(response, targets))
      results_f1.append(scorers.best_f1(response, targets))
    df['scored_responses'] = graded_responses
    df['rouge'] = results
    df['em'] = results_em
    df['f1'] = results_f1
  elif dataset_config['task_type'] == 'summarization':
    for out, ans in zip(df['responses'], df['ans']):
      graded_responses.append(out)
      results.append(scorers.best_rouge_l(out, ans))
    df['scored_responses'] = graded_responses
    df['rouge'] = results

  gc.collect()
  return df

def columnize_meta_field(df, meta_field):
  metadata = [m[meta_field] for m in df['meta']]
  for k in metadata[0].keys():
    df[meta_field+"-"+k] = [m[k] for m in metadata]

def combine_dataframe(dfs):
  df =  pd.concat(dfs).reset_index(drop=True)
  return df

def coerce_to_bounded_int(output_text, min_val=1, max_val=3):
    """
    Coerces raw text output into a strict integer between min_val and max_val.
    Returns -1 if the model outputs an invalid response, multiple choices, 
    or numbers outside the allowed boundary.
    """
    if not output_text:
        return -1
        
    text = output_text.strip()
    
    # 1. Catch comma-separated multi-selections or sequences first (e.g., "1,2,3")
    # If more than one digit exists in the output, it's ambiguous/invalid for an EM match
    all_digits = re.findall(r'\d', text)
    if len(set(all_digits)) > 1:
        return -1

    # 2. Extract standard integers using a prefix pattern or a standalone digit check
    # Catches: "Answer: 2", "2\n\n", "The score is 3."
    match = re.search(r'(?:answer\s*:\s*)?(\d+)', text, re.IGNORECASE)
    
    if match:
        val = int(match.group(1))
        # 3. Check boundaries (Strictly between 1 and 3)
        if min_val <= val <= max_val:
            return val
            
    return -1