import gc
import numpy as np
import pandas as pd
import re
from . import likelihood
from . import scorers
from . import text
import ast
import string

# ------------------------------------------------------------------------------
# 1. Normalized MCQ Evaluator (Fixes HellaSwag '1' -> 'B' and Qwen 'B.' prefixes)
# ------------------------------------------------------------------------------
def evaluate_mcq_robust(response_text, gold_answer, options_list=["A", "B", "C", "D", "E"]):
    if not response_text:
        return 0

    # A. Normalize Gold Answer (handles integer 1 -> 'B', '2' -> 'C')
    gold_str = str(gold_answer).strip()
    if gold_str.isdigit():
        idx = int(gold_str)
        if 0 <= idx < len(options_list):
            gold_str = options_list[idx]

    gold_str = gold_str.upper()

    # B. Clean response text
    txt = str(response_text).strip()

    # C. Match explicit 'Answer: B' or 'The correct answer is B'
    explicit = re.search(r'(?:correct\s+answer|answer|choice|option)\s*(?:is|:|=)?\s*["\']?([A-E])["\']?', txt, re.IGNORECASE)
    if explicit:
        pred = explicit.group(1).upper()
        return 1 if pred == gold_str else 0

    # D. Match FIRST standalone letter token (e.g., ' B\n', 'B.', 'B)')
    tokens = re.findall(r'(?<![A-Za-z0-9])([A-E])(?![A-Za-z0-9])', txt)
    if tokens:
        first_pred = tokens[0].upper()
        if first_pred == gold_str:
            return 1

    return 0


# ------------------------------------------------------------------------------
# 2. Robust Math / GSM8K Evaluator (Extracts numbers from CoT & \boxed{})
# ------------------------------------------------------------------------------
def evaluate_gsm8k_robust(response_text, gold_answer):
    if not response_text:
        return 0

    txt = str(response_text).strip()
    gold_str = str(gold_answer).strip()

    # Extract clean target number from gold string
    gold_nums = re.findall(r'[-+]?\d+(?:\.\d+)?', gold_str.replace(',', ''))
    if not gold_nums:
        return 0
    gold_val = gold_nums[-1]

    # A. Check \boxed{16} format
    boxed_match = re.search(r'\\boxed\{([-+]?\d+(?:\.\d+)?)\}', txt)
    if boxed_match:
        if boxed_match.group(1).replace(',', '') == gold_val:
            return 1

    # B. Check #### 16 format
    if '####' in txt:
        tail = txt.split('####')[-1]
        tail_nums = re.findall(r'[-+]?\d+(?:\.\d+)?', tail.replace(',', ''))
        if tail_nums and tail_nums[-1] == gold_val:
            return 1

    # C. Check 'Answer: 16' or 'is 16'
    ans_match = re.search(r'(?:answer|equals?|total|is)\s*[:=]?\s*\$?\s*([-+]?\d+(?:\.\d+)?)', txt, re.IGNORECASE)
    if ans_match:
        if ans_match.group(1).replace(',', '') == gold_val:
            return 1

    # D. Fallback: Check the VERY LAST number in the generated text
    all_nums = re.findall(r'[-+]?\d+(?:\.\d+)?', txt.replace(',', ''))
    if all_nums and all_nums[-1] == gold_val:
        return 1

    return 0


# ------------------------------------------------------------------------------
# 3. MBPP / Python Code Evaluator
# ------------------------------------------------------------------------------
def evaluate_mbpp_code(response_text, source_row_tests):
    """
    Safely executes model Python functions against MBPP test list assertions.
    """
    if not response_text or not source_row_tests:
        return 0

    code_str = str(response_text).strip()

    # Clean markdown triple backticks if present
    if "```python" in code_str:
        code_str = code_str.split("```python")[1].split("```")[0]
    elif "```" in code_str:
        code_str = code_str.split("```")[1].split("```")[0]

    # Ensure tests are in list format
    if isinstance(source_row_tests, str):
        try:
            tests = ast.literal_eval(source_row_tests)
        except Exception:
            tests = [source_row_tests]
    else:
        tests = source_row_tests

    # Standardize function definition if code generation was incomplete
    execution_code = code_str + "\n\n" + "\n".join(tests)

    # Isolated execution environment
    exec_globals = {}
    try:
        # Run code against test assertions in isolated dict namespace
        exec(execution_code, exec_globals)
        return 1  # All test cases passed
    except Exception:
        return 0  # Failed runtime, assertion, or syntax check
    

def clean_mcq_strict(output_text, options_list):
    """
    Parse LLM MCQ outputs into one of the allowed option tokens.
    Prioritizes explicit answer statements, then falls back to first/last standalone tokens.
    """
    if not output_text or not options_list:
        return "none"

    txt = str(output_text).strip()
    lookup = {str(opt).lower(): opt for opt in options_list}
    escaped = [re.escape(str(opt)) for opt in options_list]
    options_pattern = "|".join(escaped)

    token = rf'(?<![A-Za-z0-9])({options_pattern})(?![A-Za-z0-9])'

    answer_pattern = (
        rf'(?i)'
        rf'(?:'
        rf'final\s+answer|'
        rf'correct\s+answer|'
        rf'answer|'
        rf'choice'
        rf')'
        rf'\s*(?:is|should\s+be|=|:|-)?\s*["\']?'
        rf'({options_pattern})'
        rf'["\']?'
        rf'(?![A-Za-z0-9])'
    )

    # 1. Match explicit statement
    matches = list(re.finditer(answer_pattern, txt))
    if matches:
        return lookup[matches[-1].group(1).lower()]

    # 2. Match standalone tokens (Prefer FIRST match for direct generations like " C\n")
    matches = list(re.finditer(token, txt, flags=re.IGNORECASE))
    if matches:
        return lookup[matches[0].group(1).lower()]

    return "none"


def evaluate_mcq_with_clean_strict(output_text, gold_answer, options_list=["A", "B", "C", "D", "E"]):
    """
    Evaluates MCQ output using clean_mcq_strict while normalizing gold answers
    (converting integer indices like 2 -> 'C' or '2' -> 'C').
    """
    if not output_text:
        return 0

    gold_str = str(gold_answer).strip()

    # Normalize integer target indices (e.g., HellaSwag gold '2' -> 'C')
    if gold_str.isdigit():
        idx = int(gold_str)
        if 0 <= idx < len(options_list):
            gold_str = options_list[idx]

    gold_str = gold_str.upper()

    # Extract prediction using clean_mcq_strict
    pred = clean_mcq_strict(output_text, options_list)

    if pred == "none":
        return 0

    return 1 if pred.upper() == gold_str else 0

""" def clean_mcq_strict(output_text, options_list):
    
    Parse LLM MCQ outputs into one of the allowed option tokens.

    Preference:
      1. Explicit answer statements ("Answer:", "Final answer:", etc.)
      2. Last standalone option token in the text.
      3. "none" if nothing valid is found.
    
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

    return "none" """


def process_dataframe_routerbench(df, dataset_config, metric_dict, self_conf=False, p_true=False, thinking=False):
    """
    Processes model logits and evaluates RouterBench mixed workload generations
    producing clean boolean (0/1) correctness labels.
    """
    # Drop empty responses
    df = df[df['responses'].str.len() > 0].copy()

    # --- 1. Extract Likelihoods & Confidence Signals ---
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
        for i, token in enumerate(gen_tokens):
            try:
                idx = candidate_tokens[i].index(token)
            except Exception:
                idx = -1
            token_likes.append(like_values[i][idx] if idx != -1 else -10.0)
            token_probs.append(all_prob[i][idx] if idx != -1 else 0.0)

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
        df['self_conf'] = [m['self_conf'] for m in df['meta']]

    if p_true:
        df['p_true'] = [m['p_true'] for m in df['meta']]

    # --- 2. Dynamic Task-Based 0/1 Scoring ---
    scores = []
    scored_responses = []

    for idx, row in df.iterrows():
        # Get response string (unwrapping list if needed)
        resp = row['responses']
        resp_text = resp[0] if isinstance(resp, list) and len(resp) > 0 else str(resp)
        
        gold_ans = row['ans']
        task_fam = row.get('task_family', '')

        # Resolve task family if nested in meta or source_row
        if not task_fam and isinstance(row.get('meta'), dict):
            task_fam = row['meta'].get('task_family', '')

        score = 0
        
        # A. Multiple Choice Families (MMLU, ARC, HellaSwag, WinoGrande)
        if task_fam in ['mmlu', 'arc-challenge', 'hellaswag', 'winogrande']:
            score = evaluate_mcq_robust(resp_text, gold_ans)
        elif task_fam == 'grade-school-math':
            score = evaluate_gsm8k_robust(resp_text, gold_ans)
        else:
            score = evaluate_mcq_robust(resp_text, gold_ans)

        scores.append(score)
        scored_responses.append(resp_text)

    # Attach clean 0/1 target vector y for router training & evaluation
    df['scored_responses'] = scored_responses
    df['is_correct'] = scores
    df['acc'] = scores  # 0/1 numeric score matching existing pipeline metrics

    gc.collect()
    return df


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