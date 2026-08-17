import gc
import numpy as np
import pandas as pd
import re
from . import likelihood
from . import scorers
from . import text
import ast
import string

# ==============================================================================
# 1. HELPER PARSERS & TASK EVALUATORS
# ==============================================================================

def detect_options_from_prompt(prompt_text):
    """
    Detects allowed option letters in the prompt (handles 4-choice A-D and 5-choice A-E).
    """
    txt = str(prompt_text)
    if re.search(r'(?<![A-Za-z0-9])[E|e][\)\.\:\s]', txt) or "E)" in txt or "E." in txt:
        return ["A", "B", "C", "D", "E"]
    return ["A", "B", "C", "D"]


def clean_mcq_strict(output_text, options_list=None, prompt_text=None):
    """
    Parse LLM MCQ outputs into one of the allowed option tokens.
    Prefers explicit answer statements, then falls back to first standalone option token.
    """
    if not output_text:
        return "none"

    if options_list is None:
        options_list = detect_options_from_prompt(prompt_text) if prompt_text else ["A", "B", "C", "D", "E"]

    txt = str(output_text).strip()
    lookup = {str(opt).lower(): opt for opt in options_list}
    escaped = [re.escape(str(opt)) for opt in options_list]
    options_pattern = "|".join(escaped)

    token_pattern = rf'(?<![A-Za-z0-9])({options_pattern})(?![A-Za-z0-9])'

    answer_pattern = (
        rf'(?i)'
        rf'(?:'
        rf'final\s+answer|'
        rf'correct\s+answer|'
        rf'answer|'
        rf'choice|'
        rf'option'
        rf')'
        rf'\s*(?:is|should\s+be|=|:|-)?\s*["\']?'
        rf'({options_pattern})'
        rf'["\']?'
        rf'(?![A-Za-z0-9])'
    )

    # 1. Match explicit answer statement ("Answer: C", "The choice is C")
    matches = list(re.finditer(answer_pattern, txt))
    if matches:
        return lookup[matches[-1].group(1).lower()]

    # 2. First standalone option token in output stream
    matches = list(re.finditer(token_pattern, txt, flags=re.IGNORECASE))
    if matches:
        return lookup[matches[0].group(1).lower()]

    return "none"


def evaluate_mcq_robust(response_text, gold_answer, prompt_text=None, options_list=None):
    """
    Evaluates MCQ output against pre-aligned gold letter target.
    """
    if not response_text or not gold_answer:
        return 0

    if options_list is None:
        options_list = detect_options_from_prompt(prompt_text) if prompt_text else ["A", "B", "C", "D", "E"]

    gold_str = str(gold_answer).strip().upper()

    # 1. Fast path: Match leading choice letter token at start of generation ("A\n", "A.", "A)")
    txt = str(response_text).strip()
    first_token = re.match(rf'^\s*([A-E])(?![A-Za-z0-9])', txt, re.IGNORECASE)
    if first_token:
        return 1 if first_token.group(1).upper() == gold_str else 0

    # 2. Pattern path: Extract target choice via strict regex
    pred = clean_mcq_strict(response_text, options_list=options_list, prompt_text=prompt_text)
    if pred == "none":
        return 0

    return 1 if pred.upper() == gold_str else 0


def evaluate_gsm8k_robust(response_text, gold_answer):
    """
    Extracts numerical target from CoT / reasoning outputs and matches against gold numeric target.
    """
    if not response_text or gold_answer is None:
        return 0

    txt = str(response_text).strip()
    gold_str = str(gold_answer).strip()

    gold_nums = re.findall(r'[-+]?\d+(?:\.\d+)?', gold_str.replace(',', ''))
    if not gold_nums:
        return 0
    gold_val = gold_nums[-1]

    # A. Check \boxed{16}
    boxed_match = re.search(r'\\boxed\{([-+]?\d+(?:\.\d+)?)\}', txt)
    if boxed_match and boxed_match.group(1).replace(',', '') == gold_val:
        return 1

    # B. Check #### 16
    if '####' in txt:
        tail = txt.split('####')[-1]
        tail_nums = re.findall(r'[-+]?\d+(?:\.\d+)?', tail.replace(',', ''))
        if tail_nums and tail_nums[-1] == gold_val:
            return 1

    # C. Check 'Answer: 16' or 'is 16'
    ans_match = re.search(r'(?:answer|equals?|total|is)\s*[:=]?\s*\$?\s*([-+]?\d+(?:\.\d+)?)', txt, re.IGNORECASE)
    if ans_match and ans_match.group(1).replace(',', '') == gold_val:
        return 1

    # D. Fallback: Last numeric token
    all_nums = re.findall(r'[-+]?\d+(?:\.\d+)?', txt.replace(',', ''))
    if all_nums and all_nums[-1] == gold_val:
        return 1

    return 0


def evaluate_mbpp_code(response_text, source_row_tests):
    """
    Executes generated Python code against test assertions.
    """
    if not response_text or not source_row_tests:
        return 0

    code_str = str(response_text).strip()

    if "```python" in code_str:
        code_str = code_str.split("```python")[1].split("```")[0]
    elif "```" in code_str:
        code_str = code_str.split("```")[1].split("```")[0]

    code_str = code_str.strip()

    if isinstance(source_row_tests, str):
        try:
            tests = ast.literal_eval(source_row_tests)
        except Exception:
            tests = [source_row_tests]
    else:
        tests = source_row_tests

    exec_script = f"{code_str}\n\n" + "\n".join(tests)
    namespace = {}
    try:
        exec(exec_script, namespace)
        return 1
    except Exception:
        return 0
    
def process_dataframe(df, dataset_config, metric_dict=None, self_conf=False, p_true=False, thinking=False):
    """
    Main evaluation and feature extraction entry point.
    Extracts likelihoods, Chow confidence features, thinking tags, and evaluates 0/1 accuracy.
    """
    # 1. Filter out empty responses
    df = df[df['responses'].str.len() > 0].copy()

    # 2. Extract Token Likelihoods & Chow Confidence Features
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
            token_likes.append(like_values[i][idx] if idx != -1 else 0.0)
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

    # 3. Handle Thinking / Chain-of-Thought Text Splitting
    if thinking:
        splits = [text.split_tagged_text(a[0]) for a in df['responses']]
        df['thinking_text'] = [s[1] for s in splits]
        df['responses'] = [[s[0]] for s in splits]
        df = df[df['responses'].str.len() > 0].copy()
    else:
        df['thinking_text'] = None

    # 4. Handle Metadata Flags
    if self_conf and 'meta' in df.columns:
        df['self_conf'] = [m.get('self_conf') for m in df['meta']]

    if p_true and 'meta' in df.columns:
        df['p_true'] = [m.get('p_true') for m in df['meta']]

    # 5. Task Evaluation Blocks
    graded_responses = []
    results = []
    results_bl = []
    results_em = []
    results_f1 = []

    # --------------------------------------------------------------------------
    # 'mixed' MULTI-TASK EVALUATION (RouterBench-10K)
    # --------------------------------------------------------------------------
    task_type = dataset_config.get('task_type')
    if task_type == 'mixed':
        for idx, row in df.iterrows():
            out = row['responses']
            resp_str = out[0] if isinstance(out, list) and len(out) > 0 else str(out)
            ans = row['ans']
            prompt = row['prompts']
            task_fam = str(row['task_family']).lower()
            source_row = row.get('source_row', {})

            if task_fam in ['mmlu', 'arc-challenge', 'hellaswag', 'winogrande']:
                pred = clean_mcq_strict(resp_str, options_list=dataset_config.get("options"), prompt_text=prompt)
                score = evaluate_mcq_robust(resp_str, ans, prompt_text=prompt)
                response = [pred]

            elif task_fam == 'grade-school-math':
                score = evaluate_gsm8k_robust(resp_str, ans)
                response = [resp_str]

            elif task_fam == 'mbpp':
                tests = source_row.get('test_list', ans) if isinstance(source_row, dict) else ans
                score = evaluate_mbpp_code(resp_str, tests)
                response = [resp_str]

            else:
                pred = clean_mcq_strict(resp_str, options_list=dataset_config.get("options"), prompt_text=prompt)
                score = evaluate_mcq_robust(resp_str, ans, prompt_text=prompt)
                response = [pred]

            graded_responses.append(response)
            results_em.append(score)

        df['scored_responses'] = graded_responses
        df['em'] = results_em
        df['is_correct'] = results_em
    elif task_type == 'translation':
        for out, ans, question in zip(df['responses'], df['ans'], df['prompts']):
            graded_responses.append(out)
            results.append(metric_dict['meteor'].compute(predictions=out, references=[ans]))
            results_bl.append(metric_dict['bleurt'].compute(predictions=out, references=[ans]))
        df['scored_responses'] = graded_responses
        df['meteor'] = [r['meteor'] for r in results]
        df['bleurt'] = [r['scores'][0] for r in results_bl]
    elif task_type == 'qa':
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
    gc.collect()
    return df


def summarize_accuracy_by_task(df):
    """
    Prints a clean aggregate summary of accuracy broken down by task family.
    """
    summary = df.groupby('task_family').agg(
        total_samples=('is_correct', 'count'),
        correct_count=('is_correct', 'sum'),
        accuracy=('is_correct', 'mean')
    ).reset_index()

    summary['accuracy_pct'] = (summary['accuracy'] * 100).round(2).astype(str) + '%'
    
    print("\n=== Accuracy & Breakdown by Task Family ===")
    print(summary.to_string(index=False))
    return summary

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