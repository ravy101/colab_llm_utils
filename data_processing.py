import gc
import numpy as np
import pandas as pd
import re
from . import likelihood
from . import scorers
from . import text
import ast
import string
from difflib import SequenceMatcher

def detect_options_from_prompt(prompt_text):
    """
    Detect whether the prompt appears to contain A-D or A-E options.
    Defaults to A-D.
    """
    txt = str(prompt_text)

    # Look for an actual E option marker, e.g. "E)", "E.", "E:"
    if re.search(r'(?<![A-Za-z0-9])E[\)\.\:](?![A-Za-z0-9])', txt, re.IGNORECASE):
        return ["A", "B", "C", "D", "E"]

    return ["A", "B", "C", "D"]


def clean_mcq_strict(output_text, options_list=None, prompt_text=None):
    """
    Extract a single MCQ option letter from an LLM response.
    """
    if output_text is None:
        return "none"

    txt = str(output_text).strip()

    if not txt:
        return "none"

    if options_list is None:
        options_list = (
            detect_options_from_prompt(prompt_text)
            if prompt_text
            else ["A", "B", "C", "D", "E"]
        )

    options = [str(x).strip().upper() for x in options_list]
    lookup = {x.lower(): x for x in options}

    options_pattern = "|".join(re.escape(x) for x in options)

    # ---------------------------------------------------------
    # 1. Explicit answer statements
    # ---------------------------------------------------------
    answer_pattern = rf"""
        (?:
            final\s+answer |
            correct\s+answer |
            answer |
            choice |
            option
        )
        \s*
        (?:is|should\s+be|=|:|-)? 
        \s*
        [\(\[]?
        ({options_pattern})
        [\)\]]?
        (?![A-Za-z0-9])
    """

    matches = list(
        re.finditer(
            answer_pattern,
            txt,
            flags=re.IGNORECASE | re.VERBOSE
        )
    )

    if matches:
        return lookup[matches[-1].group(1).lower()]

    # ---------------------------------------------------------
    # 2. If the entire response starts with an option
    # ---------------------------------------------------------
    leading_pattern = rf"""
        ^
        \s*
        [\(\[]?
        ({options_pattern})
        [\)\]]?
        (?=
            \s |
            [\.\:\-] |
            $
        )
    """

    match = re.search(
        leading_pattern,
        txt,
        flags=re.IGNORECASE | re.VERBOSE
    )

    if match:
        return lookup[match.group(1).lower()]

    # ---------------------------------------------------------
    # 3. Standalone option token
    # ---------------------------------------------------------
    token_pattern = rf"""
        (?<![A-Za-z0-9])
        ({options_pattern})
        (?![A-Za-z0-9])
    """

    matches = list(
        re.finditer(
            token_pattern,
            txt,
            flags=re.IGNORECASE | re.VERBOSE
        )
    )

    if matches:
        return lookup[matches[0].group(1).lower()]

    return "none"


def normalize_mcq_gold(gold_answer, options_list):
    """
    Convert common gold-answer representations into canonical
    option letter A/B/C/D/E.
    """

    if gold_answer is None:
        return "none"

    # Handle list/tuple answers such as ['C']
    if isinstance(gold_answer, (list, tuple)):
        if len(gold_answer) == 0:
            return "none"
        gold_answer = gold_answer[0]

    gold = str(gold_answer).strip()

    if not gold:
        return "none"

    options = [str(x).strip().upper() for x in options_list]

    # Already a letter
    if gold.upper() in options:
        return gold.upper()

    # Numeric answer
    try:
        n = int(gold)

        # 0-based indexing
        if 0 <= n < len(options):
            return options[n]

        # 1-based indexing
        if 1 <= n <= len(options):
            return options[n - 1]

    except (ValueError, TypeError):
        pass

    # Handle things like "(C)", "[C]", "C.", "C:"
    match = re.match(
        rf'^\s*[\(\[]?({"|".join(re.escape(x) for x in options)})'
        rf'[\)\]\.\:]?\s*$',
        gold,
        re.IGNORECASE
    )

    if match:
        return match.group(1).upper()

    return "none"

def extract_source_answer_text(row):
    """
    Extract the authoritative correct answer text from the original MMLU
    source row.

    source_row["answer"] is the zero-indexed position of the correct choice.
    """
    source_row = row["source_row"]

    choices = source_row["choices"]
    answer_idx = source_row["answer"]

    if not isinstance(choices, (list, tuple)):
        raise ValueError("source_row['choices'] must be a list or tuple.")

    if not isinstance(answer_idx, int) or not 0 <= answer_idx < len(choices):
        raise ValueError(
            f"Invalid source answer index: {answer_idx!r} "
            f"for {len(choices)} choices."
        )

    return str(choices[answer_idx]).strip()

def extract_prompt_choices(prompt):
    """
    Extract shuffled multiple-choice options from an MMLU prompt.

    Returns:
        {
            "A": "choice text",
            "B": "choice text",
            ...
        }
    """
    # The prompts use the format:
    # A) ...
    # B) ...
    # C) ...
    # D) ...
    #
    # Capture everything between one option marker and the next marker
    # (or the instruction text / Answer: at the end).
    pattern = re.compile(
        r'([A-D])\)\s*(.*?)'
        r'(?=\s+[A-D]\)\s*|\s+Print only a single choice|\s+Answer:|$)',
        flags=re.IGNORECASE | re.DOTALL,
    )

    matches = pattern.findall(str(prompt))

    choices = {
        letter.upper(): text.strip()
        for letter, text in matches
    }

    return choices

def find_mmlu_correct_answer(row):
    """
    Determine which answer letter in the shuffled prompt corresponds to
    the authoritative answer from the original MMLU source row.

    Returns:
        The prompt answer letter, e.g. "A", "B", "C", or "D".
    """
    source_answer = extract_source_answer_text(row)
    prompt_choices = extract_prompt_choices(row["prompts"])

    if not prompt_choices:
        raise ValueError("Could not extract any choices from prompt.")

    def normalize(text):
        text = str(text).strip().lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text

    source_norm = normalize(source_answer)

    scores = {
        letter: SequenceMatcher(
            None,
            source_norm,
            normalize(choice_text)
        ).ratio()
        for letter, choice_text in prompt_choices.items()
    }

    best_letter = max(scores, key=scores.get)

    return best_letter

def evaluate_mcq_robust(
    response_text,
    gold_answer,
    prompt_text=None,
    options_list=None
):
    """
    Extract prediction and compare against normalized gold answer.
    """
    if response_text is None or gold_answer is None:
        return 0

    if options_list is None:
        options_list = (
            detect_options_from_prompt(prompt_text)
            if prompt_text
            else ["A", "B", "C", "D", "E"]
        )

    pred = clean_mcq_strict(
        response_text,
        options_list=options_list,
        prompt_text=prompt_text
    )

    gold = normalize_mcq_gold(
        gold_answer,
        options_list
    )

    return int(pred != "none" and pred.upper() == gold.upper())


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

            if task_fam in ['arc-challenge', 'hellaswag', 'winogrande']:
                pred = clean_mcq_strict(resp_str, options_list=dataset_config.get("options"), prompt_text=prompt)
                score = evaluate_mcq_robust(resp_str, ans, prompt_text=prompt)
                response = [pred]

            elif task_fam == 'mmlu':
                pred = clean_mcq_strict(resp_str, options_list=dataset_config.get("options"), prompt_text=prompt)
                ans = find_mmlu_correct_answer(row)
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
    elif task_type == 'multiple_choice':
        for idx, row in df.iterrows():
            out = row['responses']
            resp_str = out[0] if isinstance(out, list) and len(out) > 0 else str(out)
            ans = row['ans']
            prompt = row['prompts']

            pred = clean_mcq_strict(
                resp_str,
                options_list=dataset_config.get("options"),
                prompt_text=prompt
            )

            score = evaluate_mcq_robust(
                resp_str,
                ans,
                options_list=dataset_config.get("options"),
                prompt_text=prompt
            )

            response = [pred]
            graded_responses.append(response)
            results_em.append(score)
        df['scored_responses'] = graded_responses
        df['em'] = results_em
        df['is_correct'] = results_em


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