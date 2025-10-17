import torch.functional as F
import string
import evaluate

MAX_ALIASES = 8
_bem_tokenizer = None 
_bem_model = None
_meteor = None
_bleurt = None
_rouge = None

# -------------------
# Lazy-loaded metric getters
# -------------------
def get_meteor():
    """Return the METEOR scorer (lazy loaded)."""
    global _meteor
    if _meteor is None:
        print("Loading METEOR metric...")
        _meteor = evaluate.load("meteor")
    return _meteor

def get_bleurt():
    """Return the BLEURT scorer (lazy loaded)."""
    global _bleurt
    if _bleurt is None:
        print("Loading BLEURT metric...")
        _bleurt = evaluate.load("bleurt", module_type="metric")
    return _bleurt

def get_rouge():
    """Return the ROUGE scorer (lazy loaded)."""
    global _rouge
    if _rouge is None:
        print("Loading ROUGE metric...")
        _rouge = evaluate.load("rouge")
    return _rouge



def _initialize_model():
    """Initialize the model and tokenizer if not already loaded."""
    global _bem_model, _bem_tokenizer
    if _bem_model is None or _bem_tokenizer is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        _bem_tokenizer = AutoTokenizer.from_pretrained("kortukov/answer-equivalence-bem")
        _bem_model = AutoModelForSequenceClassification.from_pretrained("kortukov/answer-equivalence-bem")

    return _bem_model, _bem_tokenizer


def get_bem_model():
    model, _ = _initialize_model()
    return model


def get_bem_tokenizer():
    _, tokenizer = _initialize_model()
    return tokenizer

def normalize_answer(s):
    """Lower text and remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return " ".join([w for w in text.split() if w not in ("a", "an", "the")])
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return text.translate(str.maketrans("", "", string.punctuation))
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * (precision * recall) / (precision + recall)

def bem_score(prediction, ground_truth, question, binary_out = False):
  bem_model, bem_tokenizer = _initialize_model()
  text = f"[CLS] {prediction} [SEP]"
  text_pair = f"{ground_truth} [SEP] {question} [SEP]"
  inputs = bem_tokenizer(text=text, text_pair=text_pair, add_special_tokens=False, padding='max_length', truncation=True, return_tensors='pt')
  out = bem_model(**inputs)
  if binary_out:
    prediction = float(F.softmax(out.logits, dim=-1).argmax().item())
  else:
    prediction = F.softmax(out.logits, dim=-1).tolist()[0][1]
  return prediction



# BEST AGGREGATORS

def best_rouge_l(prediction, reference_aliases):
    # Compute ROUGE-L for all aliases and take the best F1
    scores = []
    #if type(prediction) == list:
    #  prediction = prediction[0]

    for alias in reference_aliases[:MAX_ALIASES]:
        result = get_rouge().compute(
            predictions=prediction,
            references=[alias],
            rouge_types=["rougeL"]
        )
        scores.append(result["rougeL"])
    return max(scores)

def best_bleurt(prediction, reference_aliases):
    # Compute ROUGE-L for all aliases and take the best F1
    scores = []
    #if type(prediction) == list:
    #  prediction = prediction[0]
    if type(reference_aliases) == str:
      reference_aliases = [reference_aliases]
    for alias in reference_aliases[:MAX_ALIASES]:
      print(f"computing bleurt {prediction} == {alias}")
      result = get_bleurt().compute(predictions=prediction, references=[alias])
      scores.append(result["scores"])
    return max(scores)

def best_em(prediction, reference_aliases):
    # Compute EM for all aliases and take the best F1
    scores = []
    #if type(prediction) == list:
    #  prediction = prediction[0]
    if type(reference_aliases) == str:
      reference_aliases = [reference_aliases]
    if type(prediction) == list:
      prediction = prediction[0]
    for alias in reference_aliases[:MAX_ALIASES]:
      print(f"computing em {prediction} == {alias}")
      result = exact_match(prediction, alias)
      scores.append(float(result))
    return max(scores)

def best_f1(prediction, reference_aliases):
    # Compute EM for all aliases and take the best F1
    scores = []
    if type(reference_aliases) == str:
      reference_aliases = [reference_aliases]
    if type(prediction) == list:
      prediction = prediction[0]
    for alias in reference_aliases[:MAX_ALIASES]:
      print(f"computing f1 {prediction} == {alias}")
      result = f1_score(prediction, alias)
      scores.append(result)
    return max(scores)

def best_bem(prediction, reference_aliases, question):
    # Compute BEM for all aliases and take the best F1
    scores = []
    if type(reference_aliases) == str:
      reference_aliases = [reference_aliases]
    if type(prediction) == list:
      prediction = prediction[0]
    for alias in reference_aliases[:MAX_ALIASES]:
      print(f"computing bem {prediction} == {alias}")
      result = bem_score(prediction, alias, question)
      scores.append(result)
    return max(scores)

