# Datsets config
languages = {"en": "English", "fr": "French", "de": "German",  "cs": "Czech", "ru": "Russian"}

def doc_to_text_wmt_fr(item, from_lang = 'fr', to_lang = 'en'):
  return f"{languages[from_lang]} source: {item['translation'][from_lang]}\n{languages[to_lang]} translation:"

def doc_to_text_wmt_fr_inst(item, from_lang = 'fr', to_lang = 'en'):
  return f"Translate  the following from {languages[from_lang]} to {languages[to_lang]}: {item['translation'][from_lang]}\nTranslation:"

def doc_to_text_wmt_ru(item, from_lang = 'en', to_lang = 'ru'):
  return f"{languages[from_lang]} source: {item['translation'][from_lang]}\n{languages[to_lang]} translation:"

def doc_to_text_wmt_de(item, from_lang = 'de', to_lang = 'en'):
  return f"{languages[from_lang]} source: {item['translation'][from_lang]}\n{languages[to_lang]} translation:"

def doc_to_text_wmt_de_inst(item, from_lang = 'de', to_lang = 'en'):
  return f"Translate  the following from {languages[from_lang]} to {languages[to_lang]}: {item['translation'][from_lang]}\nTranslation:"

def doc_to_text_qa(item):
  return f"Provide a short answer without explanation.\n Question: {item['question']}\nShort Answer:"

def doc_to_text_nq(item):
  text = f"""
  Please provide the specific answer to the following question. Do not include reasoning, explanation or conversational filler. Output only the required information as concisely as possible.
  Question: {item['question']}?
Answer:"""
  return text


def doc_to_text_qa_conf(item):
  return f"Provide a short answer and a percentage reflecting how confident you are it is correct without any explanation.\n Question: {item['question']}\nShort Answer:"

def doc_to_answer_qa(item):
  return item['answer']

def doc_to_text_sciq(item):
  return f"Provide the answer without explanation in as few words as possible.\n Question: {item['question']}\nShort Answer:"

def p_true_text():
  return (
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "Is the answer above true? Answer with 'True' or 'False'.<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n")  


def doc_to_answer_sciq(item):
  return item['correct_answer']

def doc_to_answer_wmt_fr(item, from_lang = 'fr', to_lang = 'en'):
  return item['translation'][to_lang]

def doc_to_answer_wmt_ru(item, from_lang = 'en', to_lang = 'ru'):
  return item['translation'][to_lang]

def doc_to_answer_wmt_de(item, from_lang = 'de', to_lang = 'en'):
  return item['translation'][to_lang]

def doc_to_text_truthful(item):
  return f"Provide a short, truthful, factual answer to this question. {item['question']}\nAnswer:"

def doc_to_answer_truthful(item):
  return item['correct_answers']

wmt14 = {"clean_name": "wmt14fr-en",
        "dataset_name": "fr-en",
        "dataset_location": "wmt/wmt14",
        "options": None,
        "subset": "test",
        "task_type": "translation",
        "dict_ans": False,
        "doc_to_text": doc_to_text_wmt_fr,
        "doc_to_ans": doc_to_answer_wmt_fr}

triviaqa = {"clean_name": "TriviaQA",
        "dataset_name": "rc",
        "dataset_location": "mandarjoshi/trivia_qa",
        "options": None,
        "subset": "train",
        "task_type": "qa",
        "dict_ans": True,
        "doc_to_text": doc_to_text_qa,
        "doc_to_ans": doc_to_answer_qa}

nqopen = {"clean_name": "NQOpen",
        "dataset_name": "nq_open",
        "dataset_location": "google-research-datasets/nq_open",
        "options": None,
        "subset": "train",
        "task_type": "qa",
        "dict_ans": False,
        "doc_to_text": doc_to_text_nq,
        "doc_to_ans": doc_to_answer_qa}

truthfulqa = {"clean_name": "TruthfulQA",
        "dataset_name": "generation",
        "dataset_location": "truthfulqa/truthful_qa",
        "options": None,
        "subset": "validation",
        "task_type": "qa",
        "dict_ans": True,
        "doc_to_text": doc_to_text_truthful,
        "doc_to_ans": doc_to_answer_truthful}

wmt14ru = {"clean_name": "wmt14ru-en",
        "dataset_name": "ru-en",
        "dataset_location": "wmt/wmt14",
        "options": None,
        "subset": "test",
        "task_type": "translation",
        "dict_ans": False,
        "doc_to_text": doc_to_text_wmt_ru,
        "doc_to_ans": doc_to_answer_wmt_ru}

wmt19de = {"clean_name": "wmt19de-en",
        "dataset_name": "de-en",
        "dataset_location": "wmt/wmt19",
        "options": None,
        "subset": "train",
        "task_type": "translation",
        "dict_ans": False,
        "doc_to_text": doc_to_text_wmt_de,
        "doc_to_ans": doc_to_answer_wmt_de}

sciq = {"clean_name": "SciQ",
        "dataset_name": "default",
        "dataset_location": "allenai/sciq",
        "options": None,
        "subset": "train",
        "task_type": "qa",
        "dict_ans": False,
        "doc_to_text": doc_to_text_sciq,
        "doc_to_ans": doc_to_answer_sciq}

