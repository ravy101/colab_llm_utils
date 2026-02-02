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
  Question: ({item['question']}?)
Short Answer: ("""
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

def build_gold_context(example):
    """
    Constructs context using only gold-supporting paragraphs.
    """
    # Map titles to sentences
    relevant_titles = list(set(example['supporting_facts']['title']))
    titles = example['context']['title']
    sentences = example['context']['sentences']
    paragraphs = []
    for relevent_t in relevant_titles:
        topic_s = []
        for t, s in zip(titles, sentences):
            if relevent_t == t:
                topic_s = topic_s + s
        paragraphs.append( f"{relevent_t}:\n" + " ".join(topic_s))
    return "\n\n".join(paragraphs)

def doc_to_text_hotpot(item):
    context = build_gold_context(item)
    question = item["question"]

    prompt =  f"Answer the question using the information provided in the context. Do not include any detail, notes or explanation, only the required answer.\nContext: {context}\nQuestion: {question}\nAnswer:"
  
    return prompt

def doc_to_ans_hotpot(item):
  return item['answer']


def doc_to_text_summarization(doc):
    prompt = f"""Write a concise, factual summary of the text below.
The summary should capture the main event or most important point, not every message individually.
Do not add new information.
Keep the summary brief and self-contained (1 to 3 numbered sentences).

Text:
{doc.get("document") or doc.get("dialogue")}

Summary:
1. """
    return prompt

def doc_to_text_cnn(doc):
    text = doc.get("article")
    prompt = f"Article: {text}.\nSummarize the article in three sentences.\nSummary:"
    return prompt

def doc_to_summary_cnn(doc):
   return doc.get("highlights").strip()

def doc_to_text_xsum(doc):
    text = doc.get("article")
    prompt = f"Text: {text}.\nSummarize the text one sentence.\nSummary:"
    return prompt

def doc_to_summary_xsum(doc):
   return doc.get("summary").strip()


def doc_to_summary(doc):
    return doc["summary"]

wmt14 = {"clean_name": "wmt14fr-en",
        "dataset_name": "fr-en",
        "dataset_location": "wmt/wmt14",
        "options": None,
        "subset": "train",
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


hotpotqa = {"clean_name": "HotpotQA",
        "dataset_name": "distractor",
        "dataset_location": "hotpotqa/hotpot_qa",
        "options": None,
        "subset": "train",
        "task_type": "qa",
        "dict_ans": True,
        "doc_to_text": doc_to_text_hotpot,
        "doc_to_ans": doc_to_ans_hotpot}


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

wmt14de = {"clean_name": "wmt14de-en",
        "dataset_name": "de-en",
        "dataset_location": "wmt/wmt14",
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

xsum = {
    "clean_name": "XSum",
    "dataset_name": "default",
    "dataset_location": "EdinburghNLP/xsum",
    "options": None,
    "subset": "train",
    "task_type": "summarization",
    "dict_ans": False,
    "doc_to_text": doc_to_text_xsum,
    "doc_to_ans": doc_to_summary_xsum,
}

samsum = {
    "clean_name": "SAMSum",
    "dataset_name": "default",
    "dataset_location": "knkarthick/samsum",
    "options": None,
    "subset": "train",
    "task_type": "summarization",
    "dict_ans": False,
    "doc_to_text": doc_to_text_summarization,
    "doc_to_ans": doc_to_summary,
}

cnn_dailymail = {
    "clean_name": "CNN_Daily Mail",
    "dataset_name": "3.0.0",
    "dataset_location": "abisee/cnn_dailymail",
    "options": None,
    "subset": "train",
    "task_type": "summarization",
    "dict_ans": False,
    "doc_to_text": doc_to_text_cnn,
    "doc_to_ans": doc_to_summary_cnn,
}
