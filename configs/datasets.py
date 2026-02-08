# Datsets config
languages = {"en": "English", "fr": "French", "de": "German",  "cs": "Czech", "ru": "Russian"}

cnn_samples = [{"article": """(CNN)French striker Bafetimbi Gomis, who has a history of fainting, said he is now "feeling well" after collapsing during Swansea's 3-2 loss at Tottenham in the Premier League on Wednesday. The worrying incident occurred in the first half at White Hart Lane -- after Tottenham scored in the seventh minute -- but the 29-year-old left the pitch conscious following about five minutes of treatment. The Guardian added that he was wearing an oxygen mask. Play was temporarily stopped before resuming. As the match progressed, Swansea tweeted that Gomis was "fine," with manager Garry Monk using the same word to describe Gomis' condition. Gomis spent the night in hospital as a precaution, Swansea said on its website. "I wanted to reassure you concerning my health," Gomis told the website. "It actually looks much scarier than it is physically dangerous, and I am feeling well now. "I have been under a great deal of stress and fatigue due to my father's health, which requires me to go back and forth from France. "I was disappointed that I couldn't help my team tonight, but now everything is back in order. I also want to thank everyone for their support and get well messages." Gomis had similar fainting spells in France, which prompted the president of his former club, Jean-Michel Aulas of Lyon, to tell French television in 2009: "We can't not be worried, it scares you each time." Swansea ran tests on Gomis, said Monk, prior to signing him on a free transfer last July. "He just has a little bit of low blood pressure which causes you a little bit of problems," Monk said in a televised interview on Sky. "It's been part of his life. We were well aware of that when we signed him. He's done all the hospital checks and all the medical checks you can possibly do and it's just part of his life. "It's no problems whatsoever. It's not as serious as it looks." Gomis has scored two league goals for Swansea this season, mostly in a backup role. He became the Welsh side's top striker when Wilfried Bony signed with Manchester City in January. Almost exactly three years ago at White Hart Lane, then Bolton midfielder Fabrice Muamba collapsed after suffering a cardiac arrest. He was near death, according to Bolton, but survived after being treated at the London Chest Hospital. He subsequently retired. Other footballers, including Cameroon international Marc-Vivien Foe in 2003 and Spanish international Antonio Puerta in 2007, didn't survive after collapsing on the pitch.""",
                "highlights": """Bafetimbi Gomis collapses within 10 minutes of kickoff at Tottenham. But he reportedly left the pitch conscious and wearing an oxygen mask. Gomis later said that he was "feeling well" The incident came three years after Fabrice Muamba collapsed at White Hart Lane."""},
              {"article": """(CNN)The search for a comic book artist missing in the Cayman Islands since Thursday is now being called a recovery mission. Norman Lee, an artist for DC and Marvel comics, went missing while snorkeling with his wife off the eastern coast of Grand Cayman, CNN affiliate WCVB reported. Strong currents hindered the search, which lasted until Friday evening, Cayman 27 reported. "It is unlikely that we will make any recovery at this stage," Chief Inspector Brad Ebanks told Cayman 27. Lee, 47, of Weymouth, Massachusetts, was known and for his work on "Wolverine Annual," "Supergirl," "Starman" and other comic book titles. Tributes flooded his Facebook page and Twitter from friends, fans and colleagues who knew him from art school and comic conventions. "I cannot express how shaken I am that I will never get the chance to see that smile again, and it saddens me that this world has lost a wonderful man in Norman Lee. To his wife Jan, and his family and all his friends and fans that loved him, my sincerest condolences," friend and fellow graphic artist Chris Kinniery said on Facebook. "I'm so sorry to hear about Norman Lee's disappearance. My condolences go out to his family. ... He was an amazing talent in the industry and it was always a pleasure to work with him," freelance artist .""",
                "highlights": """Comic book artist Norman Lee went missing in the Cayman Islands on Thursday . Authorities called off search on Friday evening ."""},
              {"article": """(CNN)The flight crew of the Delta Air Lines plane that skidded into a fence at LaGuardia Airport last week cited brake issues during the landing, according to an update on Monday from the NTSB. The crew said they did not sense any deceleration from the wheel brake upon landing, despite the auto brakes being set to "max," according to an ongoing investigation by the National Transportation Safety Board. The runway appeared all white in the moments before landing, according to the report. They based their decision to land after receiving a brake action report of "good" from air traffic control, the NTSB said. "The automatic spoilers did not deploy," the crew told the NTSB, "but that the first officer quickly deployed them manually." The captain said he was unable to stop the aircraft from drifting left, according to the report. The Boeing MD-88 sustained significant damage to the left wing, flight spoilers, the nose of the plane and the left wing fuel tank, according to the NTSB. Delta Flight 1086 departed from Atlanta shortly after 9 a.m. Thursday. LaGuardia was dealing with snow and freezing fog as the flight approached its destination about two hours later. The aircraft briefly circled New York because of issues with snow and ice before touching down shortly after 11 a.m. The plane slid off the runway with its nose busting through a fence before skidding to a halt mere feet from frigid waters. Twenty three passengers received minor injuries, and others were transported to the hospital for evaluation. An NTSB meteorologist is examining the weather conditions at the time of the accident, said the report. The cause of the accident has not been determined.""",
               "highlights": """Delta Air Lines Flight 1086 skidded into a fence last week at a LaGuardia Airport beset by winter weather . The NTSB says the crew reported they did not sense any deceleration from the wheel brake upon landing. There were some minor injuries."""}]


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

    prompt = (
        """You are a question answering system.
Answer the question using the information provided in the context.
Answer must contain at least one token.
The answer should be short (one or two words, or a short phrase).\n"""
        f"Context:\n{context}\n"
        f"Question:\n{question}\n"
        "Answer:\n"
    )
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
    text_sample = cnn_samples[0]["article"]
    summary_sample = cnn_samples[0]["highlights"]
    prompt = f""" You are a news summarization assistant.

    Article:
{text_sample}

Summary: {summary_sample}

Article:
{text}

Write a very short summary of the article in less than 100 words.
Paraphrase the content.
Focus on the main events and outcomes.

Summary:"""
    return prompt

def doc_to_summary_cnn(doc):
   return doc.get("highlights").strip()

def doc_to_text_xsum(doc):
    text = doc.get("article")
    prompt = f"Text: {text}.\nSummarize the article concisely in three sentences. Do not copy sentences verbatim. Focus on the main events and outcomes.\nSummary:"
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
