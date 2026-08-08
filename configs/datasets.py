# Datsets config
import string
import re

languages = {"en": "English", "fr": "French", "de": "German",  "cs": "Czech", "ru": "Russian"}

cnn_samples = [{"article": """(CNN)French striker Bafetimbi Gomis, who has a history of fainting, said he is now "feeling well" after collapsing during Swansea's 3-2 loss at Tottenham in the Premier League on Wednesday. The worrying incident occurred in the first half at White Hart Lane -- after Tottenham scored in the seventh minute -- but the 29-year-old left the pitch conscious following about five minutes of treatment. The Guardian added that he was wearing an oxygen mask. Play was temporarily stopped before resuming. As the match progressed, Swansea tweeted that Gomis was "fine," with manager Garry Monk using the same word to describe Gomis' condition. Gomis spent the night in hospital as a precaution, Swansea said on its website. "I wanted to reassure you concerning my health," Gomis told the website. "It actually looks much scarier than it is physically dangerous, and I am feeling well now. "I have been under a great deal of stress and fatigue due to my father's health, which requires me to go back and forth from France. "I was disappointed that I couldn't help my team tonight, but now everything is back in order. I also want to thank everyone for their support and get well messages." Gomis had similar fainting spells in France, which prompted the president of his former club, Jean-Michel Aulas of Lyon, to tell French television in 2009: "We can't not be worried, it scares you each time." Swansea ran tests on Gomis, said Monk, prior to signing him on a free transfer last July. "He just has a little bit of low blood pressure which causes you a little bit of problems," Monk said in a televised interview on Sky. "It's been part of his life. We were well aware of that when we signed him. He's done all the hospital checks and all the medical checks you can possibly do and it's just part of his life. "It's no problems whatsoever. It's not as serious as it looks." Gomis has scored two league goals for Swansea this season, mostly in a backup role. He became the Welsh side's top striker when Wilfried Bony signed with Manchester City in January. Almost exactly three years ago at White Hart Lane, then Bolton midfielder Fabrice Muamba collapsed after suffering a cardiac arrest. He was near death, according to Bolton, but survived after being treated at the London Chest Hospital. He subsequently retired. Other footballers, including Cameroon international Marc-Vivien Foe in 2003 and Spanish international Antonio Puerta in 2007, didn't survive after collapsing on the pitch.""",
                "highlights": """Bafetimbi Gomis collapses within 10 minutes of kickoff at Tottenham. But he reportedly left the pitch conscious and wearing an oxygen mask. Gomis later said that he was "feeling well" The incident came three years after Fabrice Muamba collapsed at White Hart Lane."""},
              {"article": """(CNN)The search for a comic book artist missing in the Cayman Islands since Thursday is now being called a recovery mission. Norman Lee, an artist for DC and Marvel comics, went missing while snorkeling with his wife off the eastern coast of Grand Cayman, CNN affiliate WCVB reported. Strong currents hindered the search, which lasted until Friday evening, Cayman 27 reported. "It is unlikely that we will make any recovery at this stage," Chief Inspector Brad Ebanks told Cayman 27. Lee, 47, of Weymouth, Massachusetts, was known and for his work on "Wolverine Annual," "Supergirl," "Starman" and other comic book titles. Tributes flooded his Facebook page and Twitter from friends, fans and colleagues who knew him from art school and comic conventions. "I cannot express how shaken I am that I will never get the chance to see that smile again, and it saddens me that this world has lost a wonderful man in Norman Lee. To his wife Jan, and his family and all his friends and fans that loved him, my sincerest condolences," friend and fellow graphic artist Chris Kinniery said on Facebook. "I'm so sorry to hear about Norman Lee's disappearance. My condolences go out to his family. ... He was an amazing talent in the industry and it was always a pleasure to work with him," freelance artist .""",
                "highlights": """Comic book artist Norman Lee went missing in the Cayman Islands on Thursday . Authorities called off search on Friday evening ."""},
              {"article": """(CNN)The flight crew of the Delta Air Lines plane that skidded into a fence at LaGuardia Airport last week cited brake issues during the landing, according to an update on Monday from the NTSB. The crew said they did not sense any deceleration from the wheel brake upon landing, despite the auto brakes being set to "max," according to an ongoing investigation by the National Transportation Safety Board. The runway appeared all white in the moments before landing, according to the report. They based their decision to land after receiving a brake action report of "good" from air traffic control, the NTSB said. "The automatic spoilers did not deploy," the crew told the NTSB, "but that the first officer quickly deployed them manually." The captain said he was unable to stop the aircraft from drifting left, according to the report. The Boeing MD-88 sustained significant damage to the left wing, flight spoilers, the nose of the plane and the left wing fuel tank, according to the NTSB. Delta Flight 1086 departed from Atlanta shortly after 9 a.m. Thursday. LaGuardia was dealing with snow and freezing fog as the flight approached its destination about two hours later. The aircraft briefly circled New York because of issues with snow and ice before touching down shortly after 11 a.m. The plane slid off the runway with its nose busting through a fence before skidding to a halt mere feet from frigid waters. Twenty three passengers received minor injuries, and others were transported to the hospital for evaluation. An NTSB meteorologist is examining the weather conditions at the time of the accident, said the report. The cause of the accident has not been determined.""",
               "highlights": """Delta Air Lines Flight 1086 skidded into a fence last week at a LaGuardia Airport beset by winter weather . The NTSB says the crew reported they did not sense any deceleration from the wheel brake upon landing. There were some minor injuries."""}]

xsum_samples = [{"article": """Four police officers were injured in the incident on Friday night.
  A man, aged 19, and a boy, aged 16, have been charged with six counts of aggravated vehicle taking.
  They are due to appear before Belfast Magistrates' Court on Monday.
  The 19-year-old man has also been charged with driving while disqualified and using a motor vehicle without insurance.""",
  "summary": """Two teenagers have been charged in connection with an incident in west Belfast in which a car collided with two police vehicles."""},
  {"article": """The think tank said the city's 1,536 schools needed to save £360m in the first year if the government's National Funding Formula (NFF) plan goes ahead.
  The amount is the equivalent of 12,857 qualified teachers, on an average salary of £28,000.
  The government said London was the highest funded part of the country.
  It added that under the plans, which are under consultation, inner-city schools would be allocated 30% more money per pupil than the national average.
  But London Councils, which represents the city's 32 boroughs and the City, said no school would gain enough funding from the NFF to compensate for increased cost pressures from inflation, higher pension contributions and national insurance.
  Ministers said the new formula was needed to tackle uneven levels of funding across England, with the best funded areas getting more than £6,300 per pupil per year, while the worst-funded averaging £4,200.
  It said the funding cut was on top of National Audit Office figures which showed England schools faced an eight per cent real-terms cut per pupil by 2019-20 because it wider cost pressures.
  In a statement, London Councils said: "At a time when UK schools are seen as underperforming by international standards, and when businesses based in London are facing massive uncertainty about recruiting skilled staff, there is an urgent need to invest in schools in London and across the rest of the country."
  It added: "Without the right qualifications and skills, London's children will be unable to access jobs and contribute to the national economy. Over 60% of jobs in inner London require a degree and around 45% of jobs in the rest of the capital require a degree." """,
  "summary": """About 70% of London schools could face budget cuts under government plans to change how they are funded, according to London Councils."""},
  {"article": """The referendum will take place on 10 March, but Bath Conservative MP Ben Howlett said he was concerned about a "lack of awareness" about the issue.
  Mr Howlett also said he is worried about the public's level of engagement.
  Bath and North East Somerset Council said the referendum had been publicised in press releases and tweets.
  It also said it was the subject of a two-page article in the winter edition of the council magazine which was distributed to all households in the region.
  A further news release and polling cards will also be sent out to all households this week, the authority added.
  Supporters of the referendum say Bath needs a mayor to give local government more visibility.
  Directly elected mayors were created by the Local Government Act 2000 as one option for local government, as long as the idea was backed in a referendum.
  Mr Howlett said he was "personally concerned" that an elected mayor was not appropriate for an area "as diverse" as Bath and North East Somerset, and that it could "lead to an increase in the cost of local politics".
  "The level of misinformation on this issue is worrying - many people seem to still believe this is about a mayor of Bath and not understanding it would cover all of Bath and North East Somerset.
  "I hope in the coming weeks more information will be forthcoming to enable residents to make an informed decision," he added.""",
  "summary": """An MP has criticised "the level of misinformation" about a referendum on an elected mayor for Bath and North East Somerset."""}]

samsum_samples = [{"dialogue": """Sadie: can i borrow your bike again please?
Chloe: when?
Sadie: on thursday, i need to go to the dentist quickly after work
Chloe: sure, let me know when you want to pick it up
Sadie: wednesday evening will be good?
Chloe: sure, come over and please remember to lock it properly!!""",
  "summary": """Sadie will borrow Chloe's bike on Wednesday evening. She has a dentist appointment on Thursday after work."""},
  {"dialogue": """Diane: Will you be my kids' Lorelai?
Kate: Awww :3
Kate: of course, I will!
Kate: Not that they would ever need one with such an amazing mother as you...
Diane: Well, I'm not sure
Diane: I'm terrified
Kate: It's normal, everyone gets scared before the birth
Kate: But once you hold her in your arms...
Kate: You'll forget everything
Diane: Yeah, I guess...
Diane: I just can't wait for it to happen
Diane: The waiting is the worst.
Kate: Hang in there a bit longer, honey <3
Kate: you can do it!
Kate: <file_gif>
Diane: loool, thanks :D""",  
"summary": """Diane is pregnant and can't wait to give birth, she thinks the waiting is the worst. Kate thinks she'll be an amazing mother."""},
  {"dialogue": """Aubray: Hi! What r u doing tomorrow?
Kate: Nothing special
Aubray: How about movie?
Kate: What kind of?
Aubray: something funny?
Kate: comedy you say.. is there anything worth watching?
Aubray: there's this new movie with SRK
Kate: please, don't say it's one of your Bollywood thing
Aubray: well, yes... but this one even you will like
Kate: How do you know? I realy can't stand all this singing and dancing
Aubray: Don't you find it a little bit funny? You can realy stop thinking for a while and just enjoy :D
Kate: yeah, cause watching 3h movie in a weird language is such a joy.
Aubray: oh please, I realy want to watch it!
Kate: u know I don't like that stuff
Aubray: pretty, pretty please? 4 the last time? If u tell me after that u r done with it I'll never ask again
Kate: 4 real?
Aubray: Yes, I swear
Kate: ok, I go, 4 the last time
Aubray: thank u tahnk u thank u :*
Kate: yeah, yeah. See u tomorrow""",
  "summary": """Aubray wants to watch Bollywood movie with Kate tomorrow. Kate doesn't like this type of movie. In the end, she agrees to join Aubray."""}]

wmt_fr_en_samples = [{
"en": "It says that this should be done despite the principle of relative stability.",
"fr": "Il précise que cela devrait être fait malgré le principe de stabilité relative."
},
{
"en": "We know, and we have stated as much in very many resolutions indeed, including specifically during the last plenary part-session of last year, that this is not solely a legal case and that it is wrong for Alexander Nikitin to be accused of criminal activity and treason because of our involvement as the beneficiaries of his findings.",
"fr": "Nous savons, et nous l'avons d'ailleurs établi dans de très nombreuses résolutions - y compris lors de la dernière période de session de l'année dernière -, que ce cas n'est pas seulement de nature juridique et qu'il est faux d'accuser Alexandre Nikitin d'activité criminelle et de trahison car nous sommes concernés par ses résultats et nous en profitons."
},
{
"en": "I shall also refer the matter to the College of Quaestors, and I am certain that they will be keen to ensure that we comply with the regulations we ourselves vote on.",
"fr": "Je vais soumettre également le problème au Collège des questeurs et je suis certaine que nos questeurs auront à cur de faire en sorte que nous respections la réglementation qu' en effet nous votons."
}]

wmt_de_en_samples = [{
"de": "Frau Präsidentin, zunächst besten Dank dafür, daß Sie Wort gehalten haben und nun in der ersten Sitzungsperiode des neuen Jahres das Angebot an Fernsehprogrammen in unseren Büros tatsächlich enorm erweitert ist.",
"en": "Madam President, I would firstly like to compliment you on the fact that you have kept your word and that, during this first part-session of the new year, the number of television channels in our offices has indeed increased considerably."
},
{
"de": "Und ich hatte noch darauf hingewiesen, die anderen Präsidentenkollegen werden sich noch daran erinnern, daß es nicht darum geht, ob man für oder gegen die Tobin-Steuer ist, sondern darum, ob wir bereit sind, uns anzuhören, was die Kommission und der Rat davon halten.",
"en": "As my fellow chairmen will recall, I even mentioned that it was not a matter of knowing whether one was for or against the Tobin tax, but of whether one dared to hear what the Commission and the Council thought of it."
},
{
"de": "Ich bedauere das, aber die Abstimmung ist durchgeführt worden, die Entscheidung ist gefallen, also lassen wir die Dinge.",
"en": "I regret this, but the vote has already been taken and the decision is made so let us leave the matter there."
},
{
"de": "Patienten ohne Schmerzlinderung werden immer häufiger die Möglichkeit haben, auf diese terminale Sedierung zurückzugreifen.",
"en": "Increasingly, an unrelieved patient will have the option of having such palliative sedation."
},
{
"de": "Darüber hinaus werden durch diese Gesetze ebenfalls die Zeiträume für die vorzeitige Stimmabgabe verkürzt, das Recht für ungültig erklärt, sich am Wahltag als Wähler zu registrieren, und Staatsbürgern das Wahlrecht abgesprochen, für die eine Gerichtsakte vorliegt.",
"en": "Furthermore, these laws also reduce early voting periods, invalidate the right to register as a voter on election day and withdraw the right to vote of citizens with a criminal record."
}]


def doc_to_text_wmt_fr(item, from_lang = 'fr', to_lang = 'en'):
  return f"{languages[from_lang]} source: {item['translation'][from_lang]}\n{languages[to_lang]} translation:"

def doc_to_text_wmt_fr_inst(item, from_lang = 'fr', to_lang = 'en'):
  return f"Translate  the following from {languages[from_lang]} to {languages[to_lang]}: {item['translation'][from_lang]}\nTranslation:"

def doc_to_text_wmt_fr_few_shot(item, from_lang = 'fr', to_lang = 'en'):
  return f"""Translate  the following from {languages[from_lang]} to {languages[to_lang]}: {wmt_fr_en_samples[0][from_lang]}\nTranslation: {wmt_fr_en_samples[0][to_lang]}\n
  Translate  the following from {languages[from_lang]} to {languages[to_lang]}: {wmt_fr_en_samples[1][from_lang]}\nTranslation: {wmt_fr_en_samples[1][to_lang]}\n
Translate  the following from {languages[from_lang]} to {languages[to_lang]}: {item['translation'][from_lang]}\nTranslation:"""

def doc_to_text_wmt_ru(item, from_lang = 'en', to_lang = 'ru'):
  return f"{languages[from_lang]} source: {item['translation'][from_lang]}\n{languages[to_lang]} translation:"

def doc_to_text_wmt_de(item, from_lang = 'de', to_lang = 'en'):
  return f"{languages[from_lang]} source: {item['translation'][from_lang]}\n{languages[to_lang]} translation:"

def doc_to_text_wmt_de_inst(item, from_lang = 'de', to_lang = 'en'):
  return f"Translate  the following from {languages[from_lang]} to {languages[to_lang]}: {item['translation'][from_lang]}\nTranslation:"

def doc_to_text_wmt_de_few_shot(item, from_lang = 'de', to_lang = 'en'):
  return f"""Translate  the following from {languages[from_lang]} to fluent, natural {languages[to_lang]}: {wmt_de_en_samples[0][from_lang]}\nTranslation: {wmt_de_en_samples[0][to_lang]}\nTranslate  the following from {languages[from_lang]} to fluent, natural {languages[to_lang]}: {wmt_de_en_samples[1][from_lang]}\nTranslation: {wmt_de_en_samples[1][to_lang]}\nTranslate  the following from {languages[from_lang]} to fluent, natural {languages[to_lang]}: {wmt_de_en_samples[3][from_lang]}\nTranslation: {wmt_de_en_samples[3][to_lang]}\n\nTranslate  the following from {languages[from_lang]} to fluent, natural {languages[to_lang]}: {wmt_de_en_samples[4][from_lang]}\nTranslation: {wmt_de_en_samples[4][to_lang]}\nTranslate  the following from {languages[from_lang]} to fluent, natural {languages[to_lang]}: {item['translation'][from_lang]}\nTranslation:"""


def doc_to_text_qa(item):
  return f"Provide a short answer without explanation.\n Question: {item['question']}\nShort Answer:"

def doc_to_text_nq(item):
  text = f"""Provide the specific answer to the following question. Do not include reasoning, explanation or conversational filler. Output only the required information as concisely as possible. If unsure or unable to answer, output only your best guess.
  Question: Who is hosting the superbowl in 2019?
Short Answer: Atlanta
Provide the specific answer to the following question. Do not include reasoning, explanation or conversational filler. Output only the required information as concisely as possible. If unsure or unable to answer, output only your best guess.
  Question: When was Puerto Rico acquired by the US?
Short Answer: December 1898
Provide the specific answer to the following question. Do not include reasoning, explanation or conversational filler. Output only the required information as concisely as possible. If unsure or unable to answer, output only your best guess.
  Question: {item['question']}?
Short Answer:"""
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
    prompt = f""" You are a dialogue summarization assistant.

    Dialogue:
{samsum_samples[1]["dialogue"]}

Summary: {samsum_samples[1]["summary"]}

--------------------------------------

Dialogue:
{samsum_samples[2]["dialogue"]}

Summary: {samsum_samples[2]["summary"]}

--------------------------------------

Write a concise, factual summary of the text below.
The summary should capture the main event or most important point, not every message individually.
Do not add new information.
Keep the summary brief and self-contained.

Dialogue:
{doc.get("dialogue")}

Summary:
"""
    return prompt

def doc_to_text_cnn(doc):
    text = doc.get("article")
    prompt = f""" You are a news summarization assistant.

    Article:
{cnn_samples[1]["article"]}

Summary: {cnn_samples[1]["highlights"]}

--------------------------------------

Article:
{cnn_samples[2]["article"]}

Summary: {cnn_samples[2]["highlights"]}

--------------------------------------

Article:
{text}

Write a very short summary of the article in less than 100 words.
Paraphrase the content, do not copy it.
Focus on the main events and outcomes.

Summary:"""
    return prompt

def doc_to_summary_cnn(doc):
   return doc.get("highlights").strip()

def doc_to_text_xsum(doc):
    text = doc.get("document")
    if len(text) > 12000:
      print("Cropping *************************************")
      text = text[:12000]
    prompt = f""" You are a news summarization assistant.

    Article:
{xsum_samples[1]["article"]}

Summary: {xsum_samples[1]["summary"]}

--------------------------------------

Article:
{xsum_samples[2]["article"]}

Summary: {xsum_samples[2]["summary"]}

--------------------------------------

Article:
{text}

Write a very short, one sentence summary of the article.
Paraphrase the content, do not copy it.
Focus on the main events and outcomes.

Summary:"""
    return prompt

def doc_to_summary_xsum(doc):
   return doc.get("summary").strip()


def doc_to_summary(doc):
    return doc["summary"]

wmt14 = {"clean_name": "wmt14fr-en",
        "dataset_name": "fr-en",
        "dataset_location": "wmt/wmt14",
        "local_file": None,
        "options": None,
        "subset": "train",
        "task_type": "translation",
        "dict_ans": False, 
        "shuffle": False,
        "doc_to_text": doc_to_text_wmt_fr_few_shot,
        "doc_to_ans": doc_to_answer_wmt_fr}

triviaqa = {"clean_name": "TriviaQA",
        "dataset_name": "rc",
        "dataset_location": "mandarjoshi/trivia_qa",
        "local_file": None,
        "options": None,
        "subset": "train",
        "task_type": "qa",
        "dict_ans": True, 
        "shuffle": False,
        "doc_to_text": doc_to_text_qa,
        "doc_to_ans": doc_to_answer_qa}


hotpotqa = {"clean_name": "HotpotQA",
        "dataset_name": "distractor",
        "dataset_location": "hotpotqa/hotpot_qa",
        "local_file": None,
        "options": None,
        "subset": "train",
        "task_type": "qa",
        "dict_ans": True, 
        "shuffle": False,
        "doc_to_text": doc_to_text_hotpot,
        "doc_to_ans": doc_to_ans_hotpot}


nqopen = {"clean_name": "NQOpen",
        "dataset_name": "nq_open",
        "dataset_location": "google-research-datasets/nq_open",
        "local_file": None,
        "options": None,
        "subset": "train",
        "task_type": "qa",
        "dict_ans": False, 
        "shuffle": False,
        "doc_to_text": doc_to_text_nq,
        "doc_to_ans": doc_to_answer_qa}

truthfulqa = {"clean_name": "TruthfulQA",
        "dataset_name": "generation",
        "dataset_location": "truthfulqa/truthful_qa",
        "local_file": None,
        "options": None,
        "subset": "validation",
        "task_type": "qa",
        "dict_ans": True, 
        "shuffle": False,
        "doc_to_text": doc_to_text_truthful,
        "doc_to_ans": doc_to_answer_truthful}

wmt14ru = {"clean_name": "wmt14ru-en",
        "dataset_name": "ru-en",
        "dataset_location": "wmt/wmt14",
        "local_file": None,
        "options": None,
        "subset": "test",
        "task_type": "translation",
        "dict_ans": False, 
        "shuffle": False,
        "doc_to_text": doc_to_text_wmt_ru,
        "doc_to_ans": doc_to_answer_wmt_ru}

wmt19de = {"clean_name": "wmt19de-en",
        "dataset_name": "de-en",
        "dataset_location": "wmt/wmt19",
        "local_file": None,
        "options": None,
        "subset": "train",
        "task_type": "translation",
        "dict_ans": False, 
        "shuffle": False,
        "doc_to_text": doc_to_text_wmt_de_few_shot,
        "doc_to_ans": doc_to_answer_wmt_de}


wmt14de = {"clean_name": "wmt14de-en",
        "dataset_name": "de-en",
        "dataset_location": "wmt/wmt14",
        "local_file": None,
        "options": None,
        "subset": "train",
        "task_type": "translation",
        "dict_ans": False, 
        "shuffle": False,
        "doc_to_text": doc_to_text_wmt_de_few_shot,
        "doc_to_ans": doc_to_answer_wmt_de}

sciq = {"clean_name": "SciQ",
        "dataset_name": "default",
        "dataset_location": "allenai/sciq",
        "local_file": None,
        "options": None,
        "subset": "train",
        "task_type": "qa",
        "dict_ans": False, 
        "shuffle": False,
        "doc_to_text": doc_to_text_sciq,
        "doc_to_ans": doc_to_answer_sciq}

xsum = {
    "clean_name": "XSum",
    "dataset_name": "default",
    "dataset_location": "EdinburghNLP/xsum",
    "local_file": None,
    "options": None,
    "subset": "train",
    "task_type": "summarization",
    "dict_ans": False, 
    "shuffle": False,
    "doc_to_text": doc_to_text_xsum,
    "doc_to_ans": doc_to_summary_xsum,
}

samsum = {
    "clean_name": "SAMSum",
    "dataset_name": "default",
    "dataset_location": "knkarthick/samsum",
    "local_file": None,
    "options": None,
    "subset": "train",
    "task_type": "summarization",
    "dict_ans": False, 
    "shuffle": False,
    "doc_to_text": doc_to_text_summarization,
    "doc_to_ans": doc_to_summary,
}

cnn_dailymail = {
    "clean_name": "CNN_Daily Mail",
    "dataset_name": "3.0.0",
    "dataset_location": "abisee/cnn_dailymail",
    "local_file": None,
    "options": None,
    "subset": "train",
    "task_type": "summarization",
    "dict_ans": False, 
    "shuffle": False,
    "doc_to_text": doc_to_text_cnn,
    "doc_to_ans": doc_to_summary_cnn,
}


def doc_to_text_strategyqa(item):
  return f"Provide a short answer without explanation.\n Answer 'Yes' or 'No'.\nQuestion: {item['question']}\nShort Answer:"

strategyqa = {
    "clean_name": "StrategyQA",
    "dataset_name": None,
    "dataset_location": "tasksource/strategy-qa",
    "local_file": None,
    "options": ["Yes", "No"],
    "subset": "train",
    "task_type": "qa",
    "dict_ans": False, # Usually stored as a direct boolean/string 
    "shuffle": False,
    "doc_to_text": doc_to_text_strategyqa, # Needs to prompt for "Step-by-step"
    "doc_to_ans": doc_to_answer_qa
}

def doc_to_text_musique(item):
  return f"Provide a short answer without explanation.\n Question: {item['question']}\nShort Answer:"

musique = {
    "clean_name": "MuSiQue",
    "dataset_name": "multihop_reasoning",
    "dataset_location": "dgslibisey/MuSiQue",
    "local_file": None,
    "options": None,
    "subset": "train",
    "task_type": "qa",
    "dict_ans": True, # MuSiQue answers are usually in a list/dict format 
    "shuffle": False,
    "doc_to_text": doc_to_text_musique,
    "doc_to_ans": doc_to_answer_qa
}

def doc_to_text_mmlu(item):
    choices = item.get('choices', [])
    
    formatted_choices = []
    for i, choice in enumerate(choices):
        letter = string.ascii_uppercase[i]
        formatted_choices.append(f"{letter}. {choice}")
    
    choices_str = "\n".join(formatted_choices)
    
    return f"""Answer the following multiple choice question with only the letter corresponding to the correct answer.
Question: {item['question']}
Options: {choices_str}\n

IMPORTANT INSTRUCTION - Answer ONLY with the correct option letter.\n"

Best answer option: """
    

def doc_to_choices_mmlu(item):
    """Return the list of choice strings for an MMLU item.

    MMLU schema: choices is a list[str] (almost always length 4).
    """
    choices = item.get("choices")
    if not choices:
        raise ValueError(
            f"MMLU item missing choices: keys={list(item.keys())}"
        )
    return list(choices)


def doc_to_choices_arc(item):
    """Return the list of choice strings for an ARC item.

    ARC schema: choices = {"text": [str, ...], "label": [str, ...]}.
    Choice count varies (mostly 4, occasionally 3 or 5). We return only
    the text list; labels are handled by doc_to_text_arc downstream.
    """
    raw_choices = item.get("choices", {})
    texts = raw_choices.get("text", [])
    if not texts:
        raise ValueError(
            f"ARC item missing choice text: keys={list(item.keys())}"
        )
    return list(texts)



def doc_to_answer_mmlu(item):
    ans = item.get('answer', item.get('Answer'))
    # If the answer is already a letter, return it; if it's an index, map it.
    if isinstance(ans, int):
        return string.ascii_uppercase[ans]
    return str(ans)

mmlu = {
    "clean_name": "MMLU",
    "dataset_name": "all",
    "dataset_location": "cais/mmlu",
    "local_file": None,
    "options": ["A", "B", "C", "D"],
    "subset": "test",
    "task_type": "multiple_choice",
    "dict_ans": False,
    "shuffle": True,
    "doc_to_text": doc_to_text_mmlu,        
    "doc_to_ans": doc_to_answer_mmlu,       
    "doc_to_choices": doc_to_choices_mmlu,    
}

def axis_follow_up_prompt(axes = ["contextual information retrieval", "reasoning tokens"]):
    options = ""
    for i, a in enumerate(axes):
       options = options + f"{i+1}. {a}\n"
    additional_prompt = f"""Additional resources of which type would be more useful in answering the above question:
    {options}
    Type: """
    return additional_prompt


def doc_to_text_arc(item):
    # ARC-Challenge nests 'text' inside the 'choices' dict
    choices_data = item.get('choices', {})
    choices = choices_data.get('text', [])
    labels = choices_data.get('label', [])
    
    formatted_choices = []
    for i, choice in enumerate(choices):
        # Fallback to standard alphabetical letters if labels are missing or numeric
        # though ARC typically provides its own labels (A, B, C, D or 1, 2, 3, 4)
        letter = labels[i] if i < len(labels) else string.ascii_uppercase[i]
        formatted_choices.append(f"{letter}. {choice}")
    
    choices_str = "\n".join(formatted_choices)
    
    return f"""Answer the following multiple choice question with only the letter corresponding to the correct answer.
Question: {item['question']}
Choices: {choices_str}\n
Answer: """

def doc_to_answer_arc(item):
    # ARC uses 'answerKey' instead of 'answer'
    ans = item.get('answerKey', '')
    
    # If the dataset used numeric strings (e.g., '1', '2') instead of ('A', 'B')
    # map them safely back to letters to keep downstream parsing uniform
    if str(ans).isdigit():
        idx = int(ans) - 1 # 1-indexed to 0-indexed
        if 0 <= idx < 26:
            return string.ascii_uppercase[idx]
            
    return str(ans)


arc_challenge = {
    "clean_name": "ARC-Challenge",
    "dataset_name": "ARC-Challenge",
    "dataset_location": "allenai/ai2_arc",
    "local_file": None,
    "options": ["A", "B", "C", "D", "E"],
    "subset": "train+test",
    "task_type": "multiple_choice",
    "dict_ans": False,
    "shuffle": False,
    "doc_to_text": doc_to_text_arc,         
    "doc_to_ans": doc_to_answer_arc,        
    "doc_to_choices": doc_to_choices_arc,     
}
   

def doc_to_text_gooaq(item):
    # Free-form short-answer QA prompt, matching the TriviaQA "Short Answer:" style
    return (
        "Provide a short answer without explanation.\n"
        f"Question: {item['question'].strip()}\n"
        "Short Answer:"
    )


def doc_to_answer_gooaq(item):
    # GooAQ has a single gold answer string per question.
    # Returned as a list so the F1/scoring path treats it like the
    # other QA datasets (TriviaQA/NQ-Open) that allow multiple aliases.
    ans = item["answer"]
    if isinstance(ans, list):
        return [str(a).strip() for a in ans]
    return [str(ans).strip()]


gooaq = {"clean_name": "gooaq",
        "dataset_name": "rc",
        "dataset_location": "allenai/gooaq",
        "local_file": None,
        "options": None,
        "subset": "train",
        "task_type": "qa",
        "dict_ans": False, 
        "shuffle": False,
        "doc_to_text": doc_to_text_gooaq,
        "doc_to_ans": doc_to_answer_gooaq}



def doc_to_text_mixeval(item):
    options = item.get('options', [])
    
    formatted_choices = []
    for i, choice in enumerate(options):
        # Map indices to A, B, C, D, etc.
        letter = string.ascii_uppercase[i]
        formatted_choices.append(f"{letter}. {choice}")
    
    choices_str = "\n".join(formatted_choices)
    
    # MixEval sometimes uses both 'context' and 'prompt'
    context = item.get('context', '')
    prompt = item.get('prompt', '')
    
    if context and str(context).lower() != 'none':
        question_text = f"Context: {context}\nQuestion: {prompt}"
    else:
        question_text = f"Question: {prompt}"
    
    text =  f"""Answer the following question with only the correct answer.
{question_text}\n"""
    if len(options) > 1:
        text =text + f"Options: \n{choices_str}\n"
    text = text + """

IMPORTANT INSTRUCTIONS - Output ONLY the correct answer. DO NOT reference any context, sources or reasoning.

Answer: """
    return text

def doc_to_choices_mixeval(item):
    """Return the list of choice strings for a MixEval item."""
    options = item.get("options", [])
    if not options:
        raise ValueError(f"MixEval item missing options: keys={list(item.keys())}")
    return list(options)

def doc_to_answer_mixeval(item):
    """Return the target answer for a MixEval item."""
    target = item.get('target', [])
    options = item.get('options', [])
    if len(options) > 1:
       ans = [string.ascii_uppercase[int(target[0])]]
    else:
       ans = target
    return ans

mixeval_mc = {
    "clean_name": "MixEval-MC",
    "dataset_name": "MixEval", 
    "dataset_location": "MixEval/MixEval",
    "local_file": None,
    "options": ["A", "B", "C", "D", "E"], 
    "subset": "multiple_choice",
    "task_type": "multiple_choice",
    "dict_ans": False,
    "shuffle": False,
    "doc_to_text": doc_to_text_mixeval,
    "doc_to_ans": doc_to_answer_mixeval,
    "doc_to_choices": doc_to_choices_mixeval,
    "filter_fn": None,
    #"filter_fn": lambda x: x.get("problem_type") == "multiple-choice",
}

def doc_to_text_bbh(item):
    return f"Read the following problem and provide a short, exact answer.\n\n{item['input']}\n\nAnswer:"

def doc_to_answer_bbh(item):
    return str(item['target']).strip()

bbh_all = {
    "clean_name": "BBH-All",
    "dataset_name": "all_special",  # Custom trigger for the generation script
    "dataset_location": "lukaemon/bbh",
    "local_file": None,
    "options": None, 
    "subset": "test", 
    "task_type": "qa", 
    "dict_ans": False,
    "shuffle": True,        # We want to shuffle the concatenated dataset
    "doc_to_text": doc_to_text_bbh,
    "doc_to_ans": doc_to_answer_bbh,
    "filter_fn": None
}


mixeval_ff = {
    "clean_name": "MixEval-FF",
    "dataset_name": "MixEval", 
    "dataset_location": "MixEval/MixEval",
    "local_file": None,
    #"options": ["A", "B", "C", "D", "E"], 
    "subset": "free_form",
    "task_type": "qa",
    "dict_ans": True,
    "shuffle": False,
    "doc_to_text": doc_to_text_mixeval,
    "doc_to_ans": doc_to_answer_mixeval,
    "doc_to_choices": doc_to_choices_mixeval,
    "filter_fn": None,
    #"filter_fn": lambda x: x.get("problem_type") == "multiple-choice",
}

def doc_to_text_routerbench(item):
    """
    Extracts the prompt string for RouterBench items, stripping bracket/quote wrappers
    and removing multi-shot GSM8K preamble examples to enforce a clean 0-shot setting.
    """
    prompt = item.get("prompt", item.get("prompts", item.get("rb_prompt", "")))
    
    # Handle single-element list wrappers or stringified list representation "['...']"
    if isinstance(prompt, list) and len(prompt) > 0:
        prompt = prompt[0]
    elif isinstance(prompt, str) and prompt.startswith("['") and prompt.endswith("']"):
        prompt = prompt[2:-2]
    elif isinstance(prompt, str) and prompt.startswith('["') and prompt.endswith('"]'):
        prompt = prompt[2:-2]
        
    prompt = str(prompt).strip().replace('\\n', '\n')
    task_family = item.get("task_family", "")

    # --- Strip Hardcoded 5-Shot Examples from GSM8K Prompts ---
    if task_family == "grade-school-math" or "The following are examples of grade school math problems" in prompt:
        if "Shawn has five toys." in prompt:
            parts = prompt.split("Answer: 9\n\n")
            if len(parts) > 1:
                target_q = parts[-1].strip()
                return f"Solve the following grade school math problem and provide a numerical answer.\nQuestion: {target_q}\nAnswer:"
        
        # General regex cleanup for multi-shot math preambles
        examples_pattern = r"(?:The following are examples.*?\n\n)(?:Question:.*?\nAnswer:.*?\n\n)+"
        cleaned_prompt = re.sub(examples_pattern, "", prompt, flags=re.DOTALL).strip()
        if cleaned_prompt != prompt:
            return f"Solve the following grade school math problem and provide a numerical answer.\nQuestion: {cleaned_prompt}\nAnswer:"

    return prompt


def doc_to_answer_routerbench(item):
    """
    Returns the pre-aligned ground-truth target.
    
    Since target realignment is executed directly during dataset construction 
    (from full to the 10k stratified sample), this function extracts the clean 
    target string directly from the item dictionary, while providing robust fallbacks.
    """
    # 1. Primary: Use the pre-aligned target created during 10k sample building
    target = item.get("target")
    if target is not None and str(target).strip() != "":
        ans_str = str(target).strip()
        # Clean stringified list wrappers if present
        if ans_str.startswith("['") and ans_str.endswith("']"):
            ans_str = ans_str[2:-2]
        return ans_str.strip()

    # 2. Fallback: Check root answer fields
    ans = item.get("ans", item.get("answer"))
    
    # 3. Fallback: Extract from source_row metadata
    source_row = item.get("source_row")
    if ans is None and isinstance(source_row, dict):
        ans = source_row.get("label", source_row.get("answerKey", source_row.get("answer")))

    ans_str = str(ans).strip() if ans is not None else ""
    task_family = item.get("task_family", "")

    # Task-specific fallback index-to-letter mappings
    if task_family == "winogrande":
        if ans_str == "1":
            return "A"
        elif ans_str == "2":
            return "B"
    elif task_family in ["hellaswag", "mmlu"] and ans_str.isdigit():
        idx = int(ans_str)
        if 0 <= idx < 26:
            return string.ascii_uppercase[idx]
    elif task_family == "arc-challenge" and ans_str.isdigit():
        idx = int(ans_str) - 1
        if 0 <= idx < 26:
            return string.ascii_uppercase[idx]

    return ans_str.upper() if len(ans_str) == 1 and ans_str.isalpha() else ans_str


def doc_to_choices_routerbench(item):
    """
    Extracts choice option list from source_row metadata or directly parses 
    the choices presented in the prompt string if source_row is absent.
    """
    source_row = item.get("source_row")
    if isinstance(source_row, dict):
        choices = source_row.get("choices")
        # ARC format: {'text': [...], 'label': [...]}
        if isinstance(choices, dict) and "text" in choices:
            return choices["text"]
        # MMLU / HellaSwag format: ['choice1', 'choice2', ...]
        elif isinstance(choices, list):
            return choices

    # Fallback: Parse choices directly from prompt text
    prompt = doc_to_text_routerbench(item)
    parsed_choices = []
    lines = prompt.split("\n")
    for line in lines:
        line_s = line.strip()
        if len(line_s) > 2 and line_s[0].upper() in ['A', 'B', 'C', 'D', 'E'] and line_s[1] in [')', '.', ':']:
            parsed_choices.append(line_s[2:].strip())

    return parsed_choices

routerbench_10k = {
    "clean_name": "RouterBench-10K",
    "dataset_name": "routerbench_10k",
    "dataset_location": "local_pickle",
    "local_file": "/content/drive/MyDrive/phase3/routerbench_recovered/routerbench_10k_sample.pkl",
    "options": ["A", "B", "C", "D", "E"],
    "subset": "train",
    "task_type": "mixed",  # Unified task container
    "dict_ans": False,
    "shuffle": True,
    "doc_to_text": doc_to_text_routerbench,
    "doc_to_ans": doc_to_answer_routerbench,
    "doc_to_choices": doc_to_choices_routerbench,
}