import json

def score_qa_pair(client, question, answer, reference, api_model = "gpt-4o-mini"):
    """
    Scores a single QA pair against a reference (or list of references).
    Returns a JSON object with a score (0 or 1) and reasoning.
    """

    # Handle if reference is a list of strings
    if isinstance(reference, list):
        reference_text = " OR ".join([f"Option {i+1}: {r}" for i, r in enumerate(reference)])
        instruction_note = "The reference is a list of acceptable options. If the candidate answer has the same conclusionas a reference, mark it as correct even with missing or slightly differing justification. If the answer matches ANY of them, score it as Correct."
    else:
        reference_text = str(reference)
        instruction_note = "Compare the answer closely to the reference."

    # The System Prompt
    system_prompt = f"""
    You are an expert evaluator for an NLP project.
    Your task is to score a candidate answer based on a given question and reference answer.
    {instruction_note}

    Output valid JSON only:
    {{
        "score": 0 or 1,  # 1 for Correct/Equivalent, 0 for Incorrect
    }}
    """

    user_message = f"""
    Question: {question}
    Reference: {reference_text}
    Candidate Answer: {answer}
    """

    try:
        response = client.chat.completions.create(
            model=api_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"}, # Forces valid JSON output
            temperature=0 # Keep it deterministic
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"score": -1, "reasoning": f"Error: {str(e)}"}
    
def score_translation(client, source, candidate, reference):
    """
    Scores a candidate translation against a reference translation.

    Returns:
      - binary_adequacy: 0 or 1
      - adequacy: integer from 0 to 100
    """

    system_prompt = """
    You are an expert human evaluator for machine translation.

    Your task is to evaluate whether the candidate translation preserves
    the meaning of the source sentence.

    You will be given:
    - A source sentence
    - A reference translation
    - A candidate translation

    Use the reference ONLY as a guide to the intended meaning.
    Do NOT treat it as the only correct wording.
    Do NOT penalize valid paraphrases, alternative word orders,
    or stylistic differences if the meaning is preserved.

    Focus on ADEQUACY:
    - Missing information
    - Added or hallucinated information
    - Incorrect entities, relations, negation, tense, or modality

    Fluency and style should only affect the score if they obscure meaning.

    First, assign an adequacy score from 0 to 100:
      - 90–100: Meaning fully preserved (at most very minor issues)
      - 70–89: Mostly correct, minor omissions or imprecision
      - 40–69: Partially correct, major omissions or distortions
      - 0–39: Meaning largely incorrect or unrelated

    Then assign a binary adequacy score:
      - binary_adequacy = 1 if adequacy >= 70
      - binary_adequacy = 0 otherwise

    Output valid JSON only:
    {
        "adequacy": 0-100,
        "binary_adequacy": 0 or 1
    }
    """

    user_message = f"""
    Source: {source}
    Reference: {reference}
    Candidate: {candidate}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(response.choices[0].message.content)

    except Exception as e:
        return {
            "adequacy": -1,
            "binary_adequacy": -1,
            "error": str(e)
        }

def score_summary(client, dialogue, summary, api_model = "gpt-4o-mini"):
    """
    Scores a SAMSum-style dialogue summary.
    Returns a JSON object with:
      - faithful: "Yes" or "No"
      - score: integer from 0 to 100
    """

    system_prompt = """
    You are an expert evaluator for article summarization.

    Your task is to evaluate the quality of a summary of an article.

    Evaluation criteria:

    1. Faithful:
    - All statements in the summary must be directly supported by the article.
    - The summary must not add events, intentions, decisions, or facts not present.
    - Empty summaries must be marked unfaithful.
    - Incorrect speaker attributions count as unfaithful.

    2. Complete Coverage:
    - The summary should capture all main points or outcomes of the article.
    - Minor or casual details may be omitted.

    3. Clarity:
    - The summary should be clear, coherent, and understandable on its own.

    IMPORTANT:
    - First determine whether the summary is faithful.
    - Then determine whether it is sufficient.
    - If the summary is not faithful or not sufficient, the score must be 50 or lower.

    Output valid JSON only in the following format:
    {
      "faithful": 0 or 1,
      "complete": 0 or 1,
      "score": <integer from 0 to 100>
    }
    """

    user_message = f"""
    Article:
    {dialogue}

    Summary:
    {summary}
    """

    try:
        response = client.chat.completions.create(
            model=api_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(response.choices[0].message.content)

    except Exception as e:
        return {
            "faithful": "Error",
            "complete": "Error",
            "score": -1,
            "reasoning": f"Error: {str(e)}"
        }
    
def score_short_summary(client, dialogue, summary, limit = "three", doc_type="Article", crop = False, api_model = "gpt-4o-mini"):

    system_prompt = f"""
    You are an expert evaluator for {doc_type.lower()} summarization.

    Your task is to evaluate the quality of a {limit}-sentence summary of an {doc_type.lower()}.

    Evaluation criteria:

    1. Faithful:
    - All statements in the summary must be supported by the article.
    - The summary must not add events, intentions, decisions, or facts not present.
    - Empty summaries must be marked Not faithful.

    2. Complete:
    - The summary should capture all main points or outcomes of the {doc_type.lower()}.
    - Minor or casual details should be omitted.

    3. Clarity:
    - The summary should be clear, coherent, and understandable on its own.

    IMPORTANT:
    - First determine whether the summary is faithful.
    - Then determine whether it is sufficient.
    - Score the summary out of 100; with 0 to 40 points for faithfulness, 0 to 40 points for sufficiency and, 0 to 20 points for clarity.

    """ + """Output valid JSON only in the following format:
    {
      "faithful": 0 or 1,
      "complete": 0 or 1,
      "score": <integer from 0 to 100>
    }
    """
    if crop:
      dialogue = dialogue.split("Write a very short")[0]

    user_message = f"""
    Article:
    {dialogue}

    Summary:
    {summary}
    """

    try:
        response = client.chat.completions.create(
            model=api_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(response.choices[0].message.content)

    except Exception as e:
        return {
            "faithful": "Error",
            "complete": "Error",
            "score": -1,
            "reasoning": f"Error: {str(e)}"
        }

def score_short_summary_facts(client, dialogue, summary, limit = "three", doc_type="Article", crop = False, api_model = "gpt-4o-mini"):

    system_prompt = f"""
    You are an expert evaluator for {doc_type.lower()} summarization.

    Your task is to evaluate the quality of a {limit}-sentence summary of an {doc_type.lower()}.

    Evaluation criteria:

    1. Faithful:
    - All statements in the summary must be supported by the article.
    - The summary must not add events, intentions, decisions, or facts not present.
    - Empty summaries must be marked Not faithful.

    2. Complete:
    - The summary should capture all main points or outcomes of the {doc_type.lower()}.
    - Minor or casual details should be omitted.

    3. Clarity:
    - The summary should be clear, coherent, and understandable on its own.

    IMPORTANT:
    - First determine whether the summary is faithful.
    - Then determine whether it is sufficient.
    - Score the summary out of 100; with 0 to 40 points for faithfulness, 0 to 40 points for sufficiency and, 0 to 20 points for clarity.

    """ + """Output valid JSON only in the following format:
    {
      "faithful": 0 or 1,
      "complete": 0 or 1,
      "score": <integer from 0 to 100>
    }
    """
    if crop:
      dialogue = dialogue.split("Write a very short")[0]

    user_message = f"""
    Article:
    {dialogue}

    Summary:
    {summary}
    """

    try:
        response = client.chat.completions.create(
            model=api_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(response.choices[0].message.content)

    except Exception as e:
        return {
            "faithful": "Error",
            "complete": "Error",
            "score": -1,
            "reasoning": f"Error: {str(e)}"
        }
    