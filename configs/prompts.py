VERIFICATION_PROMPT = """Review the candidate answer below.

QUESTION:
{QUESTION}

CANDIDATE ANSWER:
{GENERATED_TEXT}

Your task is to produce the final answer to the QUESTION.

- If the CANDIDATE ANSWER is correct, reproduce the candidate answer exactly.
- If the CANDIDATE ANSWER is incorrect or incomplete, replace it with a corrected answer.
- Do not explain your decision.
- Do not discuss the candidate answer.
- Do not describe your reasoning or verification.
- Do not say whether the answer was correct.
- Output ONLY the final answer to the QUESTION.

FINAL ANSWER:
"""