import spacy
import math

nlp = spacy.load("en_core_web_sm")

def tokens_may_collapse(token_a_id, token_b_id, tokenizer):
    """
    Return True if token_a and token_b could represent the same string
    after continuing (i.e., one is a prefix of the other).
    """
    a_str = tokenizer.decode([token_a_id], clean_up_tokenization_spaces=False).strip().lower()
    b_str = tokenizer.decode([token_b_id], clean_up_tokenization_spaces=False).strip().lower()

    return a_str.startswith(b_str) or b_str.startswith(a_str)

def get_token_importance(pos):
    if pos in ["NOUN", "VERB", "ADJ", "PROPN", "NUM"]:
          offset = 0.0    # full weight
    elif pos in ["DET", "CONJ", "ADP", "PART", "PUNCT", "SPACE"]:
          offset = 0.8    # downweight
    else:
      offset = 0.5
    return offset

def get_pos(token):
    doc = nlp(token)
    if len(doc) >0:
        result = doc[0].pos_
    else:
        result = 'N/A'
    return result

