import spacy
import math
import numpy as np
import re
from numpy.linalg import norm
from typing import List, Dict, Union

nlp = spacy.load("en_core_web_sm")

def tokens_may_collapse(token_a_id, token_b_id, tokenizer, case_sensitive=True):
    """
    Return True if token_a and token_b could represent the same string
    after continuing (i.e., one is a prefix of the other).
    """
    space_equiv = ['\u2581', '\xa0', '\u0020', '\n', '\n\n']
    #removing strip here is this a bad idea?.strip().strip()
    a_str = tokenizer.decode([token_a_id], clean_up_tokenization_spaces=False).strip()
    b_str = tokenizer.decode([token_b_id], clean_up_tokenization_spaces=False).strip()
    
    if not case_sensitive:
        a_str = a_str.lower()
        b_str = b_str.lower()

    for s in space_equiv:
        a_str = a_str.replace(s, ' ')
        b_str = b_str.replace(s, ' ')

    return a_str.startswith(b_str) or b_str.startswith(a_str)

def tokens_may_collapse2(chosen_tokens, token_b_id, tokenizer, case_sensitive=True):
    """
    Return True if token_a and token_b could represent the same string
    after continuing (i.e., one is a prefix of the other).
    """
    space_equiv = ['\u2581', '\xa0', '\u0020', '\n', '\n\n']
    #removing strip here is this a bad idea?.strip().strip()
    a_str = tokenizer.decode(chosen_tokens, clean_up_tokenization_spaces=True).strip()
    b_str = tokenizer.decode([token_b_id], clean_up_tokenization_spaces=True).strip()
    
    if not case_sensitive:
        a_str = a_str.lower()
        b_str = b_str.lower()

    for s in space_equiv:
        a_str = a_str.replace(s, ' ')
        b_str = b_str.replace(s, ' ')

    return a_str.startswith(b_str) or b_str.startswith(a_str)

def tokens_may_collapse3(chosen_tokens, token_b_id, tokenizer, case_sensitive=True):
    """
    Return True if token_a and token_b could represent the same string
    after continuing (i.e., one is a prefix of the other).
    """
    space_equiv = ['\u2581', '\xa0', '\u0020', '\n', '\n\n']
    c1 = tokens_may_collapse(chosen_tokens[0].item(), token_b_id, tokenizer, case_sensitive=case_sensitive)
    c2 = tokens_may_collapse2(chosen_tokens, token_b_id, tokenizer, case_sensitive=case_sensitive)
    #removing strip here is this a bad idea?.strip().strip()
    a_str = tokenizer.decode(chosen_tokens, clean_up_tokenization_spaces=True).strip()
    b_str = tokenizer.decode([token_b_id], clean_up_tokenization_spaces=True).strip()
    
    #if c1 != c2:
    #    print(f"disagreement on {a_str} and {b_str}")

    if not case_sensitive:
        a_str = a_str.lower()
        b_str = b_str.lower()

    for s in space_equiv:
        a_str = a_str.replace(s, ' ')
        b_str = b_str.replace(s, ' ')

    return a_str.startswith(b_str)

def is_whitespace(token_id, tokenizer):
    text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False).lower().strip()
    return len(text) == 0

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

def remove_prompt(question, separator = ":"):
    _, _, content = question.partition(separator)
    return content


def is_new_word(tokenizer, token_id):
    """
    Determines if a token starts a new word by checking for space prefixes.
    Works for Llama 2 (SentencePiece) and Llama 3 (Byte-Level BPE).
    """
    if not type(token_id) == list:
        token_id = [token_id] 
    # Get the raw vocabulary string for the token
    token_str = tokenizer.convert_ids_to_tokens(token_id)
    
    if not token_str:
        return False
    
    token_str ="".join(token_str)
    # Llama 2 / SentencePiece Style (Lower One Eighth Block)
    if token_str.startswith(" ") or token_str.startswith("▁") or token_str[0].isdigit(): 
        return True
        
    # Llama 3 / GPT-2 Style (Byte-Level BPE mapped to unicode)
    # Ġ (U+0120) is the standard mapping for space in HF's BPE implementation
    if token_str.startswith("Ġ"):
        return True
        
    # Raw Tiktoken/Llama 3 (if using raw vocab without HF mapping)
    if token_str.startswith(" "):
        return True

    return False


def extract_number(text):
    """
    Finds the first integer in a string.
    "80%" -> "80"
    "(80)" -> "80"
    "Score: 80" -> "80"
    """
        
    # \d+ matches one or more digits
    match = re.search(r'\d+)', text)
    
    if match:
        result = np.float32(match.group())
    else:
        result = 0.0
    return result

def split_conf_ans(response, verbose = False):
    if type(response) == list:
        text = response[0]
    else:
        text = response
    text_list = text.split(')')
    conf_text = text_list[0]
   
    if len(text_list) > 1:
        conf = extract_number(conf_text)
        ans_text = text[len(conf_text)+2:]
    else:
        conf = 0.0
        ans_text = text

    if verbose:
        print(f"Original: {response}")
        print(f"self confidence: {conf}")
        print(f"remaining text: {ans_text}")

    return (conf, [ans_text])



def get_word_parts(tokenizer, token_ids):
    new_words = []
    for i in range(len(token_ids)):
        if i == 0:
            new_words.append(True)
        else:
            new_words.append(is_new_word(tokenizer, token_ids[i]))
    return new_words

GRAMMATICAL_TOKENS: List[str] = [
    "the", "a", "an", "is", "are", "was", "were", "be", "being", "been", 
    "and", "or", "but", "for", "of", "in", "to", "with", "on", "at", 
    "from", "by", "as", "that", "which", "who", "whom", "where", "when",
    ",", ".", "?", "!", ":", ";", "-", "(", ")", "[", "]", "'", '"', "\n"
]

_G_vector = None

def calculate_grammatical_direction(
    embedder, 
    gram_tokens: List[int],
) -> np.ndarray:
    """
    Calculates the Grammatical Direction Vector (G) by averaging the 
    embeddings of common function words and punctuation.
    """
    embeddings: List[np.ndarray] = []
    
    # 1. Collect embeddings for all grammatical tokens
    for token in gram_tokens:
        try:
            # We assume the embedder handles tokenization nuances (e.g., Llama's leading space)
            embedding = embedder.get_token_embedding(token[1]).squeeze()
            if embedding.sum() != 0: # Check if a valid (non-zero) embedding was returned
                embeddings.append(embedding)
        except Exception as e:
            print(f"Warning: Could not get embedding for token '{token}'. Error: {e}")
            continue

    if not embeddings:
        raise ValueError("No valid grammatical embeddings found. Check your embedder or token list.")

    # 2. Average the embeddings to get the direction G
    G_vector = np.mean(embeddings, axis=0)
    
    # 3. Normalize G (optional but recommended for stable projection)
    G_vector = G_vector / norm(G_vector)
    _G_vector = G_vector
    return G_vector

def semantic_transform(
    raw_embedding: np.ndarray,
    G_vector: np.ndarray
) -> np.ndarray:
    """
    Projects the raw embedding onto the Grammatical Direction Vector (G) 
    and subtracts the resulting component to yield the purely semantic vector.

    Formula: e_sem = e_raw - ( (e_raw . G) / ||G||^2 ) * G
    Since G is normalized, ||G||^2 = 1.
    Formula simplifies to: e_sem = e_raw - (e_raw . G) * G
    """
    if G_vector is None:
        raise ValueError("G vector note calculated.")
    # 1. Calculate the scalar component (projection coefficient)
    # The dot product (e_raw . G) gives the magnitude of the raw vector along G
    projection_scalar = np.dot(raw_embedding, G_vector)
    
    # 2. Calculate the grammatical component vector (e_gram)
    # e_gram = scalar * G_vector
    e_gram = projection_scalar * G_vector
    
    # 3. Subtract the grammatical component from the raw embedding
    # This leaves the orthogonal, semantic component
    e_sem = raw_embedding - e_gram
    
    # 4. Normalize the semantic vector (recommended for cosine similarity consistency)
    e_sem_normalized = e_sem / norm(e_sem)
    
    return e_sem_normalized
