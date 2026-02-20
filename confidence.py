import math
from . import likelihood
from . import text
from . import misc
import numpy as np



def layer_decode_contrast(df, layer = 20, cont_weight = .2, tag = ''):
  all_contrasts = []
  all_logits = []
  all_probas = []
  for b, l, t in zip(df['blocks'], df['logit_outs'], df['token_outs']):
    n_sequence, n_layers, n_candidates = b.shape
    tokens = t[-len(l):]
    contrast_sequence = []
    logit_sequence = []
    top_probas = []
    for i in range(n_sequence):
      out_prob = likelihood.softmax_from_loglik(np.array(list(l[i].values())))
      layer_prob =  likelihood.softmax_from_loglik(b[i, layer, :])
      contrast = out_prob - layer_prob
      adjusted_probs = out_prob * (1 - cont_weight) + layer_prob * cont_weight
      logit_dic = {}
      top_probas.append(np.max(adjusted_probs))
      for j, k in enumerate(l[i].keys()):
        logit_dic[k] = adjusted_probs[j]
      logit_sequence.append(logit_dic)
      contrast_sequence.append(contrast)
    all_logits.append(logit_sequence)
    all_contrasts.append(np.array(contrast_sequence))
    all_probas.append(np.array(top_probas))
  df['top_probas'+ tag] = all_probas
  df['contrasts' + tag] = all_contrasts
  df['logit_outs_adj' + tag] = all_logits

def get_embedding_pos_dicts(df, embedder, tokenizer, suffix = ''):
    emb_dict = {}
    pos_dict = {}
    for logits in df['logit_outs' + suffix]:
        for l in logits:
            for k, v in l.items():
                if k not in emb_dict:
                    emb_dict[k] = embedder.get_token_embedding(k)
                    pos_dict[k] = text.get_pos(tokenizer.decode(k))
    for token_outs in df['token_outs' + suffix]:
        for token in token_outs:
            if token.item() not in emb_dict:
                emb_dict[token.item()] = embedder.get_token_embedding(token.item())
                pos_dict[token.item()] = text.get_pos(tokenizer.decode(token.item()))
    return emb_dict, pos_dict

def get_embedding_dict_from_pretrained(df, base_emb_dict, tokenizer, suffix = '', dim = 100):
    emb_dict = {}
    zeros_count = 0
    embed_count = 0
    
    for logits in df['logit_outs' + suffix]:
        for l in logits:
            for k, v in l.items():
                if k not in emb_dict:
                    try:
                        emb_dict[k] = base_emb_dict[tokenizer.decode(k).lower()]
                        embed_count += 1
                    except:
                        emb_dict[k] = np.zeros(dim)
                        zeros_count += 1

    for token_outs in df['token_outs' + suffix]:
        for token in token_outs:
            if token.item() not in emb_dict:
                try:
                    emb_dict[token.item()] = base_emb_dict[tokenizer.decode(token.item()).lower()]
                    embed_count += 1
                except:
                    emb_dict[token.item()] = np.zeros(dim)
                    zeros_count += 1

    print(f"tokens mapped to embeddings with {embed_count} successful vectors and {zeros_count} zeros.")
    return emb_dict


# it adjusts likelihoods based on the mean likelihood of a token (but only in the top 10)
def get_adj_likes(df):
    big_dict = {}
    big_dict_lens = {}
    for logits in df['logit_outs']:
        for l in logits:
            for k, v in l.items():
                if math.isinf(v):
                    continue
                if k not in big_dict:
                    big_dict[k] = []
                big_dict[k].append(v)
                    #break

    # get mean likes
    for k, v in big_dict.items():
        big_dict_lens[k] = len(v)
        big_dict[k] = np.array(v).mean()

    all_adj_likes = []
    for logits in df['logit_outs']:
        adj_likes = []
        for l in logits:
            token = list(l.keys())[0]
            like = l[token]
            adj_likes.append(like-big_dict[token])
        adj_likes = np.array(adj_likes)
        all_adj_likes.append(adj_likes)

    df['adj_likes'] = all_adj_likes
    df['adj_chow_av'] = [likelihood.chow_av(l) for l in df['adj_likes']]
    df['adj_chow_sum'] = [likelihood.chow_sum(l) for l in df['adj_likes']]
    df['adj_chow_quantile'] = [likelihood.chow_quantile(l) for l in df['adj_likes']]


def get_emb_likes(df, embedder, suffix=''):
    emb_dict = {}
    for logits in df['logit_outs' + suffix]:
        for l in logits:
            for k, v in l.items():
                if k not in emb_dict:
                    emb_dict[k] = embedder.get_token_embedding(k)
                #break
    for token_outs in df['token_outs' + suffix]:
        for token in token_outs:
            if token.item() not in emb_dict:
                emb_dict[token.item()] = embedder.get_token_embedding(token.item())
                #emb_dict[token.item()] = get_embedding(token.item(), model)
    all_dist_likes = []
    #for each response
    for logits, token_outs in zip(df['logit_outs' + suffix], df['token_outs' + suffix]):
        #list of candidate likes
        dist_likes = []
        output_tokens = token_outs[-len(logits):]
        #for each token in sequence
        for i, l in enumerate(logits):
            #lists of candidate tokens at position
            tokens = list(l.keys())
            probs = np.array(np.exp(list(l.values())))
            #print(probs)
            probs = probs/np.sum(probs)
            w_embeds = np.array([emb_dict[t].squeeze() * p for t, p in zip(tokens, probs)])
            w_sum = w_embeds.sum(axis=0)

            dist = misc.dist_mh(w_sum, emb_dict[output_tokens[i].item()].squeeze())

            dist_likes.append(dist)
        
        dist_likes = np.array(dist_likes)
        all_dist_likes.append(dist_likes)
    df['dist_likes'+suffix] = all_dist_likes
    df['dist_chow_av'+suffix] = [likelihood.chow_av(l) for l in df['dist_likes'+suffix]]
    df['dist_chow_sum'+suffix] = [likelihood.chow_sum(l) for l in df['dist_likes'+suffix]]
    df['dist_chow_quantile'+suffix] = [likelihood.chow_quantile(l) for l in df['dist_likes'+suffix]]

def poly_decay(distance, limit):
    return max(0, 1 - (distance/limit)**2)

def limit_decay(distance, limit):
    if distance > limit:
        res = 0
    else:
        res = 1
    return res

def get_cs_emb_likes(df, emb_dict, tokenizer, stopword_ids = [], logit_suffix='', token_suffix='', position_correct = True, skip_stopwords = True, collapse_prefix = True, tag = '', distance_limit = 2, sim_adjust = .5):
    all_dist_likes = []
    #for each response
    for logits, token_outs in zip(df['logit_outs' + logit_suffix], df['token_outs' + token_suffix]):
        #list of candidate likes
        dist_likes = []
        output_tokens = token_outs[-len(logits):]

        token_new_word = text.get_word_parts(tokenizer, output_tokens)

        #for each token in sequence
        for i, l in enumerate(logits):
            # embed for chosen token
            if skip_stopwords and output_tokens[i].item() in stopword_ids:
                continue


            chosen_emb = emb_dict[output_tokens[i].item()].squeeze()
            future_tokens = output_tokens[i+1:]

            #lists of candidate tokens at position
            tokens = list(l.keys())
            probs = likelihood.softmax_from_loglik(list(l.values()))
            sims = []
            for t in tokens:
                if t == output_tokens[i].item():
                    #this is the output token
                    sims.append(1)
                    continue

                # and token_new_word[i]
                if collapse_prefix and text.tokens_may_collapse(output_tokens[i].item(), t, tokenizer):
                    sims.append(1)
                elif position_correct and t in future_tokens:
                    distance = np.where(future_tokens == t)[0][0] + 1
                    decay = poly_decay(distance, distance_limit)
                    embed = emb_dict[t].squeeze()
                    sim = misc.generalized_gaussian_valley(misc.sim_cosine(chosen_emb, embed)) * sim_adjust
                    sims.append(max(sim, decay))
                else:
                    embed = emb_dict[t].squeeze()
                    sim = misc.generalized_gaussian_valley(misc.sim_cosine(chosen_emb, embed)) * sim_adjust
                    sims.append(sim)

            w_sims = np.array([s*p for s, p in zip(sims, probs)])
            w_sum = w_sims.sum(axis=0)
            dist_likes.append(max(w_sum, .01))

        if len(dist_likes) == 0:
            dist_likes.append(0)

        dist_likes = np.array(dist_likes)
        all_dist_likes.append(dist_likes)
    df[tag + '_cs_likes'+token_suffix] = all_dist_likes
    df[tag + '_cs_log_chow_av'] = [likelihood.log_chow_av(l) for l in all_dist_likes]
    df[tag + '_cs_chow_av'+token_suffix] = [likelihood.chow_av(l) for l in df[tag + '_cs_likes'+token_suffix]]
    df[tag + '_cs_cvar'+token_suffix] = [likelihood.chow_cvar_uncertainty(l) for l in df[tag + '_cs_likes'+token_suffix]]
    alpha = .4
    q1 = tag + f'_cs_chow_quant{alpha}'+token_suffix
    df[q1] = [likelihood.chow_quantile(l, alpha = alpha) for l in df[tag + '_cs_likes'+token_suffix]]
    alpha = .8
    q2 = tag + f'_cs_chow_quant{alpha}'+token_suffix
    df[q2] = [likelihood.chow_quantile(l, alpha = alpha) for l in df[tag + '_cs_likes'+token_suffix]]
    df[tag + f'_cs_chow_quant_m'+token_suffix]  = (df[q1]/2) + (df[q2]/2)
    


def get_cs_thresh_likes(df, emb_dict, pos_dict, tokenizer, stopword_ids = [], logit_suffix='', token_suffix='', position_correct = True, skip_stopwords = True, allow_empty = True, number_exception = False,
                        collapse_prefix = True, clip = 1, tag = '', distance_limit = 5, sim_thresh = .5, sem_pos_filt = [], lex_pos_filt=[]):
    all_dist_likes = []
    all_metadata = []

    #for each response
    for logits, token_outs in zip(df['logit_outs' + logit_suffix], df['token_outs' + token_suffix]):
        #list of candidate likes
        dist_likes = []
        metadata = {"lex_fragments": 0,
                    "lex_frag_weight": 0.0,
                    "position_adjustments": 0,
                    "position_adjust_weight": 0.0,
                    "semantic_collapses": 0,
                    "semantic_collapse_weight": 0.0,
                    "change_list": [],
                    "pos_list": []
                    }
        output_tokens = token_outs[-len(logits):]

        #token_new_word = text.get_word_parts(tokenizer, output_tokens)
        
        output_embeds = [emb_dict[out.item()].squeeze() for out in output_tokens]
        #for each token in sequence
        for i, l in enumerate(logits):
            # embed for chosen token
            if skip_stopwords and output_tokens[i].item() in stopword_ids:
                continue
                
            output_numeric = number_exception and misc.is_number(tokenizer.decode(output_tokens[i], clean_up_tokenization_spaces=True).strip())

            allow_sem_collapse =  pos_dict[output_tokens[i].item()] not in sem_pos_filt
            allow_lex_collapse = pos_dict[output_tokens[i].item()] not in lex_pos_filt

            chosen_emb = output_embeds[i]
            future_tokens = output_tokens[i+1:distance_limit+1]
            future_embeds = output_embeds[i+1:distance_limit+1]

            #lists of candidate tokens at position
            tokens = list(l.keys())
            probs = likelihood.softmax_from_loglik(list(l.values()))
            sims = []

            for j, t in enumerate(tokens):
                if t == output_tokens[i].item():
                    #this is the output token
                    output_prob = probs[j]
                    sims.append(1)
                    continue

                sim_result = 0
                embed = emb_dict[t].squeeze()
                future_sims = [misc.sim_cosine(f, embed) for f in future_embeds]
                max_future_sim =  max(future_sims + [0])
                if collapse_prefix and allow_lex_collapse and text.tokens_may_collapse3(output_tokens[i:], t, tokenizer, case_sensitive=False, allow_empty= allow_empty):
                    sim_result = 1
                    metadata['lex_fragments'] += 1
                    metadata['lex_frag_weight'] += probs[j]
                elif position_correct and (t in future_tokens or max_future_sim > .6):
                    
                    #distance = np.where(future_tokens == t)[0][0] + 1
                    #decay = limit_decay(distance, distance_limit)
                    #sim_result = decay
                    sim_result = 1
                    metadata['position_adjustments'] += 1
                    metadata['position_adjust_weight'] += probs[j]
                elif allow_sem_collapse:
                    
                    sim = misc.sim_cosine(chosen_emb, embed)
                    if (sim >= sim_thresh):
                        sim_result = 1
                        metadata['semantic_collapses'] += 1
                        metadata["semantic_collapse_weight"] += probs[j]
                sims.append(sim_result)

            w_sims = np.array([s*min(p,clip) for s, p in zip(sims, probs)])
            w_sum = w_sims.sum(axis=0)
            metadata['change_list'].append(w_sum - output_prob)
            metadata['pos_list'].append(pos_dict[output_tokens[i].item()])
            dist_likes.append(max(w_sum, .01))

        if len(dist_likes) == 0:
            dist_likes.append(0)

        all_metadata.append(metadata)
        dist_likes = np.array(dist_likes)
        all_dist_likes.append(dist_likes)
    df[tag + '_cs_likes'+token_suffix] = all_dist_likes
    df[tag + '_cs_log_chow_av'] = [likelihood.log_chow_av(l) for l in all_dist_likes]
    df[tag + '_cs_chow_av'+token_suffix] = [likelihood.chow_av(l) for l in df[tag + '_cs_likes'+token_suffix]]
    df[tag + '_cs_cvar'+token_suffix] = [likelihood.chow_cvar_uncertainty(l) for l in df[tag + '_cs_likes'+token_suffix]]
    df[tag + 'metadata'] = all_metadata
