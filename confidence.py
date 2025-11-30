import math
from . import likelihood
from . import text
from . import misc
import numpy as np


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
    for logits in df['logit_outs' + suffix]:
        for l in logits:
            for k, v in l.items():
                if k not in emb_dict:
                    try:
                        emb_dict[k] = base_emb_dict[tokenizer.decode(k).lower()]
                    except:
                        emb_dict[k] = np.zeros(dim)

    for token_outs in df['token_outs' + suffix]:
        for token in token_outs:
            if token.item() not in emb_dict:
                try:
                    emb_dict[token.item()] = base_emb_dict[tokenizer.decode(token.item()).lower()]
                except:
                    emb_dict[token.item()] = np.zeros(dim)
    return emb_dict


# it adjusts likelihoods based on the mean likelihood of a token (but only in the top 10)
def get_adj_likes(df):
    big_dict = {}
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

def get_cs_emb_likes(df, emb_dict, tokenizer, stopword_ids = [], suffix='', position_correct = True, skip_stopwords = True, collapse_prefix = True, tag = '', future_alpha = .9, sim_adjust = .5):
    all_dist_likes = []
    #for each response
    for logits, token_outs in zip(df['logit_outs' + suffix], df['token_outs' + suffix]):
        #list of candidate likes
        dist_likes = []
        output_tokens = token_outs[-len(logits):]
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

                if collapse_prefix and text.tokens_may_collapse(output_tokens[i].item(), t, tokenizer):
                    sims.append(1)
                elif position_correct and t in future_tokens:
                    distance = np.where(future_tokens == t)[0][0] + 1
                    decay = future_alpha**distance
                    embed = emb_dict[t].squeeze()
                    sim = misc.sim_cosine(chosen_emb, embed) * sim_adjust
                    sims.append(((1-decay)*sim + decay))
                else:
                    embed = emb_dict[t].squeeze()
                    sim = misc.sim_cosine(chosen_emb, embed) * sim_adjust
                    sims.append(sim)

            w_sims = np.array([s*p for s, p in zip(sims, probs)])
            w_sum = w_sims.sum(axis=0)
            dist_likes.append(w_sum)

        if len(dist_likes) == 0:
            dist_likes.append(0)

        dist_likes = np.array(dist_likes)
        all_dist_likes.append(dist_likes)
    df[tag + '_cs_likes'+suffix] = all_dist_likes
    df[tag + '_cs_log_chow_av'] = [likelihood.log_chow_av(l) for l in all_dist_likes]
    df[tag + '_cs_chow_av'+suffix] = [likelihood.chow_av(l) for l in df[tag + '_cs_likes'+suffix]]
    df[tag + '_cs_chow_sum'+suffix] = [likelihood.chow_sum(l) for l in df[tag + '_cs_likes'+suffix]]


def get_cs_semantic_emb_likes(df, embedder, tokenizer, stopword_ids = [], suffix='', position_correct = True, skip_stopwords = True, collapse_prefix = True, tag = '', future_alpha = .9, sim_adjust = .5):
    
    g = text.calculate_grammatical_direction(embedder, tokenizer(text.GRAMMATICAL_TOKENS)['input_ids'])

    emb_dict = {}
    pos_dict = {}
    for logits in df['logit_outs' + suffix]:
        for l in logits:
            for k, v in l.items():
                if k not in emb_dict:
                    emb_dict[k] = text.semantic_transform(embedder.get_token_embedding(k), g)
                    pos_dict[k] = text.get_pos(tokenizer.decode(k))
    for token_outs in df['token_outs' + suffix]:
        for token in token_outs:
            if token.item() not in emb_dict:
                emb_dict[token.item()] = text.semantic_transform(embedder.get_token_embedding(token.item()), g)
                pos_dict[token.item()] = text.get_pos(tokenizer.decode(token.item()))
    all_dist_likes = []
    #for each response
    for logits, token_outs in zip(df['logit_outs' + suffix], df['token_outs' + suffix]):
        #list of candidate likes
        dist_likes = []
        output_tokens = token_outs[-len(logits):]
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
        
                if collapse_prefix and text.tokens_may_collapse(output_tokens[i].item(), t, tokenizer):
                    sims.append(1)
                elif position_correct and t in future_tokens:
                    distance = np.where(future_tokens == t)[0][0] + 1
                    decay = future_alpha**distance
                    embed = emb_dict[t].squeeze()
                    sim = misc.sim_cosine(chosen_emb, embed)  * sim_adjust
                    sims.append(((1-decay)*sim + decay))
                else:
                    imp_offset = text.get_token_importance(pos_dict[t])
                    embed = emb_dict[t].squeeze()
                    sim = misc.sim_cosine(chosen_emb, embed) * sim_adjust
                    sims.append((1-imp_offset)*sim + imp_offset)

            w_sims = np.array([s*p for s, p in zip(sims, probs)])
            w_sum = w_sims.sum(axis=0)
            dist_likes.append(w_sum)

        if len(dist_likes) == 0:
            dist_likes.append(0)

        dist_likes = np.array(dist_likes)
        all_dist_likes.append(dist_likes)
    df[tag + '_cs_likes'+suffix] = all_dist_likes
    df[tag + '_cs_log_chow_av'] = [likelihood.log_chow_av(l) for l in all_dist_likes]
    df[tag + '_cs_chow_av'+suffix] = [likelihood.chow_av(l) for l in df[tag + '_cs_likes'+suffix]]
    df[tag + '_cs_chow_sum'+suffix] = [likelihood.chow_sum(l) for l in df[tag + '_cs_likes'+suffix]]

