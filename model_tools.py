import torch
import os
import numpy as np
from transformers import AutoTokenizer

class tokenizer_embedder:
    def __init__(self,  embed, tokenizer):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = tokenizer
        self.embed = embed

    def tokenize(self, text):
        return self.tokenizer(text, return_tensors="pt").to(self.device)

    def get_word_embedding(self, text):
      tokens = self.tokenize(text).input_ids
      with torch.no_grad():
        e = self.embed[tokens[0][-1]]
      e.detach()
      return e.numpy()

    def get_token_embedding(self, token):
      with torch.no_grad():
        e = self.embed[torch.IntTensor([token])]
      e.detach()
      return e.numpy()

def get_embedding(token, model):
    with torch.no_grad():
        embed = model.encoder.embed_tokens(torch.IntTensor([token]))
    embed.detach()
    return embed.numpy()

def load_embeddings(weights_file):
    state_dict = torch.load(weights_file, map_location="cpu")
    vocab_size, embedding_dim = state_dict.shape

    print(f"Detected Vocab Size: {vocab_size}")
    print(f"Detected Embedding Dim: {embedding_dim}")

    #reloaded_emb = torch.nn.Embedding(vocab_size, embedding_dim)
    #reloaded_emb.load_state_dict(state_dict)
    #reloaded_emb.eval()
    #return reloaded_emb
    return state_dict

def get_or_load_embedding(base_path, embedding_model_config, input_embeddings = True):
    if input_embeddings:
       folder_name = "embeddings"
    else:
       folder_name = "out_embeddings"
    embed_file = os.path.join(base_path, folder_name, f"{embedding_model_config['model_name'].split('/')[-1]}_embed.pt")
    if os.path.exists(embed_file):
      embedding_layer = load_embeddings(embed_file)
    else:
      model = embedding_model_config['hf_model_func'].from_pretrained(embedding_model_config['model_name'], quantization_config=embedding_model_config['bnb_config'])
      if input_embeddings:
        embedding_layer = model.model.embed_tokens.weight.data
      else:
         embedding_layer = model.get_output_embeddings().weight.data
      os.makedirs(os.path.join(base_path, folder_name), exist_ok=True)
      torch.save(embedding_layer, embed_file)

    return embedding_layer


def token_logit_seq(logits, k = 10):
  seq = []
  for i in range(logits.shape[0]):
    early_logits = logits[i]
    tokens = {}
    for t in torch.topk(early_logits, k).indices:
      tokens[t.cpu().item()] = early_logits[t].cpu().item()
    seq.append(tokens)
  return seq

def get_full_probs(logit_dict, hs, lm_head, fp_type=torch.float16):
  candidate_tokens = []
  for i, o in enumerate(logit_dict):
    candidate_tokens.append(list(o.keys()))
  #print(f"hs shape {hs.shape}")
  #print(candidate_tokens)
  token_layer_probs = []
  for i in range(len(candidate_tokens)):
    layer_probs = []
    for j in range(len(hs)):
      token_probs = lm_head(hs[j][i].to(fp_type)).detach().squeeze().cpu().numpy()
      #print(f"i {i}, j {j}")
      #print(f"token probs shape {token_probs.shape}")
      layer_probs.append(token_probs[candidate_tokens[i]])
    if len(layer_probs) > 0:
      arr = np.stack(layer_probs)
      token_layer_probs.append(arr)
  if len(token_layer_probs) > 0:
    full_probs = np.stack(token_layer_probs)
  else:
    full_probs = []
  return full_probs

class AttnWrapper(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None
        self.add_tensor = None

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        if self.add_tensor is not None:
            output = (output[0] + self.add_tensor,)+output[1:]
        self.activations = output[0]
        return output

    def reset(self):
        self.activations = None
        self.add_tensor = None

class IdentityWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.identity = torch.nn.Identity()
        self.block_output = None

    def forward(self, *args, **kwargs):
        self.block_output = self.identity(*args)
        return self.block_output

class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block, unembed_matrix, norm):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm

        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.post_attention_layernorm = self.block.post_attention_layernorm
        try:
          self.attention_type = self.block.attention_type
        except:
          print("no attention type in block.")

        self.attn_mech_output_unembedded = None
        self.intermediate_res_unembedded = None
        self.mlp_output_raw = None
        self.mlp_output_unembedded = None
        self.block_output = None
        self.block_output_unembedded = None
        self.full_block_output = []


    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        #print(f"output type {type(output)}")
        #print(f"forward pass output shape of [0]{output[0].shape}")
        self.full_block_output.append(output[0][-1].squeeze())
        self.block_output = self.norm(output[0])
        #self.block_output_unembedded = self.unembed_matrix(self.block_output)
        attn_output = self.block.self_attn.activations
        #self.attn_mech_output_unembedded = self.unembed_matrix(self.norm(attn_output))
        #attn_output += args[0]
        #self.intermediate_res_unembedded = self.unembed_matrix(self.norm(attn_output))
        #mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        #self.mlp_output_raw = self.norm(mlp_output)
        #self.mlp_output_unembedded = self.unembed_matrix(self.mlp_output_raw)
        return output

    def attn_add_tensor(self, tensor):
        self.block.self_attn.add_tensor = tensor

    def reset(self):
        self.block.self_attn.reset()
        self.attn_mech_output_unembedded = None
        self.intermediate_res_unembedded = None
        self.mlp_output_raw = None
        self.mlp_output_unembedded = None
        self.block_output = None
        self.block_output_unembedded = None
        self.full_block_output = []

    def get_attn_activations(self):
        return self.block.self_attn.activations

    def get_mlp_activations(self):
        return self.mlp_output_raw

class LlamaHelper:
    def __init__(self, model_config, inference_config, existing_model = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_config['model_name'])
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.inference = inference_config

        self.last_out_len = None

        max_memory = {0: "75GiB", "cpu": "150GiB"}

        if existing_model is not None:
          self.model = existing_model
        else:
          self.model = model_config['hf_model_func'].from_pretrained(
              model_config['model_name'],
              quantization_config=model_config['bnb_config'],
              device_map="auto",
              max_memory=max_memory,
              offload_folder="offload",
              #torch_dtype=torch.bfloat16,
              dtype=torch.bfloat16,
              low_cpu_mem_usage=True
              #device_map=device_map
              )

          #.to(self.device)
          for i, layer in enumerate(self.model.model.layers):
            if model_config['block_limit'] is None or i < model_config['block_limit']:
              self.model.model.layers[i] = BlockOutputWrapper(layer, self.model.lm_head, self.model.model.norm)
            else:
              pass
              #self.model.model.layers[i] = IdentityWrapper()

    def generate_text(self, prompt, max_length=100):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        generate_ids = self.model.generate(inputs.input_ids.to(self.device), max_length=max_length)
        return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def get_sequence_logits(self, prompt, use_beams = False, stop_at='\n'):
      if self.inference['thinking']:
        messages = [
            {"role": "user", "content": prompt}
        ]

        # Explicitly enable thinking mode
        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True, 
            enable_thinking=True  # Set to False for non-thinking/fast mode
        )
        inputs = self.tokenizer([text], return_tensors="pt")   
      else:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
      input_len = len(inputs.input_ids[0])
      #max_len = int(input_len * (1 + OUTPUT_RATIO))

      max_len = int(input_len + self.inference['max_new_tokens'])
      meta_info = {}
      
         
      print(f"max len {max_len}")

      with torch.no_grad():
        #stop_token = self.tokenizer(stop_at, return_tensors="pt")['input_ids'][0][-1].to(self.device)
        #outputs = self.model.generate(inputs.input_ids.to(model.device), output_scores=True, return_dict_in_generate=True, max_length=max_len, repetition_penalty=1.3, early_stopping=True)
        if use_beams:
          outputs = self.model.generate(inputs.input_ids.to(self.model.device), output_scores=True, return_dict_in_generate=True, max_length=max_len, repetition_penalty=1.0, early_stopping=True, num_beams = 3, temperature=self.inference['temperature'])
        else:
          outputs = self.model.generate(inputs.input_ids.to(self.model.device), output_scores=True, tokenizer = self.tokenizer, return_dict_in_generate=True, eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id, stop_strings=self.inference['terminators'], max_length=max_len, repetition_penalty=self.inference['repetition_penalty'], early_stopping=True, do_sample=self.inference['do_sample'], top_k=self.inference['top_k'], temperature = self.inference['temperature'])


        print(self.tokenizer.batch_decode(outputs.sequences[0]))
        new_tokens = outputs.sequences[0][inputs.input_ids.shape[-1]:]
        text = self.tokenizer.batch_decode([new_tokens], skip_special_tokens=True, clean_up_tokenization_spaces=True)


      print(f"decoded text {text}")
      if use_beams:
        selected_beam_vocab_logits = []
        for i, b in enumerate(outputs.beam_indices[0]):
          selected_beam_vocab_logits.append(outputs.scores[i][b])
        logit_scores = torch.stack(selected_beam_vocab_logits)
      else:
        logit_scores = torch.stack([s.detach().squeeze() for s in outputs.scores] )
        print(f"output len {len(outputs.sequences[0])}")
        print(f"logit len {logit_scores.shape}")
      logit_seq = token_logit_seq(logit_scores)

      for l1, t1 in zip(logit_seq, new_tokens):
        int_keys = [int(k) for k in l1.keys()]
        if int(t1) not in l1.keys():
          print(f"new token {t1}")
          print(f"logit_seq {l1}")

      self.last_out_len = len(logit_seq)

      if not self.inference['skip_intermediates']:
        hs = self.get_hidden_states()
        if hasattr(self.model, "config") and hasattr(self.model.config, "quantization_config"):
          q_cfg = self.model.config.quantization_config
          if getattr(q_cfg, "load_in_4bit", False):
            target_dtype = q_cfg.bnb_4bit_compute_dtype
        
        if getattr(q_cfg, "load_in_8bit", False):
            target_dtype = self.model.dtype
        
        full_probs = get_full_probs(logit_seq, hs, self.get_head(), fp_type=target_dtype)
      else:
        full_probs = {}

      return text, outputs.sequences[0].detach().squeeze().cpu().numpy(), logit_seq, full_probs, meta_info#[:len(outputs.sequences[0]) -1]


    def get_logits(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
          logits = self.model(inputs.input_ids.to(self.device)).logits
          return logits

    def set_add_attn_output(self, layer, add_output):
        self.model.model.layers[layer].attn_add_tensor(add_output)

    def get_attn_activations(self, layer):
        return self.model.model.layers[layer].get_attn_activations()

    def reset_all(self):
        for layer in self.model.model.layers:
          if type(layer) == BlockOutputWrapper:
            layer.reset()

    def print_decoded_activations(self, decoded_activations, label):
        softmaxed = torch.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = torch.topk(softmaxed, 10)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        print(label, list(zip(tokens, probs_percent)))


    def decode_all_layers(self, text, topk=10, print_attn_mech=True, print_intermediate_res=True, print_mlp=True, print_block=True):
        self.get_logits(text)
        for i, layer in enumerate(self.model.model.layers):
            #print(f'Layer {i}: Decoded intermediate outputs')
            if print_attn_mech:
                self.print_decoded_activations(layer.attn_mech_output_unembedded, 'Attention mechanism')
            if print_intermediate_res:
                self.print_decoded_activations(layer.intermediate_res_unembedded, 'Intermediate residual stream')
            if print_mlp:
                self.print_decoded_activations(layer.mlp_output_unembedded, 'MLP output')
            if print_block:
                self.print_decoded_activations(layer.block_output_unembedded, 'Block output')

    def logits_and_intermediates(self, text):
        logits = self.get_logits(text)
        attn_outs = []
        mlp_outs = []
        block_outs = []
        for i, layer in enumerate(self.model.model.layers):
          if type(layer) == BlockOutputWrapper:
            attn_outs.append(layer.get_attn_activations().cpu())
            mlp_outs.append(layer.get_mlp_activations().cpu())
            block_outs.append(layer.full_block_output)
        return logits.cpu(), attn_outs, mlp_outs, block_outs

    def get_hidden_states(self):
      hidden_states = []
      for i, layer in enumerate(self.model.model.layers):
          if type(layer) == BlockOutputWrapper:
            #print(f"layer {i} is a block wrapper")
            #print(f"full block output shape {len(layer.full_block_output)}")
            hidden_states.append(layer.full_block_output[-self.last_out_len:])
      return hidden_states

    def get_head(self):
      return self.model.lm_head