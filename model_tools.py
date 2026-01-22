import torch
import os

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
    vocab_size, embedding_dim = state_dict['weight'].shape

    print(f"Detected Vocab Size: {vocab_size}")
    print(f"Detected Embedding Dim: {embedding_dim}")

    reloaded_emb = torch.nn.Embedding(vocab_size, embedding_dim)
    reloaded_emb.load_state_dict(state_dict)
    reloaded_emb.eval()
    return reloaded_emb

def get_or_load_embedding(base_path, embedding_model_config, input_embeddings = True):
    if input_embeddings:
       folder_name = "embeddings"
    else:
       folder_name = "out_embeddings"
    embed_file = os.path.join(base_path, folder_name, f"{embedding_model_config['model_name'].split('/')[-1]}_embed.pt")
    if os.path.exists(embed_file):
      embedding_layer = load_embeddings(embed_file)
    else:
      model = embedding_model_config['hf_model_func'].from_pretrained(embedding_model_config['model_name'])
      if input_embeddings:
        embedding_layer = model.model.embed_tokens.weight.data
      else:
         embedding_layer = model.get_output_embeddings().weight.data
      os.makedirs(os.path.join(base_path, folder_name), exist_ok=True)
      torch.save(embedding_layer, embed_file)

    return embedding_layer
