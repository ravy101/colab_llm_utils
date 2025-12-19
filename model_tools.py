import torch

class tokenizer_embedder:
    def __init__(self, model, tokenizer, embed):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = tokenizer
        self.embed = embed

    def tokenize(self, text):
        return self.tokenizer(text, return_tensors="pt").to(self.device)

    def get_word_embedding(self, text):
      tokens = self.tokenize(text).input_ids
      with torch.no_grad():
        e = self.embed(tokens[0][-1])
      e.detach()
      return e.numpy()

    def get_token_embedding(self, token):
      with torch.no_grad():
        e = self.embed(torch.IntTensor([token]))
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

