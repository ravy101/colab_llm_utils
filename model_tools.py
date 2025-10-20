import torch

class tokenizer_embedder:
    def __init__(self, model, tokenizer, model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = tokenizer
        if 'llama' in model_name:
          self.embed = model.model.embed_tokens
        elif 't5' in model_name:
          self.embed = model.encoder.embed_tokens

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



