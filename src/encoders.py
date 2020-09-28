import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=1)
        # self.linear = nn.Linear(hidden_dim, 1)
    
    def forward(self, seq): 
        emb = self.emb(seq)
        h, _ = self.lstm(emb)
        # take last hidden state only:
        # out= self.linear(h[-1, :, :])
        return h[-1, :, :]

class LookupTable:
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)

    def forward(self, seq): 
        emb = self.emb(seq)
        return emb