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
        o, h = self.lstm(emb)
        # take last output state only:
        return o[-1, :, :]#.unsqueeze(1)
