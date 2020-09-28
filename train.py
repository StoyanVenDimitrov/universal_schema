import torch
import torch.nn as nn
import torchtext.data as data
import tqdm
import copy
from torch.utils.tensorboard import SummaryWriter

from src.encoders import LSTMEncoder, LookupTable
from src.attention import Attention

# --- prepare data tensors --- 

E1 = data.Field(sequential=False)
E2 = data.Field(sequential=False)
EP = data.Field(sequential=False)
REL = data.Field(sequential=False)
SEQ = data.Field()
LABEL = data.Field(use_vocab=False, sequential=False)

rel_set = data.TabularDataset(path='train.tsv', format='tsv', skip_header=True,
fields=[('e1', E1),
        ('e2', E2),
        ('ep', EP),
        ('rel', REL),
        ('seq', SEQ),
        ('label', LABEL)
        ])

SEQ.build_vocab(rel_set)
REL.build_vocab(rel_set)
E1.build_vocab(rel_set)
EP.build_vocab(rel_set)
E2.build_vocab(rel_set)
train_iterator = data.BucketIterator(
    dataset=rel_set, batch_size=6,
    sort_key=lambda x: len(x.seq)
    )
# TODO: check if no inverting is needed 

writer = SummaryWriter()

# --- declare models ----
class UniversalSchema(nn.Module):
    def __init__(self, params):
        super().__init__()
        # create basic encoders


        # TODO: check again the vocab_sizes in UniversalSchema.lua



        if params['lstm_encoder']:
            vocab_size = len(SEQ.vocab)
            self.col_encoder = LSTMEncoder(vocab_size, params['emb_dim'], 32)
            self.row_encoder = LSTMEncoder(vocab_size, params['emb_dim'], 32)
            
        else:
            row_vocab_size = len(EP.vocab)
            col_vocab_size = len(REL.vocab)
            self.row_encoder = nn.Embedding(row_vocab_size, params['emb_dim'])
            self.col_encoder = nn.Embedding(col_vocab_size, params['emb_dim'])
            if params['pooling'] == 'attention':
                # rel encoder to be applied with the attention weights
                self.col_encoder_output = copy.deepcopy(self.col_encoder)
                self.attention = Attention(params['emb_dim'])
    
    def forward(self, batch):

        # IMPORTANT: if there is only one relation per sample, all the pooling has no effect!

        # unsqueeze replaces the actual  len  of query and context and assumes one ep and one relation per sample
        rows = self.row_encoder(batch.ep).unsqueeze(1)
        relations_encoding = self.col_encoder(batch.rel).unsqueeze(1)
        if params['pooling'] == 'attention':
            # second encoder for relations. It is copied at the begining but then it's not the same 
            col_out = self.col_encoder_output(batch.rel).unsqueeze(1)
            relations = self.attention(rows, relations, col_out)
        if params['pooling'] == 'mean':
            # taking the mean of relations for one example. 
            # it assumes one or more relations per ep, but not one relation for more than one eps.
            relations = torch.unsqueeze(torch.mean(relations_encoding, 1), 1)
        # elementwise multiplication rows and cols, sum and sigmoid:
        match_scores = torch.sigmoid(torch.sum(torch.mul(rows, relations), -1))
         
        return match_scores
        # return self.row_encoder(seq)

        
params = {'emb_dim': 4, 'lstm_encoder': False, 'pooling': 'mean'}        
model = UniversalSchema(params)
# writer.add_graph(model)

# --- train loop ---
def train():
    loss_func = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

    model.train()
    #train_batches = list(train_iterator)
    for batch in tqdm.tqdm(train_iterator):
        x = batch
        y = torch.unsqueeze(batch.label.float(),1)
        y_ = model(x)
        loss = loss_func(y, y_)
        opt.zero_grad()
        loss.backward()
        opt.step()

# --- evaluate ---
def evaluate():
    # need a kb encoder and a text enocder. then just take the cosine distance