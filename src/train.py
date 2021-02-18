import torch
import torch.nn as nn
import torchtext.data as data
import tqdm
import copy

from torchnlp.nn import Attention
from encoders import LSTMEncoder
from attention import Attention

# --- prepare data tensors --- 

row = data.Field(sequential=False)
mention = data.Field(sequential=True)
mentions = data.NestedField(mention)
# TODO: try shared vocabulary here
column = data.Field(sequential=True)
label = data.Field(sequential=False)

dataset = data.TabularDataset(
path='data/final_dataset.json', format='json',
fields= {
    "entity_pair": ('row', row),
    "seen_with": ('mentions', mentions),
    "relation": ('column', column),
    "label": ('label', label)
    } 
)

row.build_vocab(dataset)
mentions.build_vocab(dataset)
column.build_vocab(dataset)
label.build_vocab(dataset)

train_iterator = data.BucketIterator(
    dataset=dataset, batch_size=3,
    sort_key=lambda x: len(x.relation),
    shuffle=True
    )

batch = next(iter(train_iterator))

class UniversalSchema(nn.Module):
    def __init__(self, params):
        super().__init__()

        row_vocab_size = len(row.vocab)
        col_vocab_size = len(column.vocab)
        self.row_encoder = nn.Embedding(row_vocab_size, params['emb_dim'])
        # encode the mentions with LSTM:
        self.mention_col_encoder = LSTMEncoder(col_vocab_size, params['emb_dim'], params['lstm_hid'])
        self.query_col_encoder = LSTMEncoder(col_vocab_size, params['emb_dim'], params['lstm_hid'])
        if params.get('pooling', None) == 'attention':
            # TODO: using tying for the attention encoder
            self.attention_col_encoder = LSTMEncoder(col_vocab_size, params['emb_dim'], params['lstm_hid'])
            self.attention = Attention(256)

    def forward(self, batch):
        """Row-less universal schema forward pass
        Args:
            batch: batch of (pair, mentions, column, label)
        """
        query = self.query_col_encoder(batch.column)#.permute(1,0))
        mentions = batch.mentions.permute(1,2,0)
        for col in mentions:
            embed = self.mention_col_encoder(col)
            print(embed)
            print('##############')
        return query


params = {'emb_dim': 50, 'lstm_encoder': True, 'pooling': 'attention', 'lstm_hid':5}   
model = UniversalSchema(params)


def train():
    loss_func = nn.BCEWithLogitsLoss()
    # https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html
    # https://discuss.pytorch.org/t/difference-between-cross-entropy-loss-or-log-likelihood-loss/38816/2
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

train()
