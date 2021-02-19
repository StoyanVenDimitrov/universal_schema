import torch
import torch.nn as nn
import torchtext.data as data
import tqdm
import copy

from torchnlp.nn import Attention
from utils import LSTMEncoder

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
            self.attention = Attention(params['lstm_hid'], attention_type='dot')

    def forward(self, batch):
        """Row-less universal schema forward pass
        Args:
            batch: batch of (pair, mentions, column, label)
        """
        query = self.query_col_encoder(batch.column)
        # men_len x seq_len x batch_size
        mentions = batch.mentions.permute(1,2,0)
        # mentions_embed = torch.empty(mentions.shape)
        
        li = []
        for col in mentions: # max num of mentions (columns)
            # seq_len x batch_size. This way LSTM gets what it needs - a column.
            embed = self.mention_col_encoder(col) # for the n-mention, in a batch 
            li.append(embed)
        mentions_embed = torch.stack(li, dim=1) #  batch_size x men_len x hidd_size
        if params['pooling'] == 'mean_pool':
            row_aggregation = torch.sum(mentions_embed, dim=1)
        if params['pooling'] == 'max_pool':
            row_aggregation = torch.max(mentions_embed, dim=1).values
        if params['pooling'] == 'max_relation':
            c_matmul = torch.matmul(mentions_embed, torch.unsqueeze(query,2))
            c_max_indices = torch.argmax(c_matmul, dim=1)
            c_max_gather = torch.unsqueeze(c_max_indices.repeat(1,params['lstm_hid']),1)
            # TODO: watch out that sometims the all-PAD seq is choosen:
            row_aggregation = torch.squeeze(torch.gather(mentions_embed, 1, c_max_gather),1)
        if params['pooling'] == 'attention':
            expanded_query = torch.unsqueeze(query, 1)
            _, weights = self.attention(expanded_query, mentions_embed)
            weighted_mentions = torch.mul(mentions_embed, torch.transpose(weights, 2,1))
            row_aggregation = torch.sum(weighted_mentions, dim=1)
        row_m = torch.unsqueeze(row_aggregation, 1)
        col_m = torch.unsqueeze(query, 2)
        score = torch.bmm(row_m, col_m)
        return torch.squeeze(score, dim=1)  # skip sigmoid if using BCEWithLogitsLoss


params = {'emb_dim': 50, 'lstm_encoder': True, 'pooling': 'attention', 'lstm_hid':5}   
model = UniversalSchema(params)


def train():
    loss_func = nn.BCEWithLogitsLoss() # (reduction='none')
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
