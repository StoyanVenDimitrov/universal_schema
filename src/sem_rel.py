import copy
from datetime import datetime

import torch
import torch.nn as nn
import torchtext.data as data
import tqdm
from sklearn.metrics import confusion_matrix, f1_score
from torchnlp.nn import Attention

from utils import LSTMEncoder

# --- prepare data tensors --- 

row = data.Field(sequential=False)
mention = data.Field(sequential=True)
mentions = data.NestedField(mention)
# TODO: try shared vocabulary here
column = data.Field(sequential=True)
label = data.LabelField(dtype = torch.float, use_vocab=False, preprocessing=float)

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

test_dataset = data.TabularDataset(
path='data/test_dataset.json', format='json',
fields= {
    "entity_pair": ('row', row),
    "seen_with": ('mentions', mentions),
    "relation": ('column', column), 
    "label": ('label', label)
    } 
)

train_iterator = data.BucketIterator(
    dataset=dataset, batch_size=8,
    shuffle=True
    )

# ! The batch size must be always equal the number of rel classes
test_iterator = data.BucketIterator(
    dataset=test_dataset, batch_size=9,
    shuffle=False
    )


class UniversalSchema(nn.Module):
    def __init__(self, params):
        super().__init__()

        row_vocab_size = len(row.vocab)
        col_vocab_size = len(column.vocab)
        mentions_vocab_size = len(mentions.vocab)
        self.row_encoder = nn.Embedding(row_vocab_size, params['emb_dim'])
        # encode the mentions with LSTM:
        self.mention_col_encoder = LSTMEncoder(mentions_vocab_size, params['emb_dim'], params['lstm_hid'])
        #TODO: if no mentions as query, this can be a simple table:
        self.query_col_encoder = LSTMEncoder(col_vocab_size, params['emb_dim'], params['lstm_hid'])
        if params.get('pooling', None) == 'attention':
            # TODO: using tying for the attention encoder
            self.attention_col_encoder = LSTMEncoder(mentions_vocab_size, params['emb_dim'], params['lstm_hid'])
            self.attention = Attention(params['lstm_hid'], attention_type='dot')

    def forward(self, batch):
        """Row-less universal schema forward pass
        Args:
            batch: batch of (pair, mentions, column, label)
        """
        query = self.query_col_encoder(batch.column)
        # men_len x seq_len x batch_size
        mentions = batch.mentions.permute(1,2,0)
        
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
            # TODO: watch out that sometimes the all-PAD seq is choosen:
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


params = {'emb_dim': 100, 'lstm_encoder': True, 'pooling': 'attention', 'lstm_hid':64}   
model = UniversalSchema(params)


def train():
    loss_func = nn.BCEWithLogitsLoss() # (reduction='none')
    # https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html
    # https://discuss.pytorch.org/t/difference-between-cross-entropy-loss-or-log-likelihood-loss/38816/2
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    # model.train()
    for epoch in range(1):
        running_loss = 0.0
        for i, batch in enumerate(train_iterator,0): # enumerate(tqdm(train_iterator)): 
            x = batch
            y = torch.unsqueeze(batch.label.float(),1)
            opt.zero_grad()
            y_ = model(x)
            loss = loss_func(y_, y)
            # print(loss)
            loss.backward()
            opt.step()

            running_loss += loss.item()
            if i % 20 == 19:    # print every 2000 mini-batches
                print( running_loss / 20)
                running_loss = 0.0
    

    PATH = 'models/{mode}-{save_as}.pth'.format(save_as = datetime.today().strftime('%m-%d-%H:%M:%S'), mode=params['pooling'])
    torch.save(model.state_dict(), PATH)

def test():
    model = UniversalSchema(params)
    model.load_state_dict(torch.load("models/attention-04-03-14:09:23.pth"))
    classes = column.vocab.itos[2:] # col vocab without <unk> and <pad>
    predictions = []
    true_labels = []
    for i, example in enumerate(test_iterator,0):
        scores = model(example)
        output = classes[torch.argmax(scores)]
        true_label = classes[torch.argmax(example.label)]
        predictions.append(output)
        true_labels.append(true_label)
    print(confusion_matrix(true_labels,predictions))
    print(f1_score(true_labels,predictions, average='macro'))


        

        

# train()
test()
