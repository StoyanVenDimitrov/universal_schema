import copy
import json
import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchtext.data as data
import tqdm
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from torchnlp.nn import Attention

from utils import LSTMEncoder

for dim in [100, 50]:
    for epochs in [10, 15, 20]:
        for pooling in ['attention' ,'mean_pool', 'max_pool', 'max_relation']:
            params = {'emb_dim': dim, 'lstm_encoder': True, 'pooling': pooling, 'lstm_hid':64, 'epochs': epochs}   

            # --- prepare data tensors --- 

            row = data.Field(sequential=False)
            mention = data.Field(sequential=True)
            mentions = data.NestedField(mention)
            column = data.Field()
            neg_columns = data.NestedField(column)
            # label = data.LabelField()

            dataset = data.TabularDataset(
            path='data/final_dataset.json', format='json',
            fields= {
                "entity_pair": ('row', row),
                "seen_with": ('mentions', mentions),
                "relations": ('columns', neg_columns),
                "relation": ('column', column)
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
                "relations": ('columns', neg_columns),
                "relation": ('column', column)
                } 
            )

            train_iterator = data.BucketIterator(
                dataset=dataset, batch_size=5,
                shuffle=True
                )

            # ! The batch size must be always equal the number of rel classes
            test_iterator = data.BucketIterator(
                dataset=test_dataset, batch_size=2,
                shuffle=False
                )

            CLASSES = ['P279','P31','P1889','P361','P1552','P366','P460','P2283','P527']

            mypath = os.path.join(
                'models', 
                str(params['pooling']), 
                str(params['emb_dim']) + '_emb_dim', 
                str(params['lstm_hid']) + '_lstm_hid', 
                str(params['epochs'])+ '_epochs'
            )
            os.makedirs(mypath, exist_ok = True)

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
                    # self.query_col_encoder = nn.Embedding(col_vocab_size, params['lstm_hid'])
                    if params.get('pooling', None) == 'attention':
                        # TODO: using tying for the attention encoder
                        self.attention_col_encoder = LSTMEncoder(mentions_vocab_size, params['emb_dim'], params['lstm_hid'])
                        self.attention = Attention(params['lstm_hid'], attention_type='dot')

                def forward(self, batch):
                    """Row-less universal schema forward pass
                    Args:
                        batch: batch of (pair, mentions, column, label)
                    """
                    # men_len x seq_len x batch_size
                    mentions = batch.mentions.permute(1,2,0)
                    columns = batch.columns.permute(1,2,0)
                    
                    li = []
                    for mention in mentions: # max num of mentions (columns)
                        # seq_len x batch_size. This way LSTM gets what it needs - a column.
                        embed = self.mention_col_encoder(mention) # for the n-mention, in a batch 
                        li.append(embed)
                    mentions_embed = torch.stack(li, dim=1) #  batch_size x men_len x hidd_size

                    scores = []
                    for col in columns:
                        query = self.query_col_encoder(col)
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
                        scores.append(torch.squeeze(score.T))
                    try:
                        result = torch.stack(scores, dim=1)
                    except IndexError:
                        result = torch.unsqueeze(torch.stack(scores, dim=0),0)
                    return result  # skip sigmoid if using BCEWithLogitsLoss


            def train():
                # loss_func = nn.BCEWithLogitsLoss() # (reduction='none')
                loss_func = nn.CrossEntropyLoss() # (reduction='none')
                # https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html
                # https://discuss.pytorch.org/t/difference-between-cross-entropy-loss-or-log-likelihood-loss/38816/2
                opt = torch.optim.Adam(model.parameters())

                loss_values = []
                for epoch in range(params['epochs']):
                    running_loss = 0.0
                    for i, batch in enumerate(train_iterator,0): # enumerate(tqdm(train_iterator)): 
                        x = batch
                        # to share vocab, use batch.relation again. 
                        # but it's ALWAYS sequential, so skip UND and PAD by -2:
                        _y = torch.squeeze(batch.column[0].T) - 2
                        opt.zero_grad()
                        y_ = model(x)
                        loss = loss_func(y_, _y)
                        loss.backward()
                        opt.step()

                        running_loss += loss.item()
                        if i % 2 == 1: 
                            # print( running_loss / 2)
                            loss_values.append(running_loss / 2)
                            running_loss = 0.0
                

                PATH = '{path}/{save_as}.pth'.format(
                    save_as = datetime.today().strftime('%m-%d-%H:%M:%S'), 
                    path = mypath
                )
                torch.save(model.state_dict(), PATH)
                return model, loss_values

            def test(model=None):
                if not model:
                    model = UniversalSchema(params)
                    PATH = '{path}/04-06-16:59:25.pth'.format(
                        path = mypath
                    )
                    model.load_state_dict(torch.load(PATH))
                # classes = column.vocab.itos[2:] # col vocab without <unk> and <pad>

                predictions = []
                targets = []
                predicted_labels = []
                true_labels = []
                for i, example in enumerate(test_iterator,0):
                    scores = model(example)
                    results = torch.argmax(scores, dim=1)
                    predictions.extend(results)
                    target= torch.squeeze(example.column[0].T) - 2
                    targets.extend(target)
                predicted_labels.extend([column.vocab.itos[l+2] for l in predictions])
                true_labels.extend([column.vocab.itos[l+2] for l in targets])
                conf_matrix = confusion_matrix(true_labels,predicted_labels, labels=CLASSES)
                class_report = classification_report(true_labels,predicted_labels, output_dict=True)
                # print (f1_score(true_labels,predictions, average='macro'))
                # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                matrix_file = mypath + '/conf_matrix'
                with open(matrix_file, 'w') as fp:
                    np.savetxt(fp, conf_matrix,fmt='%10.1f')
                
                report = mypath + '/report.json'
                with open(report, 'w') as fp:
                    frame = pd.DataFrame(class_report)
                    frame.to_json(report, indent=4)

                    

            model = UniversalSchema(params)       

            trained_model, losses = train()
            test(trained_model)
            plt.plot(losses)
            loss_curve_file = mypath + '/loss.png'
            plt.savefig(loss_curve_file)
            del model
            del trained_model
            del losses
            plt.clf()