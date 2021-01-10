# universal_schema for relations extraction from manuals 

The goal is extraction of the structured information from in a text, e.g. a manual or a textbook. It is very likely that the texts 
describe relations between entities, unseen in any other KB, so normal University schema is not going to work - some entities will simply have no embeddings at test time. We need to compute row, or entity pair, embeddings on the fly. These row embeddings are computed as aggregation of observed columns for the row we want to represent. We can take the mean of these vectors, but this will give the same embedding regardless of the query relation we want to predict for if it is hold by the row. We can use the observed column embedding that is most similar to the relation we are trying to predict. Another way to do query specific embeddings is by the weighted average of observed columns for this row, using attention. When we got a surface pattern/text between two entities, no matter seen or not, we can take the embedding of this surface pattern as row embedding, or aggregate more if there are more. This column embedding can be an explicit one for this surface pattern (then it works only with patterns seen at training time), or generated from a column encoder like a LSTM. The results in the paper show performance on known entity pairs similar to when using explicit row embeddings. But the performance with row-less Universal schema stays the same also for unseen entity pairs, while with explicit embeddings it logically drops drastically due to the inability to represent the unseen pair. 
Using surface patterns from the textbook only is too little training data. On the other hand, the same relation is probably expressed with the same surface patterns on the same domain, regardless of the exact entity pair. This is why I extract surface patterns from Wikipedia for entity pairs holding a specific relation on Wikidata, boosting the training data by orders of magnitude. Especially for short manuals, this is the only way to train a model.