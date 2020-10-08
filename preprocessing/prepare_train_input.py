import nltk
import csv
import numpy as np
import requests
from itertools import groupby, zip_longest
from kb_pool import single_query

def read_from_pdf(file_path):
    content_as_list = []
    with open(file_path) as file:
        for line in file.readlines():
            line_as_list = nltk.word_tokenize(line) # good tokenizer will avoid splitting on each . 
            if len(line_as_list) >3:
                content_as_list = content_as_list + line_as_list
    i = (list(g) for _, g in groupby(content_as_list, key='.'.__ne__)) #TODO: dont split if next token starts with lowercase
    content = [a + b for a, b in zip(i, i)]
    return content

def read_index(index_path):
    entity_index = dict()
    with open(index_path) as file:
        for line in file.readlines():
            line_list = line.strip().split(',')
            if len(line_list) > 1 and not line_list[0].isdigit() and not line_list[0].isupper():
                name = line_list[0].lower()
                entity_index[name] = name.split()
    # TODO: Map entites with 'see ...' together
    return entity_index

def get_wikidata_facts(entities):
    # split the entites in groups of n to send them with one request
    batch_size= 5 # take n entities to put in a query together
    batches = [iter(entities)] * 5
    buckets = zip_longest(fillvalue=None, *batches)
    code_triples = []
    label_triples = []
    for i in buckets:
        entity_str = '"' + '"@en "'.join(list(i)) + '"' + '@en'
        c, l = single_query(entity_str)   
        code_triples.extend(c)
        label_triples.extend(l)
    print(label_triples)


def prepare_neg_data(neg_samples):
    """negative training samples: from each existing ep+rel pair, create
        #neg_samples samples that are not existing at the real data"""

def run(file_path, index_path):
    # reading from a .pdf converted to .txt:
    content = read_from_pdf(file_path)
    entity_index = read_index(index_path)
    get_wikidata_facts(entity_index)
    # for entity in entity_index.keys():
    #     single_query(entity)
    # entity = ['optimization', 'deep learning']
    # entity_str = '"' + '"@en "'.join(entity) + '"' + '@en'
    # single_query(entity_str)

    # take only unique 
    text_examples = set()
    rel_map = {}

    for sentence in content:

        sentence_iter = iter(enumerate(sentence))

        found_here = []
        found_here_on = []
        last_found = None
        last_found_on = None

        for i,token in sentence_iter:
            token = token.lower()
            candidates = [item for item in entity_index.items() if item[1][0]==token]
            if not candidates:
                continue
            # if we match single-word index token directly

            str_the_rest = ' '.join(sentence[i:]).lower()
            to_remove = []
            for c in candidates:
                str_c = ' '.join(c[1])
                seen_here = str_the_rest.find(str_c, 0, len(str_c))
                to_remove.append(seen_here)
            candidates = [candidate for seen, candidate in zip(to_remove, candidates) if seen>-1]
            if not candidates:
                continue
            single_candidate = max(candidates, key=lambda item: len(item[1]))
            # iter only if something was found:
            start_pos = i
            end_pos = i +1
            for skip in range(len(single_candidate[1])-1):
                i, _ = next(sentence_iter)
                end_pos = i +1
            span = (start_pos, end_pos)
            w = single_candidate[0]
            if last_found != None and last_found != w:
                # apply aliases from list of aliases
                found_here.append((last_found, w))  
                found_here_on.append((last_found_on, span[0]))
            last_found_on = span[1]
            last_found = w
        relations = [' '.join(sentence[i[0]:i[1]]) for i in found_here_on if i[0]!=i[1]]
        for rel_str in relations:
            rel_map.setdefault(rel_str, str(len(rel_map) + 1))
        result = []
        for e, i in zip(found_here, found_here_on):
            if i[0]!=i[1]:
                seq = ' '.join(sentence[i[0]:i[1]])
                result.append((e[0], e[1], '\t'.join([e[0], e[1]]), rel_map[seq], '$ARG1 '+ seq +' $ARG2'))
        # result = [ (e[0] + '\t' + e[1], e[0], e[1],  '$ARG1 '+' '.join(sentence[i[0]:i[1]])+' $ARG2') for e, i in zip(found_here, found_here_on) if i[0]!=i[1]]

        text_examples.update(result)
    return list(text_examples)

def wikidata_query(query):
    url = 'https://query.wikidata.org/sparql'
    data = requests.get(url, params={'query': query, 'format': 'json'}).json()
    return len(data['results']['bindings'])

    #TODO: import to table with prefix  pre:...
    

if __name__ == "__main__":
    data = run("data/[11]part-2-chapter-6.txt", "data/[28]index.txt")
    print(data[3])
    with open('train.tsv','w') as out:
        csv_out=csv.writer(out, delimiter='\t')
        csv_out.writerow(['e1','e2', 'ep', 'relation_id', 'sequence', '1'])
        for row in data:
            row = row + (1,)
            csv_out.writerow(row)

    # query = '''    '''
    # print(wikidata_query(query))

    """
    16      10      392     493     3 1008 130 16 60 8 132 35 10 31 12 15 25 1009 9 35 10 105 12 14 5 4     1

    23      64      219     494     3 6 90 17 243 22 28 49 240 9 82 37 10 13 88 153 6 5 114 1010 11 8 1011 15 5 4   1

    87      5       393     495     3 11 27 4       1

    21      4       394     496     3 78 5 1012 13 5 178 9 5 4      1

    119     16      395     497     3 11 1013 13 39 80 6 95 5 4     1

    28      20      396     498     3 121 21 179 20 7 4     1
    """