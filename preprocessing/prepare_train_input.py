import nltk
import csv
import numpy as np
import requests
import json
import os,glob
import itertools  
from itertools import groupby, zip_longest
from kb_pool import single_query, rel_from_domain

nltk.download('punkt')
def read_from_pdf(file_dir):
    content = []
    for filename in glob.glob(os.path.join(file_dir, '*.txt')):
        content_as_list = []
        with open(filename) as file:
            for line in file.readlines():
                line_as_list = nltk.word_tokenize(line) # good tokenizer will avoid splitting on each . 
                if len(line_as_list) >3:
                    content_as_list = content_as_list + line_as_list
        i = (list(g) for _, g in groupby(content_as_list, key='.'.__ne__)) #TODO: dont split if next token starts with lowercase
        chapter_content = [a + b for a, b in zip(i, i)]
        content.extend(chapter_content)
    return content

def read_index(index_path):
    entity_index = dict()
    with open(index_path) as file:
        for line in file.readlines():
            line_list = line.strip().split(',')
            if len(line_list) > 1 and not line_list[0].isdigit() and not line_list[0].isupper():
                # name = line_list[0]
                # entity_index[name] = name.split()
                name = line_list[0].lower()
                entity_index[name] = name.split()
    # TODO: Map entites with 'see ...' together
    return entity_index

def get_initial_wikidata_facts(entities):
    # split the entites in groups of n to send them with one request
    batch_size= 5 # take n entities to put in a query together
    batches = [iter(entities)] * batch_size
    buckets = zip_longest(fillvalue='__pad__', *batches)
    code_triples = dict()
    label_triples = dict()
    found_objects = dict()
    for i in buckets:
        for is_subj in [True, False]:
            try:
                entity_str = '"' + '"@en "'.join(list(i)) + '"' + '@en'
                c, l, obj = single_query(entity_str, is_subj)   
                for rel_code, code_tuples in c.items():
                    code_triples[rel_code] = code_triples.get(rel_code, []) + code_tuples
                # code_triples.update(c)
                for rel_label, label_tuples in l.items():
                    label_triples[rel_label] = label_triples.get(rel_label, []) + label_tuples
                # label_triples.update(l)
                found_objects.update(obj)
            except TypeError as error:
                print(error)
                continue
    return code_triples, label_triples, found_objects

def get_secondary_wikidata_facts(code_facts, top_k):
    """given the initially extracted facts,
    extract more facts with the most frequent relations
    from specific domain
    """
    code_triples = dict()
    label_triples = dict()
    found_objects = dict()
    min_count = 1
    keys = sorted(code_facts.items(), key=lambda item: len(item[1]), reverse=True)[:top_k]
    facts = {k[0]: code_facts[k[0]] for k in keys if len(code_facts[k[0]])>min_count}
    # check for one 'domain': Q21198 Computer Science
    category = 'Q21198'
    for rel in facts.keys():
        c, l, obj = rel_from_domain(rel, category)
        for rel_code, code_tuples in c.items():
            code_triples[rel_code] = code_triples.get(rel_code, []) + code_tuples
        # code_triples.update(c)
        for rel_label, label_tuples in l.items():
            label_triples[rel_label] = label_triples.get(rel_label, []) + label_tuples
        # label_triples.update(l)
        found_objects.update(obj)
    label_triples = dict()
    found_objects = dict()
    with open('test_secondary_facts.json', 'w+') as f:
        # this would place the entire output on one line
        # use json.dump(lista_items, f, indent=4) to "pretty-print" with four spaces per indent
        json.dump(code_triples, f, indent=4)
    return code_triples, label_triples, found_objects
    

def prepare_neg_data(neg_samples):
    """negative training samples: from each existing ep+rel pair, create
        #neg_samples samples that are not existing at the real data"""

def run(file_dir, index_path):
    # reading from a .pdf converted to .txt:
    content = read_from_pdf(file_dir)
    entity_index = read_index(index_path)
    
    facts = dict()
    # start extracting facts with the index:
    # ... and expand with the found objects: 
    expand_iterations = 1
    for i  in range(expand_iterations):
        code_f, f, new_entities = get_initial_wikidata_facts(dict(itertools.islice(entity_index.items(), 15))  ) 
        # code_f, f, new_entities = get_initial_wikidata_facts(entity_index)
        facts.update(f)
        sec_code_triples, sec_label_triples, sec_found_objects = get_secondary_wikidata_facts(code_f, 15)
        entity_index.update(new_entities)
    
    with open('test_entitiy_index.json', 'w+') as f:
        # this would place the entire output on one line
        # use json.dump(lista_items, f, indent=4) to "pretty-print" with four spaces per indent
        json.dump(entity_index, f, indent=4)

    with open('kb_facts_test.json', 'w+') as fout:
        #print(*facts, sep="\n", file=fout)
        json.dump(facts, fout, indent=4)

    # reset facts to [] just to see which KB pairs have textual mentions
    for k, _ in facts.items():
        facts[k] = []
    for sentence in content:

        sentence_iter = iter(enumerate(sentence))

        found_here = []
        found_here_on = []
        last_found = None
        last_found_on = None

        # finding entity mentions in the sentence:
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
        # result = []
        for e, i in zip(found_here, found_here_on):
            if i[0]!=i[1]:
                seq = ' '.join(sentence[i[0]:i[1]])
                # result.append((e[0], e[1], '*****'.join([e[0], e[1]]), '$ARG1 '+ seq +' $ARG2'))
                # adding mentions for pairs found in KB + all other pairs of entites from the index or found in KB relations:
                facts.setdefault(e[0] +'*****'+e[1],[]).append('$ARG1 '+ seq +' $ARG2')
        # result = [ (e[0] + '\t' + e[1], e[0], e[1],  '$ARG1 '+' '.join(sentence[i[0]:i[1]])+' $ARG2') for e, i in zip(found_here, found_here_on) if i[0]!=i[1]]

    with open('text_facts_test.json', 'w+') as fout:
        #print(*facts, sep="\n", file=fout)
        json.dump(facts, fout, indent=4)

    # return list(text_examples)
   

if __name__ == "__main__":
    run("data/test_chapters", "data/[28]index.txt")
    # with open('train.tsv','w') as out:
    #     csv_out=csv.writer(out, delimiter='\t')
    #     csv_out.writerow(['e1','e2', 'ep', 'relation_id', 'sequence', '1'])
    #     for row in data:
    #         row = row + (1,)
    #         csv_out.writerow(row)

    # query = '''    '''
    # print(wikidata_query(query))
