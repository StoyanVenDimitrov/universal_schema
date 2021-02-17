import random
import nltk
import csv
import numpy as np
import requests
import json
import os,glob
import itertools  
from itertools import groupby, zip_longest
from collections import defaultdict
from bs4 import BeautifulSoup
from get_requests import single_query, rel_from_domain, wikimedia_request

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
    # TODO: Map entities with 'see ...' together
    return entity_index


def get_initial_wikidata_facts(entities):
    """get all KB relations where the entity is 
       subject or object
    Args:
        entities ([list]): index entity list
    Returns:
        dict: {KB_rel:[(subj, obj),..],...}
    """
    # split the entites in groups of n to send them with one request
    try:
        with open('data/initial_code_facts.json') as f_in:
            code_triples = json.load(f_in)
        with open('data/initial_label_facts.json') as f_in:
            label_triples = json.load(f_in)
        with open('data/initial_entities.json') as f_in:
            found_objects = json.load(f_in)
        return code_triples, label_triples, found_objects
    except FileNotFoundError:
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
        with open('data/initial_code_facts.json', 'w+') as f:
            json.dump(code_triples,  f, indent=4)
        with open('data/initial_label_facts.json', 'w+') as f:
            json.dump(label_triples, f, indent=4)
        with open('data/initial_entities.json', 'w+') as f:
            json.dump(found_objects, f, indent=4)
        return code_triples, label_triples, found_objects


def get_secondary_wikidata_facts(code_facts, top_k):
    """given the initially extracted facts,
    extract more facts with the most frequent relations
    from specific domain
    Args:
        code_facts ([dict]): key: KB relations, values: (subj,obj)
        top_k ([int]): k most frequent KB relations 
    Returns:
        dict: {KB_rel:[(subj, obj),..],...}
    """
    try:
        with open('data/secondary_code_facts.json') as f_in:
            code_triples = json.load(f_in)
        with open('data/secondary_label_facts.json') as f_in:
            label_triples = json.load(f_in)
        with open('data/secondary_entities.json') as f_in:
            found_objects = json.load(f_in)
        return code_triples, label_triples, found_objects
    except FileNotFoundError:
        code_triples = dict()
        label_triples = dict()
        found_objects = dict()
        min_count = 1
        keys = sorted(code_facts.items(), key=lambda item: len(item[1]), reverse=True)[:top_k]
        facts = {k[0]: code_facts[k[0]] for k in keys if len(code_facts[k[0]])>min_count}
        # check for one 'domain': Q21198 Computer Science
        # category = ['Q21198', 'Q395']
        # TODO: more categories
        for category in ['Q21198', 'Q395']:
            for rel in facts.keys():
                c, l, obj = rel_from_domain(rel, category)
                for rel_code, code_tuples in c.items():
                    code_triples[rel_code] = code_triples.get(rel_code, []) + code_tuples
                # code_triples.update(c)
                for rel_label, label_tuples in l.items():
                    label_triples[rel_label] = label_triples.get(rel_label, []) + label_tuples
                # label_triples.update(l)
                found_objects.update(obj)
            # label_triples = dict()
            # found_objects = dict()
        with open('data/secondary_code_facts.json', 'w+') as f:
            json.dump(code_triples,  f, indent=4)
        with open('data/secondary_label_facts.json', 'w+') as f:
            json.dump(label_triples, f, indent=4)
        with open('data/secondary_entities.json', 'w+') as f:
            json.dump(found_objects, f, indent=4)
        return code_triples, label_triples, found_objects


def get_textual_mentions(term_pair):

    """search for textual mentions in wiki snippets
    Args:
        term_pair (tuple): pair of KB terms
    Returns:
        list of strings, found on wikipedia pages between 
        the two KB terms
    """
    mentions = set()
    cont = None
    while True:
        res = wikimedia_request(term_pair[0], cont)
        try:
            for i in res['query']['search']:
                soup = BeautifulSoup(i['snippet'])
                plain_text_snippet = soup.get_text().lower()
                # !!! important later for search on whole page:
                # no need to search if the term is not in the snippet
                if term_pair[0] in plain_text_snippet: 
                    #TODO: search on whole pages
                    pos_0 = plain_text_snippet.find(term_pair[0]) + len(term_pair[0])
                    pos_1 = plain_text_snippet.find(term_pair[1])
                    if pos_1 > pos_0:
                        mentions.add(plain_text_snippet[pos_0:pos_1])
    # TODO: extract mentions of the second term
        except KeyError:
            continue



        # for experimentation:
        break # only srlimit pages max





        try:
            cont = res['continue']
        except KeyError:
            break
    return list(mentions)


def get_wikipedia_evidences(sec_labels):
    """collect textual mentions from Wikipedia
    Args:
        sec_labels: secondary extracted facts, with labels
    Returns:
        dict with text mentions per entity pair 
    """

    try:
        with open('data/wikipedia_evidences.json') as f_in:
            wikipedia_evidences = json.load(f_in)
    except FileNotFoundError:
        wikipedia_evidences = dict()
        # TODO: by now, the initial KB facts are NOT used to extract text patterns:
        id_start = 0
        for relation, pairs in sec_labels.items():
            entity_text_mentions = dict()
            entity_id_mentions = dict()            
            for _id, i in enumerate(pairs, id_start):
                mentions = get_textual_mentions(i)
                if mentions:
                    mentions.append(relation)
                    entity_text_mentions['*'.join(i)] = mentions
                    entity_id_mentions[_id] = mentions
                # entity_text_mentions['*'.join(i)] = get_textual_mentions(i)
            wikipedia_evidences[relation] = entity_text_mentions
            id_start = _id
        with open('data/wikipedia_evidences.json', 'w+') as fout:
            #print(*facts, sep="\n", file=fout)
            json.dump(wikipedia_evidences, fout, indent=4)
    return wikipedia_evidences

def prepare_neg_data(all_pos_samples, num_samples):
    """negative training samples: from each existing ep+rel pair, create
        #neg_samples samples that are not existing at the real data
    Args:
        num_samples (int): number of negative samples
        all_pos_samples (dict): the extraction so far (e.g. wikidata_evidences.json)
    """
    try:
        with open('data/negative_evidences.json') as f_in:
            neg_samples = json.load(f_in)
            # TODO: check if the content satisfies num_samples and
            # all_pos_samples have been changed
    except FileNotFoundError:
        neg_samples = dict()
        for relation, pos_samples in all_pos_samples.items():
            neg_pool = set()
            for rel, examples in all_pos_samples.items():
                # take negatives only when NOT expressing the same relation:
                if rel != relation:
                    for e in examples.values():
                        neg_pool.update(e)
                  
            for _id, values in pos_samples.items():
                neg_examples = set()
                while len(neg_examples) < num_samples:
                    neg_rel = random.choice(list(neg_pool))
                    if not neg_rel in values:
                        neg_examples.add(neg_rel)
                    # sample entity pairs:
                    # sample_ids = random.choices(list(pos_samples.keys()), k=num_samples)
                    # for sample_pair in sample_ids:
                    #     if sample_pair != _id:
                    #     # sample relation from the entity pair's list:
                    #         neg_rel = random.choice(list(pos_samples.keys()))
                    #         if not neg_rel in values:
                    #             neg_examples.append(neg_rel) 
                neg_samples[_id] = list(neg_examples)
        with open('data/negative_evidences.json', 'w+') as f:
            json.dump(neg_samples,  f, indent=4)
    return neg_samples


def get_training_data(index):
    """gather training data, starting from the book's index,
       finding Wikidata KB relations and textual mentions on wikipedia
    Args:
        index ([type]): book's index
    """
    facts = dict()
    # codes, labels, new_entities = get_initial_wikidata_facts(dict(itertools.islice(entity_index.items(), 15)))
    codes, labels, new_entities = get_initial_wikidata_facts(index) # s or o from the book index
    facts.update(labels)
    index.update(new_entities)  # take in the new entities 
    sec_codes, sec_labels, sec_new_entities = get_secondary_wikidata_facts(codes, 4)
    # search textual patterns between entities of found KB facts for the most prominent relations 
    wikipedia_evidences = get_wikipedia_evidences(sec_labels)
    neg_data = prepare_neg_data(wikipedia_evidences, 40)

    # [{row:..., seen_with:[...], column:..., label: 0 or 1}, ...]
    with open('data/final_dataset.json', 'w+') as outfile:
        # final_dataset = []
        for _, evidences in wikipedia_evidences.items():
            for pair, relations in evidences.items():
                for mention in relations:
                    to_add = {
                            'entity_pair': pair, 
                            'seen_with': relations, 
                            'relation': mention,
                            'label': 1
                        }
                json.dump(to_add, outfile)
                outfile.write('\n')
                neg_relations = neg_data[pair]
                for neg_mention in neg_relations:
                    to_add = {
                        'entity_pair': pair, 
                        'seen_with': relations, 
                        'relation': neg_mention,
                        'label': 0
                    }
                    json.dump(to_add, outfile)
                    outfile.write('\n')
    # final_json = {"training_set": final_dataset}
    # with open('data/final_dataset.json', 'w+') as f:
    #         json.dump(final_json,  f, indent=4)
    return index 


def run(file_dir, index_path):
    # ------ get the book' index -------
    index = read_index(index_path)
    
    # ---- preparing training data ------
    entity_index = get_training_data(index)

    # ---- prepare test data - (s, text, o) with s,o from entity_index of all 'interesting' entities
    """manual_content = read_from_pdf(file_dir)

    for sentence in manual_content:

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
                # facts.setdefault(e[0] +'*****'+e[1],[]).append('$ARG1 '+ seq +' $ARG2')
        # result = [ (e[0] + '\t' + e[1], e[0], e[1],  '$ARG1 '+' '.join(sentence[i[0]:i[1]])+' $ARG2') for e, i in zip(found_here, found_here_on) if i[0]!=i[1]]

    # with open('test_entitiy_index.json', 'w+') as f:
    #     # this would place the entire output on one line
    #     # use json.dump(lista_items, f, indent=4) to "pretty-print" with four spaces per indent
    #     json.dump(entity_index, f, indent=4)

    # with open('kb_facts_test.json', 'w+') as fout:
    #     #print(*facts, sep="\n", file=fout)
    #     json.dump(facts, fout, indent=4)

    # with open('text_facts_test.json', 'w+') as fout:
    #     #print(*facts, sep="\n", file=fout)
    #     json.dump(facts, fout, indent=4)
    
    # return list(text_examples)
   """

if __name__ == "__main__":
    run("resources/chapters", "resources/[28]index.txt")
    # with open('train.tsv','w') as out:
    #     csv_out=csv.writer(out, delimiter='\t')
    #     csv_out.writerow(['e1','e2', 'ep', 'relation_id', 'sequence', '1'])
    #     for row in data:
    #         row = row + (1,)
    #         csv_out.writerow(row)

    # query = '''    '''
    # print(wikidata_query(query))
