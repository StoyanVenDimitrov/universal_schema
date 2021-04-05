import csv
import glob
import itertools
import json
import os
import random
import re
from collections import defaultdict
from itertools import groupby, zip_longest

import nltk
import numpy as np
import requests
from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer

from get_requests import rel_from_domain, single_query, wikimedia_request

nltk.download('punkt')

STEMMER = PorterStemmer()


def read_from_pdf(file_dir):
    """read from pdf-to-txt converted texts

    Args:
        file_dir (str): all files

    Returns:
        List[List[str]]: Sentences[Tokens]
    """
    content = []
    dirs = glob.glob(os.path.join(file_dir, '*.txt')) if glob.glob(os.path.join(file_dir, '*.txt')) else [file_dir]
    for filename in dirs:
        content_as_list = []
        with open(filename) as file:
            for line in file.readlines():
                line_as_list = nltk.word_tokenize(line) 
                if len(line_as_list) >3:
                    content_as_list = content_as_list + line_as_list
        i = (list(g) for _, g in groupby(content_as_list, key='.'.__ne__))
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
    # TODO: use nltk stemmer
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
            x = {label:len(value) for label,value in label_triples.items()}
            print({k: v for k, v in sorted(x.items(), key=lambda item: item[1], reverse=True)})
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
        # ! Domains: Mathematics, CS.
        skip_following = ['P101', 'P1269', 'P461', 'P2579', 'P373', 'P1659', 'P1855', 'P1629', 'P703']
        for category in ['Q21198', 'Q395','Q816264', 'Q12483', 'Q8078', 'Q11023','Q413']:
            for rel in facts.keys():
                if rel not in skip_following:
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

    # TODO: the while is unneccessery if break after 1 iteration remains:
    for term in term_pair:
        while True:
            res = wikimedia_request(term, cont)
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
                            mention = plain_text_snippet[pos_0:pos_1]
                            # TODO: if not in different sentences
                            if "." not in mention:
                                if mention not in [" ", ", " ]:
                                    mentions.add(mention)
            except KeyError:
                continue
            #! for experimentation (but meaningful anyway)
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
        with open('data/wikipedia_evidences copy.json') as f_in:
            wikipedia_evidences = json.load(f_in)
    except FileNotFoundError:
        wikipedia_evidences = dict()
        id_start = 0
        for relation, pairs in sec_labels.items():
            entity_text_mentions = dict()
            entity_id_mentions = dict()            
            for _id, i in enumerate(pairs, id_start):
                mentions = get_textual_mentions(i)
                if mentions:
                    # mentions.append(relation)
                    entity_text_mentions[' * '.join(i)] = mentions
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
        # ! negative pool - only KB relations:
        neg_pool = set(all_pos_samples.keys())
        for relation, pos_samples in all_pos_samples.items():
            # ! removing mentions as negatives
            # adding all neg mentions
            # neg_pool = set()
            # for rel, examples in all_pos_samples.items():
            #     # take negatives only when NOT expressing the same relation:
            #     if rel != relation:
            #         for e in examples.values():
            #             neg_pool.update(e)
                  
            for _id, values in pos_samples.items():
                neg_examples = set()
                while len(neg_examples) < num_samples:
                    neg_rel = random.choice(list(neg_pool))
                    if neg_rel != relation:
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
        index ([type]): book's index terms + terms related with them in the KB
    """
    top_k = 18
    facts = dict()
    # codes, labels, new_entities = get_initial_wikidata_facts(dict(itertools.islice(entity_index.items(), 15)))
    codes, labels, new_entities = get_initial_wikidata_facts(index) # s or o from the book index
    facts.update(labels)
    index.update(new_entities)  # take in the new entities 
    sec_codes, sec_labels, sec_new_entities = get_secondary_wikidata_facts(codes, top_k)
    # define the possible KB relations for test time
    # best_rels_obj = sorted(codes.items(), key=lambda item: len(item[1]), reverse=True)[:top_k]
    desired_rels = list(sec_codes.keys())
    # search textual patterns between entities of found KB facts for the most prominent relations 
    wikipedia_evidences = get_wikipedia_evidences(sec_labels)
    neg_data = prepare_neg_data(wikipedia_evidences, 5)
    data_statistics = {kb_rel:len(evidences.keys()) for kb_rel, evidences in wikipedia_evidences.items()}
    # [{row:..., seen_with:[...], column:..., label: 0 or 1}, ...]
    with open('data/final_dataset.json', 'w+') as outfile:
        # final_dataset = []
        for kb_rel, evidences in wikipedia_evidences.items():
            # ! reduce P279 and P31:
            if kb_rel in ['P279', 'P31']:
                subset = random.sample(evidences.items(),100)
                evidences = dict(subset)
            for pair, relations in evidences.items(): 
                # ! remove the relation itself to avoid predicting simply its position
                # seen_with = [i for i in relations if not i.startswith('P')]
                for mention in relations:
                    to_add = {
                            'entity_pair': pair, 
                            'seen_with': relations, 
                            'relation': kb_rel,
                            'label': 1
                        }
                    # ! query with textual mentions as in USchema is too hard and 
                    # ! not sufficient (prob. for depend. paths works).
                    # ! Go only for KB relations as query, also on. neg samples
                json.dump(to_add, outfile) # last is always path
                outfile.write('\n')
                neg_relations = neg_data[pair]
                for neg_rel in neg_relations:
                    to_add = {
                        'entity_pair': pair, 
                        'seen_with': relations, 
                        'relation': neg_rel,
                        'label': 0
                    }
                    json.dump(to_add, outfile)
                    outfile.write('\n')

    return index, desired_rels


def read_annotations(file_dir, desired_rels):
    """read the annotated text file

    Args:
        file_dir (str): text file
    """
    content = read_from_pdf(file_dir)
    all_relations = dict()
    for tokens in content: 
        entities_start = []
        entities_end = []
        for i,token in enumerate(tokens):
            if token=='_start_e1_':
                entities_start = []
                entities_end = []

            if token.startswith('_start_'):
                entities_start.append(i+1)
            if token.startswith('_end_'):
                entities_end.append(i)
            if len(entities_start)==len(entities_end):
                contexts = [tokens[end+1:start-1] for start, end in zip(entities_start[1:], entities_end[:-1])]
                if contexts:
                    entity_1 = [tokens[i:j] for i,j in zip(entities_start[:-1], entities_end[:-1])]
                    entity_2 = [tokens[i:j] for i,j in zip(entities_start[1:], entities_end[1:])]
                    for e1,c,e2 in zip(entity_1, contexts, entity_2):
                        key = ' '.join([STEMMER.stem(i) for i in e1]) + ' * ' + ' '.join([STEMMER.stem(j) for j in e2])
                        all_relations.setdefault(key, set()).add(' '.join(c))
    with open('data/_annotations.json', 'w+') as outfile: 
        for key, values in all_relations.items():
        # add the KB relation to be evaluated:
        # TODO: try directly with 'relation': desired_rels  
            for relation in desired_rels:
                to_add = {
                            'entity_pair': key, 
                            'seen_with': list(values), 
                            'relation': relation
                        }
                json.dump(to_add, outfile)
                outfile.write('\n')
    return all_relations


def get_test_data(index, file_dir, desired_rels, evaluating=False):

    if evaluating:
        # modify the index with test set entities only:
        mod_index = {}
        with open('resources/test_facts.json') as f_in:
            test_facts = json.load(f_in)['samples']
            # in evaluation, take only the test facts' entities: 
            test_index = set()
            for fact in test_facts:
                split = fact['entity_pair'].split(' * ')
                [test_index.add(i) for i in split]
            test_index = list(test_index)
            for k, v in index.items():
                mod_v = ' '.join([STEMMER.stem(i) for i in v])
                if mod_v in test_index:
                    mod_index[k] = v
                    test_index.remove(mod_v)
        index = mod_index

    manual_content = read_from_pdf(file_dir)
    result = dict()
    for sentence in manual_content:

        sentence_iter = iter(enumerate(sentence))

        found_here = []
        found_here_on = []
        last_found = None
        last_found_on = None

        # finding entity mentions in the sentence:
        for i,token in sentence_iter:
            token = token.lower()
            # TODO: only pairs of book_index entity and other entity
            candidates = [item for item in index.items() if item[1][0]==token]
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
            if last_found != None and last_found[0] != single_candidate[0]:
                # apply aliases from list of aliases
                found_here.append(
                    (
                        ' '.join([STEMMER.stem(i) for i in last_found[1]]), 
                        ' '.join([STEMMER.stem(i) for i in single_candidate[1]])
                    )
                )  
                found_here_on.append((last_found_on, span[0]))
            last_found_on = span[1]
            last_found = single_candidate

        for e, i in zip(found_here, found_here_on):
            if i[0]!=i[1]:
                pair = ' * '.join([e[0], e[1]])
                result.setdefault(pair, []).append(' '.join(sentence[i[0]:i[1]]))
    # first, drop all found mentions and prepare a dataset:
    with open('data/prediction_dataset.json', 'w+') as outfile: 
        for key, values in result.items():
            # add the KB relation to be evaluated:
            # TODO: try directly with 'relation': desired_rels  
            for relation in desired_rels:
                to_add = {
                            'entity_pair': key, 
                            'seen_with': values, 
                            'relation': relation
                        }
                json.dump(to_add, outfile)
                outfile.write('\n')
    # second, if evaluating, create dataset with the labaled data:
    if evaluating:
        # mentions from the test anntoations only
        with open('data/test_dataset.json', 'w+') as outfile: 
            for item in test_facts:
                for relation in desired_rels: 
                        to_add = {
                                    'entity_pair': item['entity_pair'], 
                                    'seen_with': item['seen_with'], 
                                    'relation': relation,
                                    'label': 1 if relation==item['label'] else 0
                                }
                        json.dump(to_add, outfile)
                        outfile.write('\n')
        # mentions from all the book
        with open('data/test_aggregated_dataset.json', 'w+') as outfile: 
            test_pairs = {i['entity_pair']:i['label'] for i in test_facts}
            for key, values in result.items():
                if key in test_pairs.keys():
                    # add the KB relation to be evaluated:
                    # TODO: try directly with 'relation': desired_rels  
                    for relation in desired_rels: 
                        to_add = {
                                    'entity_pair': key, 
                                    'seen_with': values, 
                                    'relation': relation,
                                    'label': 1 if relation==test_pairs[key] else 0
                                }
                        json.dump(to_add, outfile)
                        outfile.write('\n')
              

def run(file_dir, index_path):
    labeled_data = []
    # with open('data/labeled_annotations.json') as f_in:
    #         annotations = json.load(f_in)
    # with open('data/edited_labeling.json', 'w+') as outfile: 
    #     for a in annotations['samples']:
    #         if a['relation']=='P31':
    #             del a['relation']
    #             labeled_data.append(a)
    #             json.dump(a, outfile)
    #             outfile.write('\n')
    # ------ get the book' index -------
    index = read_index(index_path)
    # with open('resources/test_facts.json') as f_in:
    #         test_facts = json.load(f_in)['samples']
    #         # in evaluation, take only the test facts' entities: 
    #         test_index = set()
    #         for fact in test_facts:
    #             split = fact['entity_pair'].split(' * ')
    #             [test_index.add(i) for i in split]
    # with open('resources/test_index.json', 'w+') as outfile:
    #     json.dump({'index':list(test_index)}, outfile)
    
    # ---- preparing training data ------
    extended_index, desired_rels = get_training_data(index)

    # ---- preparing training data ------
    get_test_data(extended_index, file_dir, desired_rels,evaluating=True)
    # read_annotations('resources/annotated_[11]part-2-chapter-6.txt', desired_rels)

    # ---- prepare test data - (s, text, o) with s,o from entity_index of all 'interesting' entities
    # manual_content = read_from_pdf(file_dir)
   

if __name__ == "__main__":
    run("resources/chapters/[11]part-2-chapter-6.txt", "resources/[28]index.txt")
