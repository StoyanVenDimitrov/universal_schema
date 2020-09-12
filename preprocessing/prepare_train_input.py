import nltk
import numpy as np
from itertools import groupby

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
    entity_index = []
    with open(index_path) as file:
        for line in file.readlines():
            line_list = line.strip().split(',')
            if len(line_list) > 1 and not line_list[0].isdigit():
                entity_index.append(line_list[0].lower())
    # TODO: Map entites with 'see ...' together
    return entity_index

def run(file_path, index_path):
    # reading from a .pdf converted to .txt:
    content = read_from_pdf(file_path)
    entity_index = read_index(index_path)
    # take only unique 
    text_examples = set()

    for sentence in content:

        entity_index = {'DL':['deep', 'learning'], 'DEEP':['deep'], 'FORM':['formalisms'], 'GUIDE':['guide']}
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
            if len(candidates)==1:
                w = candidates[0][0]
                span = (i, i+1)
            else:
                str_the_rest = ' '.join(sentence[i:]).lower()
                to_remove = []
                for c in candidates:
                    str_c = ' '.join(c[1])
                    seen_here = str_the_rest.find(str_c, 0, len(str_c))
                    to_remove.append(seen_here)
                candidates = [candidate for seen, candidate in zip(to_remove, candidates) if seen>-1]
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
                found_here.append((last_found, w))  
                found_here_on.append((last_found_on, span[0]))
            last_found_on = span[1]
            last_found = w

        result = [ (e[0], e[1], '$ARG1 '+' '.join(sentence[i[0]:i[1]])+' $ARG2') for e, i in zip(found_here, found_here_on) if i[0]!=i[1]]
        text_examples.update(result)

    return text_examples

if __name__ == "__main__":
    # run("data/[11]part-2-chapter-6.txt", "data/[28]index.txt")
    run("data/texts.txt", "data/[28]index.txt")

