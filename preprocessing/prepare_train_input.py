import nltk
import numpy as np

def run(file_path):
    with open(file_path) as file:
        content = [nltk.word_tokenize(x) for x in file.readlines()] # TODO: too slow, replace with split and strip
    entity_index = ('data', 'results')#, 'introduction', 'evaluation')
    for sentence in content:
        found_here = []
        found_here_on = []
        last_found = None
        last_found_on = None
        common_entites = set(sentence).intersection(entity_index)
        if len(common_entites) > 1:
            for i,w in enumerate(sentence):
                if w in common_entites:
                    if last_found != None and last_found != w:
                        found_here.append((last_found, w))
                        found_here_on.append((last_found_on, i))
                    last_found_on = i
                    last_found = w
        # print(
        #     [ (e[0], e[1], sentence[i[0]+1:i[1]]) for e, i in zip(found_here, found_here_on)]
        #     )
        result = [ (e[0], e[1], '$ARG1 '+' '.join(sentence[i[0]+1:i[1]])+' $ARG2') for e, i in zip(found_here, found_here_on)]

if __name__ == "__main__":
    run("data/texts.txt")

