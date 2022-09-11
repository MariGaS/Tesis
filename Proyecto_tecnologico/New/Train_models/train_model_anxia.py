from text_functions import (get_text_chunk,
                            get_text_test_anorexia)
from gensim.models import FastText
from gensim.models import phrases, word2vec
from nltk import TweetTokenizer
import re
from os import listdir
from os.path import isfile, join
from gensim.models import utils
from gensim.models import FastText
import gensim.downloader
from gensim.test.utils import get_tmpfile, datapath
import fasttext
import fasttext.util
import pickle



tokenizer = TweetTokenizer()
def normalize(document):
    #eliminate link 
    document = [re.sub(r'{link}', '', x) for x in document]
    # eliminate video
    document = [re.sub(r"\[video\]", '', x) for x in document]
    # eliminate url
    document = [re.sub(r'https?:\/\/\S+', '', x) for x in document]
    # eliminate url
    document = [re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', x) for x in document]
    # eliminate #
    document = [x.replace("#","") for x in document]

    return document




### ANOREXIA'S EXPERIMENTS ####
anxia_train = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Anorexia_2018/Anorexia_Datasets_1/train'
anxia_test = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Anorexia_2018/Anorexia_Datasets_1/test'

pos = 'positive_examples'
neg = 'negative_examples'

all_pos = []
all_neg = []
for i in range(1, 11):
    path_chunk_pos = anxia_train + '/' + pos + '/chunk' + str(i)
    path_chunk_neg = anxia_train + '/' + neg + '/chunk' + str(i)

    temp1 = get_text_chunk(path_chunk_pos)
    temp2 = get_text_chunk(path_chunk_neg)
    if i == 1:
        all_pos = temp1
        all_neg = temp2
    else:
        for j in range(len(temp1)):
            all_pos[j] += temp1[j]

        for j in range(len(temp2)):
            all_neg[j] += temp2[j]

tr_anorexia = [*all_pos, *all_neg]
tr_label = []


test_url_anxia = []
test_labels_anxia = []

with open('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Anorexia_2018/Anorexia_Datasets_1/test/test_golden_truth.txt') as f:
    lines = f.readlines()
    for line in lines:
        test_url_anxia.append(line[:-3])  # only the name of the subject

        test_labels_anxia.append(int(line[-2:-1]))  # only the label
    f.close()


test_anxia = []
for i in range(1, 11):

    temp1 = get_text_test_anorexia(anxia_test, test_url_anxia, i)

    if i == 1:
        test_anxia = temp1
        # print("Text extracted from chunk: ", i)
    else:

        for j in range(len(temp1)):
            test_anxia[j] += temp1[j]
        # print("Text extracted from chunk: ", i)
        
train_clean = normalize(tr_anorexia)
test_clean = normalize(test_anxia)



sentences = [*train_clean, *test_clean]
all_text = []
for i in range(len(sentences)):
    all_text.append(sentences[i].split(" "))
print("Inicia entrenamiento")

model = FastText(vector_size=300, window=15, min_count=1)
model.build_vocab(all_text)
total_examples = model.corpus_count
model.train(all_text, total_examples=total_examples, epochs=5)

model.save('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Models/anxia.model')

print("Termina entrenamiento")
