from text_functions import (get_text_labels,
                            get_text_test)
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
#from gensim.models import fasttext
import fasttext
import fasttext.util
import pickle



tokenizer = TweetTokenizer()
def normalize(document):

    document = [x.lower()  for x in document]
    # eliminate url
    document = [re.sub(r'https?:\/\/\S+', '', x) for x in document]
    # eliminate url
    document = [re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', x) for x in document]
    # eliminate link
    document = [re.sub(r'{link}', '', x) for x in document]
    # eliminate video
    document = [re.sub(r"\[video\]", '', x) for x in document]
    document = [re.sub(r'\s+', ' ' '', x).strip() for x in document]
    # eliminate #
    document = [x.replace("#","") for x in document]
    # eliminate emoticons
    document = [re.subn(r'[^\w\s,]',"", x)[0].strip() for x in document]

    return document



train_neg_2017 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/depression2022/training_data/2017_cases/neg'
train_pos_2017 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/depression2022/training_data/2017_cases/pos'
train_neg_2018 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/depression2022/training_data/2018_cases/neg'
train_pos_2018 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/depression2022/training_data/2018_cases/pos'

#- MAKE LIST FROM 2017 AND 2018 -#
tr_neg_2017= get_text_labels(train_neg_2017)
tr_pos_2017= get_text_labels(train_pos_2017)
tr_neg_2018= get_text_labels(train_neg_2018)
tr_pos_2018= get_text_labels(train_pos_2018)

# ALL TRAINING DATA 2017
tr_txt_2017 = [*tr_neg_2017, *tr_pos_2017]
# ALL TRAINING DATA 2018
tr_txt_2018 = [*tr_neg_2018, *tr_pos_2018]



test_data = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/depression2022/test_data/datos'

test_url = []
test_labels = []

with open('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/depression2022/test_data/risk_golden_truth.txt') as f:
    lines = f.readlines()
    for line in lines:

        test_url.append(line[:-3])  # only the name of the subject

        test_labels.append(int(line[-2:-1]))  # only the label
    f.close()

test_txt = get_text_test(test_data, test_url)
train = tr_txt_2017 + tr_txt_2018 #all training data
test = test_txt  #test data



train_dep  = normalize(train)
test_dep   = normalize(test)


sentences = [*train_dep,*test_dep]
all_text = []
for i in range(len(sentences)):
    all_text.append(sentences[i].split(" "))
print("Inicia entrenamiento")


model = FastText(vector_size=300, window=15, min_count=1)
model.build_vocab(all_text)
total_examples = model.corpus_count
model.train(all_text, total_examples=total_examples, epochs=5)

model.save('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Models/depresion.model')



print("Termina entrenamiento")
