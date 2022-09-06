from text_functions import (get_text_chunk,
                            get_text_test_anorexia, get_text_labels,
                            get_text_test)
#from functions_for_vec import  normalize
import yake
from gensim.models import FastText
from gensim.models import phrases, word2vec
from nltk import TweetTokenizer
import os
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


train_neg_2017 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/depression2022/training_data/2017_cases/neg'
train_pos_2017 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/depression2022/training_data/2017_cases/pos'
train_neg_2018 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/depression2022/training_data/2018_cases/neg'
train_pos_2018 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/depression2022/training_data/2018_cases/pos'

#- MAKE LIST FROM 2017 AND 2018 -#
tr_neg_2017, tr_lab_2017 = get_text_labels(train_neg_2017, polarity='Negative')
tr_pos_2017, tr_lab_pos_2017 = get_text_labels(train_pos_2017, polarity='Pos')
tr_neg_2018, tr_lab_2018 = get_text_labels(train_neg_2018, polarity='Negative')
tr_pos_2018, tr_lab_pos_2018 = get_text_labels(train_pos_2018, polarity='Pos')

# ALL TRAINING DATA 2017
tr_txt_2017 = [*tr_neg_2017, *tr_pos_2017]
tr_y_2017 = [*tr_lab_2017, *tr_lab_pos_2017]
# ALL TRAINING DATA 2018
tr_txt_2018 = [*tr_neg_2018, *tr_pos_2018]
tr_y_2018 = [*tr_lab_2018, *tr_lab_pos_2018]



print("Inicia entrenamiento")


def get_lines(text):
        #text = normalize(text)
    text_change = []
    string = ''
    for i in range(len(text)):
        string = string + ' ' + text[i]
    #text_change.append(string)
    return string



all_pos_clean = normalize([*tr_pos_2017, *tr_pos_2018])
all_neg_clean = normalize([*tr_neg_2017, *tr_neg_2018])


kw_extractor = yake.KeywordExtractor()
language = "en"
max_ngram_size = 1
deduplication_threshold = 0.9
numOfKeywords = 800
for i in range(len(all_pos_clean)): 
	text = all_pos_clean[i]
	custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
	keywords1 = custom_kw_extractor.extract_keywords(text)
	name_key = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/train_yake/pos_dep/key_pos_dep' + str(i)
	with open(name_key, 'wb') as f:
		pickle.dump(keywords1, f)
		f.close()

for i in range(len(all_neg_clean)): 
	text = all_neg_clean[i]
	custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
	keywords1 = custom_kw_extractor.extract_keywords(text)
	name_key = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/train_yake/neg_dep/key_neg_dep' + str(i)
	with open(name_key, 'wb') as f:
		pickle.dump(keywords1, f)
		f.close()
  

print("Termina entrenamiento")
