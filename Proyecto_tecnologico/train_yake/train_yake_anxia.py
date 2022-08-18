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



### ANOREXIA'S EXPERIMENTS ####
anxia_train = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Anorexia_2018/Anorexia_Datasets_1/train'
anxia_test = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Anorexia_2018/Anorexia_Datasets_1/test'

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



print("Inicia entrenamiento")



all_pos_clean = normalize(all_pos)
all_neg_clean = normalize(all_neg)
#t = get_lines(all_pos_clean)
#s = get_lines(all_neg_clean)

def get_words_from_kw(kw):
    list1 = []
    for i in range(len(kw)):
        list1.append(kw[i][0])
    return list1

kw_extractor = yake.KeywordExtractor()
language = "en"
max_ngram_size = 1
deduplication_threshold = 0.9
numOfKeywords = 2000
for i in range(len(all_pos_clean)): 
	text = all_pos_clean[i]
	custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
	keywords1 = custom_kw_extractor.extract_keywords(text)
	name_key = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/train_yake/pos_anxia/key_pos_anxia' + str(i)
	with open(name_key, 'wb') as f:
		pickle.dump(keywords1, f)
		f.close()

for i in range(len(all_neg_clean)): 
	text = all_neg_clean[i]
	custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
	keywords1 = custom_kw_extractor.extract_keywords(text)
	name_key = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/train_yake/neg_anxia/key_neg_anxia' + str(i)
	with open(name_key, 'wb') as f:
		pickle.dump(keywords1, f)
		f.close()
  
	
#kw_extractor = yake.KeywordExtractor()
#text = t
#language = "en"
#max_ngram_size = 1
#deduplication_threshold = 0.9
#numOfKeywords = 8000
#custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
#keywords1 = custom_kw_extractor.extract_keywords(text)

#with open('key1', 'wb') as f: 
    #Pickling
#    pickle.dump(keywords1, f)
#    f.close()
    
#kw_extractor = yake.KeywordExtractor()
#text = s
#language = "en"
#max_ngram_size = 1
#deduplication_threshold = 0.9
#numOfKeywords = 8000
#custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
#keywords2 = custom_kw_extractor.extract_keywords(text)


    
#with open('key2', 'wb') as f: 
    #Pickling
#    pickle.dump(keywords2, f)
#    f.close()

#kw_extractor = yake.KeywordExtractor()
#text = t
#language = "en"
#max_ngram_size = 2
#deduplication_threshold = 0.9
#numOfKeywords = 8000
#custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
#keywords3 = custom_kw_extractor.extract_keywords(text)

#with open('key3', 'wb') as f: 
    #Pickling
#    pickle.dump(keywords3, f)
#    f.close()

#kw_extractor = yake.KeywordExtractor()
#text = s
#language = "en"
#max_ngram_size = 2
#deduplication_threshold = 0.9
#numOfKeywords = 8000
#custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
#keywords4 = custom_kw_extractor.extract_keywords(text)

#with open('key4', 'wb') as f: 
    #Pickling
#    pickle.dump(keywords4, f)
#    f.close()




print("Termina entrenamiento")
