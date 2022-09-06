from text_functions import (get_text_chunk,
                            get_text_test_anorexia, get_text_labels,
                            get_text_test)
#from functions_for_vec import  normalize
import yake
import os
import re
from os import listdir
from os.path import isfile, join
import pickle




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
tr_neg = [*tr_neg_2017, *tr_neg_2018]
# ALL TRAINING DATA 2018
tr_pos = [*tr_pos_2017, *tr_pos_2018]




print("Inicia entrenamiento")


def get_text(user):
    t = ''
    for i in range(len(user)):
        t = t + user[i] + '\n'
    return t 


language = "en"
max_ngram_size = 1
deduplication_thresold = 0.9
deduplication_algo = 'seqm'
windowSize = 1
numOfKeywords = 1500

for i in range(len(tr_pos)): 
	text = get_text(tr_pos[i])
	custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
	keywords1 = custom_kw_extractor.extract_keywords(text)
	name_key = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/train_yake/pos_dep/key_pos_dep' + str(i)
	with open(name_key, 'wb') as f:
		pickle.dump(keywords1, f)
		f.close()

for i in range(len(tr_neg)): 
	text = get_text(tr_neg[i])
	custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
	keywords1 = custom_kw_extractor.extract_keywords(text)
	name_key = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/train_yake/neg_dep/key_neg_dep' + str(i)
	with open(name_key, 'wb') as f:
		pickle.dump(keywords1, f)
		f.close()
  

print("Termina entrenamiento")
