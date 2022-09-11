from User_dictionary.text_functions import (get_text_chunk,
                            get_text_test_anorexia, get_text_labels,
                            get_text_test)
#from functions_for_vec import  normalize
import yake
from os import listdir
from os.path import isfile, join
import pickle


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

for i in range(len(all_pos)): 
	text = get_text(all_pos[i])
	custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
	keywords1 = custom_kw_extractor.extract_keywords(text)
	name_key = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/train_yake/pos_anxia/key_pos_anxia' + str(i)
	with open(name_key, 'wb') as f:
		pickle.dump(keywords1, f)
		f.close()

for i in range(len(all_neg)): 
    text = get_text(all_neg[i])
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
    keywords1 = custom_kw_extractor.extract_keywords(text)
    name_key = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/train_yake/neg_anxia/key_neg_anxia' + str(i)
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
