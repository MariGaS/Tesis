from text_functions import *
import yake
import pickle

#EXTRACT TRAIN DATA FOR THE DICTIONARIES 
print('Begin extract of train text')
anxia_train = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Anorexia_2018/Anorexia_Datasets_1/train'
pos = 'positive_examples'
neg = 'negative_examples'


pos = 'positive_examples'
neg = 'negative_examples'

all_pos = []
all_neg = []
for i in range(1,11):
    path_chunk_pos = anxia_train +'/' + pos + '/chunk'+str(i)
    path_chunk_neg = anxia_train +'/' + neg + '/chunk'+str(i)

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

flat_list_pos = [item for sublist in all_pos for item in sublist]
flat_list_neg = [item for sublist in all_neg for item in sublist]

#YAKE PARAMETERS 
#STATE YAKE PARAMTERS
language = "en"
max_ngram_size = 2
deduplication_thresold = 0.9
deduplication_algo = 'seqm'
windowSize = 1
numOfKeywords = 12000

#CONTRUCTING POSITIVE DICTIONARY
#number of posts from positive user 
print("Begin key extraction")
count_positive = count_negative(flat_list_pos)
text = define_order_post(count_positive, flat_list_pos, 'positive')
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
keywords1 = custom_kw_extractor.extract_keywords(text)
print("Ends key extraction")

name_key = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/Bigrams/anxia_pos_ver1'
with open(name_key, 'wb') as f:
	pickle.dump(keywords1, f)
	f.close()

# Negative dictionary 
print("Begin key extraction")
count_negative1 = count_negative(flat_list_neg)
text = define_order_post(count_negative1, flat_list_neg, 'negative')
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
keywords1 = custom_kw_extractor.extract_keywords(text)
print("Ends key extraction")

name_key = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/Bigrams/anxia_neg_ver1'
with open(name_key, 'wb') as f:
	pickle.dump(keywords1, f)
	f.close()

def norm(document):

    document = [x.lower()  for x in document]

    return document

#CONTRUCTING POSITIVE DICTIONARY
#number of posts from positive user 
print("Begin key extraction")
count_positive = count_negative(flat_list_pos)
t = norm(flat_list_pos)
text = define_order_post(count_positive, t, 'positive')
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
keywords1 = custom_kw_extractor.extract_keywords(text)
print("Ends key extraction")

name_key = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/Bigrams/anxia_pos_ver2'
with open(name_key, 'wb') as f:
	pickle.dump(keywords1, f)
	f.close()

# Negative dictionary 
print("Begin key extraction")
count_negative2 = count_negative(flat_list_neg)
t = norm(flat_list_neg)
text = define_order_post(count_negative2, t, 'negative')
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
keywords1 = custom_kw_extractor.extract_keywords(text)
print("Ends key extraction")

name_key = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/Bigrams/anxia_neg_ver2'
with open(name_key, 'wb') as f:
	pickle.dump(keywords1, f)
	f.close()




