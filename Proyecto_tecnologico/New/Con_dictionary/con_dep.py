from text_functions import *
import yake
import pickle



train_neg_2017 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/depression2022/training_data/2017_cases/neg'
train_pos_2017 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/depression2022/training_data/2017_cases/pos'
train_neg_2018 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/depression2022/training_data/2018_cases/neg'
train_pos_2018 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/depression2022/training_data/2018_cases/pos'

pos_2017 = get_text_depression(train_pos_2017)
pos_2018 = get_text_depression(train_pos_2018)
neg_2017 = get_text_depression(train_neg_2017)
neg_2018 = get_text_depression(train_neg_2018)

positive_dep = [*pos_2017, *pos_2018]
negative_dep = [*neg_2017, *neg_2018]

#STATE YAKE PARAMTERS
language = "en"
max_ngram_size = 1
deduplication_thresold = 0.9
deduplication_algo = 'seqm'
windowSize = 1
numOfKeywords = 8000


#CONTRUCTING POSITIVE DICTIONARY

#for positive order 
order_list1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/order_pos_dep_ver1'
with open(order_list1, "rb") as fp:   # Unpickling
    order_pos1 = pickle.load(fp)
    fp.close()

positive_text = ''
order_pos1.sort(key=lambda y: y[0], reverse = True) 
for i in range(len(order_pos1)):
    index_user = order_pos1[i][1]
    positive_text = '' + positive_dep[index_user] + '\n'

    

#for negative order 
order_list2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/order_neg_ver1'
with open(order_list2, "rb") as fp:   # Unpickling
    order_neg1 = pickle.load(fp)
    fp.close()

negative_text = ''
order_neg1.sort(key=lambda y: y[0], reverse = True) 
for i in range(len(order_neg1)):
    index_user = order_neg1[i][1]
    negative_text = '' + negative_dep[index_user] + '\n'

print("Begin key extraction")
text = positive_text
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
keywords1 = custom_kw_extractor.extract_keywords(text)
print("Ends key extraction")
name_key = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_pos_ver1'
with open(name_key, 'wb') as f:
	pickle.dump(keywords1, f)
	f.close()

#NEGATIVE DICTIONARY 
text = negative_text
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
keywords = custom_kw_extractor.extract_keywords(text)
print("End key extraction")

name_key = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_neg_ver1'
with open(name_key, 'wb') as f:
	pickle.dump(keywords, f)
	f.close()


## SECOND VERSION-- LOWER THE 
def norm(document):

    document = [x.lower()  for x in document]

    return document
#for positive order 
order_list1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/order_pos_dep_ver2'
with open(order_list1, "rb") as fp:   # Unpickling
    order_pos2 = pickle.load(fp)
    fp.close()

positive_text2 = ''
order_pos2.sort(key=lambda y: y[0], reverse = True) 
for i in range(len(order_pos2)):
    index_user = order_pos2[i][1]
    positive_text2 = '' + positive_dep[index_user] + '\n'

    

#for negative order 
order_list2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/order_neg_dep_ver2'
with open(order_list2, "rb") as fp:   # Unpickling
    order_neg2 = pickle.load(fp)
    fp.close()

negative_text2 = ''
order_neg2.sort(key=lambda y: y[0], reverse = True) 
for i in range(len(order_neg2)):
    index_user = order_neg2[i][1]
    negative_text2 = '' + negative_dep[index_user] + '\n'


text = positive_text2.lower()
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
keywords1 = custom_kw_extractor.extract_keywords(text)
print("Ends key extraction")
name_key = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_pos_ver2'
with open(name_key, 'wb') as f:
	pickle.dump(keywords1, f)
	f.close()


text = negative_text2.lower()
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
keywords = custom_kw_extractor.extract_keywords(text)
print("End key extraction")

name_key = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_neg_ver2'
with open(name_key, 'wb') as f:
	pickle.dump(keywords, f)
	f.close()
