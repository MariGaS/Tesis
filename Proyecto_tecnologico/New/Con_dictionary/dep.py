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
flat_list_pos = [item for sublist in positive_dep for item in sublist]
flat_list_neg = [item for sublist in negative_dep for item in sublist]

print("Begin key extraction")
count_positive = count_negative(flat_list_pos)
text = define_order_post(count_positive, flat_list_pos, 'positive')
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
keywords1 = custom_kw_extractor.extract_keywords(text)
print("Ends key extraction")
name_key = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_pos_ver5'
with open(name_key, 'wb') as f:
	pickle.dump(keywords1, f)
	f.close()

#NEGATIVE DICTIONARY 
count_negative1 = count_negative(flat_list_neg)
text = define_order_post(count_negative1, flat_list_neg, 'negative')
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
keywords = custom_kw_extractor.extract_keywords(text)
print("End key extraction")

name_key = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_neg_ver5'
with open(name_key, 'wb') as f:
	pickle.dump(keywords, f)
	f.close()


## SECOND VERSION-- LOWER THE 
def norm(document):

    document = [x.lower()  for x in document]

    return document


print("Begin key extraction")
count_positive = count_negative(flat_list_pos)
t = norm(flat_list_pos)
text = define_order_post(count_positive, t, 'positive')
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
keywords1 = custom_kw_extractor.extract_keywords(text)
print("Ends key extraction")

name_key = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_pos_ver6'
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

name_key = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_neg_ver6'
with open(name_key, 'wb') as f:
	pickle.dump(keywords1, f)
	f.close()
