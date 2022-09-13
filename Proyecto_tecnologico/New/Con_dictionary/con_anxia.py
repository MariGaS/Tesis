from text_functions import *
import yake
import pickle

#EXTRACT TRAIN DATA FOR THE DICTIONARIES 
print('Begin extract of train text')
anxia_train = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Anorexia_2018/Anorexia_Datasets_1/train'
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
print('End extraction')
#number of positive users 
n_p = len(all_pos)
#number of negative users     
n_n = len(all_neg)       


#STATE YAKE PARAMTERS
language = "en"
max_ngram_size = 1
deduplication_thresold = 0.9
deduplication_algo = 'seqm'
windowSize = 1
numOfKeywords = 8000


#for positive order 
order_list1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/order_pos_ver1'
with open(order_list1, "rb") as fp:   # Unpickling
    order_pos1 = pickle.load(fp)
    fp.close()

positive_text = ''
order_pos1.sort(key=lambda y: y[0], reverse = True) 
for i in range(len(order_pos1)):
    index_user = order_pos1[i][1]
    positive_text = '' + all_pos[index_user] + '\n'

    

#for negative order 
order_list2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/order_neg_ver1'
with open(order_list2, "rb") as fp:   # Unpickling
    order_neg1 = pickle.load(fp)
    fp.close()

negative_text = ''
order_neg1.sort(key=lambda y: y[0], reverse = True) 
for i in range(len(order_neg1)):
    index_user = order_neg1[i][1]
    negative_text = '' + all_neg[index_user] + '\n'

#CONTRUCTING POSITIVE DICTIONARY
#number of posts from positive user 
print("Begin key extraction")
text = positive_text
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
keywords1 = custom_kw_extractor.extract_keywords(text)
print("Ends key extraction")

name_key = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_pos_ver1'
with open(name_key, 'wb') as f:
	pickle.dump(keywords1, f)
	f.close()

#NEGATIVE DICTIONARY 
print("Begins key extraction")
text = get_order_text(negative_text)
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
keywords = custom_kw_extractor.extract_keywords(text)
print("End key extraction")

name_key = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_neg_ver1'
with open(name_key, 'wb') as f:
	pickle.dump(keywords, f)
	f.close()


## SECOND VERSION-- LOWER THE 
def norm(document):

    document = [x.lower()  for x in document]

    return document

#CONTRUCTING POSITIVE DICTIONARY
#for positive order 
order_list3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/order_pos_ver2'
with open(order_list3, "rb") as fp:   # Unpickling
    order_pos2 = pickle.load(fp)
    fp.close()

positive_text = ''
order_pos2.sort(key=lambda y: y[0], reverse = True) 
for i in range(len(order_pos2)):
    index_user = order_pos2[i][1]
    positive_text = '' + all_pos[index_user] + '\n'

    

#for negative order 
order_list4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/order_neg_ver2'
with open(order_list4, "rb") as fp:   # Unpickling
    order_neg2 = pickle.load(fp)
    fp.close()

negative_text = ''
order_neg2.sort(key=lambda y: y[0], reverse = True) 
for i in range(len(order_neg2)):
    index_user = order_neg2[i][1]
    negative_text = '' + all_neg[index_user] + '\n'

print("Begin key extraction")
text = positive_text.lower()
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
keywords1 = custom_kw_extractor.extract_keywords(text)
print("Ends key extraction")

name_key = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_pos_ver2'
with open(name_key, 'wb') as f:
	pickle.dump(keywords1, f)
	f.close()

#NEGATIVE DICTIONARY 
text = negative_text.lower()
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
keywords = custom_kw_extractor.extract_keywords(text)
print("End key extraction")
name_key = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_neg_ver2'
with open(name_key, 'wb') as f:
	pickle.dump(keywords, f)
	f.close()
