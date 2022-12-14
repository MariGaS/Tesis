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
#number of positive users 
users_2017p = len(get_urls(train_pos_2017))
users_2018p = len(get_urls(train_pos_2018))
users_2017n = len(get_urls(train_neg_2017))
users_2018n = len(get_urls(train_neg_2018))
n_p = users_2017p+users_2018p
#number of negative users     
n_n = users_2017n+users_2018n 



#STATE YAKE PARAMTERS
language = "en"
max_ngram_size = 1
deduplication_thresold = 0.9
deduplication_algo = 'seqm'
windowSize = 1
numOfKeywords = 1000


#CONTRUCTING POSITIVE DICTIONARY
#list with all the keywords from each post 
pre_dictionary = []
#list with lists with the positions of the keywords of each post 
positions = []
score_negative = []
print("Begin key extraction")
for i in range(n_p): 
    text,score = get_order_text(positive_dep[i], 'positive')
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
    keywords1 = custom_kw_extractor.extract_keywords(text)
    #append the keywords of the text given 
    score_negative.append((score, i))
    pre_dictionary.append(keywords1)
    #append the list with the positions of this 
    positions.append([x for x in range(1,len(keywords1)+1)])
print("Ends key extraction")
#a list with ALL the keywords and with their correspond positions in ther original post 
words, words_positions = get_dict_position(pre_dictionary, positions)
final_dic_pos =get_dictionary(words,words_positions, n_p)
#to order from the highest to  the lowest 
final_dic_pos.sort(key=lambda y: y[1], reverse = True) 

name_key = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_pos_ver5'
with open(name_key, 'wb') as f:
	pickle.dump(final_dic_pos, f)
	f.close()
#name_scores = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/order_pos_dep_ver1'
#with open(name_scores, 'wb') as f:
#	pickle.dump(score_negative, f)
#	f.close()

#NEGATIVE DICTIONARY 
pre_dictionary = []
#list with lists with the positions of the keywords of each post 
positions = []
#number of posts from negative user 
score_negative = []
print("Begins key extraction")
for i in range(n_n): 
    text,score = get_order_text(negative_dep[i], 'negative')
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
    keywords = custom_kw_extractor.extract_keywords(text)
    #append the keywords of the text given 
    score_negative.append((score, i))    
    pre_dictionary.append(keywords)
    #append the list with the positions of this 
    positions.append([x for x in range(1,len(keywords)+1)])
print("End key extraction")
#a list with ALL the keywords and with their correspond positions in ther original post 
words, words_positions = get_dict_position(pre_dictionary, positions)
final_dic_neg =get_dictionary(words,words_positions, n_n)
#to order from the highest to  the lowest 
final_dic_neg.sort(key=lambda y: y[1], reverse = True) 

name_key = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_neg_ver5'
with open(name_key, 'wb') as f:
	pickle.dump(final_dic_neg, f)
	f.close()

#name_scores = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/order_neg_dep_ver1'
#with open(name_scores, 'wb') as f:
#	pickle.dump(score_negative, f)
#	f.close()


## SECOND VERSION-- LOWER THE 
def norm(document):

    document = [x.lower()  for x in document]

    return document

#CONTRUCTING POSITIVE DICTIONARY
#list with all the keywords from each post 
pre_dictionary = []
#list with lists with the positions of the keywords of each post 
positions = []
score_negative = []
print("Begin key extraction")
for i in range(n_p):
    t = norm(positive_dep[i]) 
    text, score = get_order_text(t, 'positive')
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
    keywords1 = custom_kw_extractor.extract_keywords(text)
    score_negative.append((score, i))
    #append the keywords of the text given
    pre_dictionary.append(keywords1)
    #append the list with the positions of this 
    positions.append([x for x in range(1,len(keywords1)+1)])
print("Ends key extraction")
#a list with ALL the keywords and with their correspond positions in ther original post 
words, words_positions = get_dict_position(pre_dictionary, positions)
final_dic_pos =get_dictionary(words,words_positions, n_p)
#to order from the highest to  the lowest 
final_dic_pos.sort(key=lambda y: y[1], reverse = True) 

name_key = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_pos_ver6'
with open(name_key, 'wb') as f:
	pickle.dump(final_dic_pos, f)
	f.close()
#name_scores = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/order_pos_dep_ver2'
#with open(name_scores, 'wb') as f:
#	pickle.dump(score_negative, f)
#	f.close()



#NEGATIVE DICTIONARY 
pre_dictionary = []
#list with lists with the positions of the keywords of each post 
positions = []
#number of posts from negative user 
score_negative = []
print("Begins key extraction")
for i in range(n_n): 
    t = norm(negative_dep[i])
    text,score = get_order_text(t, 'negative')
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
    keywords = custom_kw_extractor.extract_keywords(text)
    score_negative.append((score, i))
    #append the keywords of the text given 
    pre_dictionary.append(keywords)
    #append the list with the positions of this 
    positions.append([x for x in range(1,len(keywords)+1)])
print("End key extraction")
#a list with ALL the keywords and with their correspond positions in ther original post 
words, words_positions = get_dict_position(pre_dictionary, positions)
final_dic_neg =get_dictionary(words,words_positions, n_n)
#to order from the highest to  the lowest 
final_dic_neg.sort(key=lambda y: y[1], reverse = True) 

name_key = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_neg_ver6'
with open(name_key, 'wb') as f:
	pickle.dump(final_dic_neg, f)
	f.close()
#name_scores = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/order_neg_dep_ver2'
#with open(name_scores, 'wb') as f:
#	pickle.dump(score_negative, f)
#	f.close()