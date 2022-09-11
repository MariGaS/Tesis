from User_dictionary.text_functions import *
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
#all the posts from positive users
positive_posts = [item for sublist in all_pos for item in sublist]
#all the posts from negative users
negative_posts = [item for sublist in all_neg for item in sublist]

#STATE YAKE PARAMTERS
language = "en"
max_ngram_size = 1
deduplication_thresold = 0.9
deduplication_algo = 'seqm'
windowSize = 1
numOfKeywords = 30


#CONTRUCTING POSITIVE DICTIONARY
#list with all the keywords from each post 
pre_dictionary = []
#list with lists with the positions of the keywords of each post 
positions = []
#number of posts from positive user 
num_pos_post = len(positive_posts)
print("Begin key extraction")
for i in range(num_pos_post): 
    text = positive_posts[i]
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
    keywords1 = custom_kw_extractor.extract_keywords(text)
    #append the keywords of the text given 
    pre_dictionary.append(keywords1)
    #append the list with the positions of this 
    positions.append([x for x in range(1,len(keywords1)+1)])
print("Ends key extraction")
#a list with ALL the keywords and with their correspond positions in ther original post 
dic_pos = get_dict_position(pre_dictionary, positions)
final_dic_pos = make_final_dic(dic_pos, n_p)
#to order from the highest to  the lowest 
final_dic_pos.sort(key=lambda y: y[1], reverse = True) 

name_key = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Post_dictionary/post_level/anxia_pos_ver1key30'
with open(name_key, 'wb') as f:
	pickle.dump(final_dic_pos, f)
	f.close()

#NEGATIVE DICTIONARY 
pre_dictionary = []
#list with lists with the positions of the keywords of each post 
positions = []
#number of posts from negative user 
num_neg_post = len(negative_posts)
print("Begins key extraction")
for i in range(num_neg_post): 
    text = negative_posts[i]
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
    keywords = custom_kw_extractor.extract_keywords(text)
    #append the keywords of the text given 
    pre_dictionary.append(keywords)
    #append the list with the positions of this 
    positions.append([x for x in range(1,len(keywords)+1)])
print("End key extraction")
#a list with ALL the keywords and with their correspond positions in ther original post 
dic_neg = get_dict_position(pre_dictionary, positions)
final_dic_neg = make_final_dic(dic_neg, n_n)
#to order from the highest to  the lowest 
final_dic_neg.sort(key=lambda y: y[1], reverse = True) 

name_key = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Post_dictionary/post_level/anxia_neg_ver1key30'
with open(name_key, 'wb') as f:
	pickle.dump(final_dic_neg, f)
	f.close()


## SECOND VERSION-- LOWER THE 
def normalize(document):

    document = [x.lower()  for x in document]

    return document

#CONTRUCTING POSITIVE DICTIONARY
#list with all the keywords from each post 
pre_dictionary = []
#list with lists with the positions of the keywords of each post 
positions = []
#number of posts from positive user 
num_pos_post = len(positive_posts)
positive_posts = normalize(positive_posts)
print("Begin key extraction")
for i in range(num_pos_post): 
    text = positive_posts[i]
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
    keywords1 = custom_kw_extractor.extract_keywords(text)
    #append the keywords of the text given 
    pre_dictionary.append(keywords1)
    #append the list with the positions of this 
    positions.append([x for x in range(1,len(keywords1)+1)])
print("Ends key extraction")
#a list with ALL the keywords and with their correspond positions in ther original post 
dic_pos = get_dict_position(pre_dictionary, positions)
final_dic_pos = make_final_dic(dic_pos, n_p)
#to order from the highest to  the lowest 
final_dic_pos.sort(key=lambda y: y[1], reverse = True) 

name_key = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Post_dictionary/post_level/anxia_pos_ver2key30'
with open(name_key, 'wb') as f:
	pickle.dump(final_dic_pos, f)
	f.close()

#NEGATIVE DICTIONARY 
pre_dictionary = []
#list with lists with the positions of the keywords of each post 
positions = []
#number of posts from negative user 
num_neg_post = len(negative_posts)
negative_posts = normalize(negative_posts)
print("Begins key extraction")
for i in range(num_neg_post): 
    text = negative_posts[i]
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
    keywords = custom_kw_extractor.extract_keywords(text)
    #append the keywords of the text given 
    pre_dictionary.append(keywords)
    #append the list with the positions of this 
    positions.append([x for x in range(1,len(keywords)+1)])
print("End key extraction")
#a list with ALL the keywords and with their correspond positions in ther original post 
dic_neg = get_dict_position(pre_dictionary, positions)
final_dic_neg = make_final_dic(dic_neg, n_n)
#to order from the highest to  the lowest 
final_dic_neg.sort(key=lambda y: y[1], reverse = True) 

name_key = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Post_dictionary/post_level/anxia_neg_ver2key30'
with open(name_key, 'wb') as f:
	pickle.dump(final_dic_neg, f)
	f.close()
