import pickle
import yake
from nltk import TweetTokenizer
import re
from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as ET #library for read xml archives
import random

tokenizer = TweetTokenizer()
def normalize(document):
    # Function thtat cleans the text 

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

## Functions to read  post from the users ##
def get_urls(path): 
    # Function that return the name of the subjects in the folder 

    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    onlyfiles.sort()

    return onlyfiles

def get_text_post(user_path):
    #Function that gets the post from a user 

    #The tree of the user 
    tree = ET.parse(user_path)
    # Obtain the root of the three
    root = tree.getroot()
    # List with the historial of the user post 
    hist_post = [] # list with entries from the user 

    #iterate recursively over all the sub-tree 
    #source : https://docs.python.org/3/library/xml.etree.elementtree.html

    #Iterate first by the root childs named 'WRITTING'
    for post in root.iter('WRITING'): 
        #The iterate by the child with name 'TITLE', this correspond to the title of each one of the posts 
        for t in post.iter('TITLE'):
            entry = t.text
            # Check that the entry is not empty 
            if entry != ' ' and entry != '  ' and entry != '' and entry!= '\n' and entry!= '   ': 
                #This is only to only obtain the text without blank space at the begining and final of the entry 
                if entry[-1] == ' ':
                    hist_post.append(entry[1:-1])
                elif entry[-1] == ' ' and entry[-2] == '  ': 
                    hist_post.append(entry[1:-2])
                elif entry[-1] == ' ' and entry[-2] == '  ' and entry[-3] == '   ': 
                    hist_post.append(entry[1:-3])
                else:
                    hist_post.append(entry[1:])
        #Iterate by the child with name 'TEXT' this entry correspond to the post's text 
        for t in post.iter('TEXT'):
            entry = t.text
            
            if entry != ' ' and entry != '  ' and entry != '' and entry != '   ':
                if entry[-1] == ' ' and entry[-2] != ' ' :
                    hist_post.append(entry[1:-1])
                elif entry[-1] == ' ' and entry[-2] == ' ' and entry[-3] != ' ': 
                    hist_post.append(entry[1:-2])
                elif entry[-1] == ' ' and entry[-2] == ' ' and entry[-3] == ' ': 
                    hist_post.append(entry[1:-3])
                else:
                    hist_post.append(entry[1:])
    #for this we only want the list of the posts
    return hist_post


# DEPRESSION PARTICULAR FUNCTIONS #

def get_text_depression(path): 
    #Function to get all the posts from training set of depression users 

    all_documents = [] #list with all the documents 

    user_path = get_urls(path)
    for i in range(len(user_path)): 
        subject = user_path[i] #for example test_subjet1005.xml 
        path_subject = path + '/' + subject 
        document = get_text_post(path_subject) #get document with all the history of a user 

        all_documents += [document] 

    #return a list with lists, each of this list are for each user in the path proportionated 
    #a each list have all the history post 
    return all_documents



# ANOREXIAS PARTICULAR FUNCTION #  
def text_from_chunk(path):
    #Function to obtain all the entries from a determinated chunk 
    #return a list which contains all the entries
    all_documents = []
    #Obtain the user path
    user_path = get_urls(path)

    for i in range(len(user_path)): 
        subject = user_path[i] #for example test_subjet1005.xml 
        path_subject = path + '/' + subject 
        #get document with all the history of a user 
        document = get_text_post(path_subject) 
        all_documents.append(document)

    return all_documents

# FUNCTIONS FOR YAKE! #
def get_lines(text, new_line = False):
    #concatenate all the strings in to a text
    string = ''
    if new_line: 
        for i in range(len(text)):
            string = string + text[i] + '\n'
    else: 
        for i in range(len(text)):
            string = string + ' ' + text[i]
    #text_change.append(string)
    return string


## ANOREXIA ##
anxia_train = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Anorexia_2018/Anorexia_Datasets_1/train'

pos = 'positive_examples'
neg = 'negative_examples'


all_pos = []
all_neg = []
for i in range(1,11):
    path_chunk_pos = anxia_train +'/' + pos + '/chunk'+str(i)
    path_chunk_neg = anxia_train +'/' + neg + '/chunk'+str(i)

    temp1 = text_from_chunk(path_chunk_pos)
    temp2 = text_from_chunk(path_chunk_neg)

    all_pos.append(temp1)
    all_neg.append(temp2)

#Concatenate all the lists with posts from each category in a list 

#this obtained all the posts from the list and make a 'flat lis', 
#flat_list has 200 list with each one of them have strings
flat_list_pos = [item for sublist in all_pos for item in sublist]
flat_list_neg = [item for sublist in all_neg for item in sublist]
#make flast list with only sentences as members
flat_list_pos = [item for sublist in flat_list_pos for item in sublist]
flat_list_neg = [item for sublist in flat_list_neg for item in sublist]
#shuffle flat_list to obatin new orders 
shuffle_list_pos= random.sample(flat_list_pos, len(flat_list_pos))
shuffle_list_neg= random.sample(flat_list_neg, len(flat_list_neg))
##TRAINING ANOREXIA YAKE! SHUFFLE VERSION## 

#normalize the posts 
all_pos_clean = normalize(shuffle_list_pos)
all_neg_clean = normalize(shuffle_list_neg)
#get the lines of the text in order to feed YAKE!
lines_pos = get_lines(all_pos_clean, False)
lines_neg = get_lines(all_neg_clean, False)

#YAKE Setup
kw_extractor = yake.KeywordExtractor()
language = "en"
max_ngram_size = 1
deduplication_threshold = 0.9
numOfKeywords = 8000
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)

#Positive dictionary 
text = lines_pos
keywords1 = custom_kw_extractor.extract_keywords(text)
name_key = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Shuffle_yake/shuffle_anxia_pos' 

with open(name_key, 'wb') as f:
	pickle.dump(keywords1, f)
	f.close()

#Negative dictionary 
text = lines_neg
keywords1 = custom_kw_extractor.extract_keywords(text)
name_key = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Shuffle_yake/shuffle_anxia_neg' 

with open(name_key, 'wb') as f:
	pickle.dump(keywords1, f)
	f.close()
#second version 


#get the lines of the text in order to feed YAKE!
lines_pos2 = get_lines(all_pos_clean, True)
lines_neg2 = get_lines(all_neg_clean, True)

#Positive dictionary 
text = lines_pos2
keywords1 = custom_kw_extractor.extract_keywords(text)
name_key = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Shuffle_yake/shuffle_anxia_pos2' 

with open(name_key, 'wb') as f:
	pickle.dump(keywords1, f)
	f.close()

#Negative dictionary 
text = lines_neg2
keywords1 = custom_kw_extractor.extract_keywords(text)
name_key = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Shuffle_yake/shuffle_anxia_neg2' 

with open(name_key, 'wb') as f:
	pickle.dump(keywords1, f)
	f.close()

## DEPRESSION ###
"""
train_neg_2017 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/depression2022/training_data/2017_cases/neg'
train_pos_2017 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/depression2022/training_data/2017_cases/pos'
train_neg_2018 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/depression2022/training_data/2018_cases/neg'
train_pos_2018 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/depression2022/training_data/2018_cases/pos'

tr_neg_2017= get_text_depression(train_neg_2017)
tr_pos_2017 = get_text_depression(train_pos_2017)
tr_neg_2018 = get_text_depression(train_neg_2018)
tr_pos_2018 = get_text_depression(train_pos_2018)

#for positive dictionary 
list_pos = [*tr_pos_2017,*tr_pos_2018]
#neg dic
list_neg = [*tr_neg_2017,*tr_neg_2018]
#make flat list
flat_dep_pos = [item for sublist in list_pos for item in sublist]
flat_dep_neg = [item for sublist in list_neg for item in sublist]
#shuffle list 
shuffle_dep_pos = random.sample(flat_dep_pos, len(flat_dep_pos))
shuffle_dep_neg = random.sample(flat_dep_neg, len(flat_dep_neg))


##TRAINING DEPRESSION YAKE! SHUFFLE VERSION## 

#normalize the posts 
pos_clean = normalize(shuffle_dep_pos)
neg_clean = normalize(shuffle_dep_neg)
#get the lines of the text in order to feed YAKE!
dep_lines_pos = get_lines(pos_clean, False)
dep_lines_neg = get_lines(neg_clean, False)

#YAKE Setup
kw_extractor = yake.KeywordExtractor()
language = "en"
max_ngram_size = 1
deduplication_threshold = 0.9
numOfKeywords = 15000
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)

#Positive dictionary 
text = dep_lines_pos
keywords1 = custom_kw_extractor.extract_keywords(text)
name_key = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Shuffle_yake/shuffle_dep_pos' 

with open(name_key, 'wb') as f:
	pickle.dump(keywords1, f)
	f.close()

#Negative dictionary 
text = dep_lines_neg
keywords1 = custom_kw_extractor.extract_keywords(text)
name_key = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Shuffle_yake/shuffle_dep_neg' 

with open(name_key, 'wb') as f:
	pickle.dump(keywords1, f)
	f.close()


#second version 
#get the lines of the text in order to feed YAKE!
dep_lines_pos2 = get_lines(pos_clean, True)
dep_lines_neg2 = get_lines(neg_clean, True)

#Positive dictionary 
text = dep_lines_pos2
keywords1 = custom_kw_extractor.extract_keywords(text)
name_key = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Shuffle_yake/shuffle_dep_pos2' 

with open(name_key, 'wb') as f:
	pickle.dump(keywords1, f)
	f.close()

#Negative dictionary 
text = dep_lines_neg2
keywords1 = custom_kw_extractor.extract_keywords(text)
name_key = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Shuffle_yake/shuffle_dep_neg2' 

with open(name_key, 'wb') as f:
	pickle.dump(keywords1, f)
	f.close()
"""
