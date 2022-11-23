import xml.etree.ElementTree as ET
from os import listdir
from os.path import isfile, join
from more_itertools import locate
import numpy as np
import re 
from nltk import TweetTokenizer
from gensim.models import FastText
model = FastText.load('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Models/emotions.model')
# #####--- FUNCTIONS TO READ TRAINING DATA ----########## #


## Functions to read  post from the users ##
def get_urls(path): 
    # Function that return the name of the subjects in the folder 

    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    onlyfiles.sort()

    return onlyfiles

def get_text_post_train(user_path):
    tree = ET.parse(user_path)
    root = tree.getroot()
    # list with entries from the user 
    all_post = []
    #iterate recursively over all the sub-tree 
    #source : https://docs.python.org/3/library/xml.etree.elementtree.html
    for post in root.iter('WRITING'): 

        for t in post.iter('TITLE'):
            p = ''
            entry = t.text
            
            if entry != ' ' and entry != '  ' and entry != '' and entry!= '\n' and entry!= '   ': 
                if entry[-1] == ' ':
                    p = entry[1:-1] + ' '
                elif entry[-1] == ' ' and entry[-2] == '  ': 
                    p = entry[1:-2] + ' '
                elif entry[-1] == ' ' and entry[-2] == '  ' and entry[-3] == '   ': 
                    p = entry[1:-3] + ' '
                else:
                    p = entry[1:] + ' '
        for t in post.iter('TEXT'):
            entry = t.text
            
            if entry != ' ' and entry != '  ' and entry != '' and entry != '   ':
                if entry[-1] == ' ' and entry[-2] != ' ' :
                    p = p + entry[1:-1]
                elif entry[-1] == ' ' and entry[-2] == ' ' and entry[-3] != ' ': 
                    p = p + entry[1:-2]
                elif entry[-1] == ' ' and entry[-2] == ' ' and entry[-3] == ' ': 
                    p = p + entry[1:-3]
                else:
                    p = p + entry[1:]

        all_post.append(p)
    return all_post


def get_text_depression(path): 
    #Function to get all the posts from training set of depression users 

    all_documents = [] #list with all the documents 

    user_path = get_urls(path)
    for i in range(len(user_path)): 
        subject = user_path[i] #for example test_subjet1005.xml 
        path_subject = path + '/' + subject 
        document = get_text_post_train(path_subject) #get document with all the history of a user 

        all_documents += [document] 


    return all_documents


def get_text_chunk(path): 
    all_documents = [] #list with all the documents 

    user_path = get_urls(path)
    for i in range(len(user_path)): 
        subject = user_path[i] #for example test_subjet1005.xml 
        path_subject = path + '/' + subject 
        document = get_text_post_train(path_subject) #get document with all the history of a user 

        all_documents += [document] 
    return all_documents
    
# DICTIONARY OF EMOTIONS

def normalize(document):
    #document = [x.lower() for x in document]
    document = [re.sub(r'https?:\/\/\S+', '', x) for x in document] #eliminate url
    document = [re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', x) for x in document] #eliminate url 
    document = [re.sub(r'{link}', '', x) for x in document] #eliminate link
    document = [re.sub(r"\[video\]", '', x) for x in document] #eliminate video 
    #document = [re.sub(r'\s+', ' ' '', x).strip() for x in document]
    #document = [re.sub(r',', ' ' '', x).strip() for x in document]
    document = [x.replace("#","") for x in document]  #eliminate #
    #document = [re.subn(r'[^\w\s,]',"", x)[0].strip() for x in document] #eliminate emoticons 
    return document



#FOR THE CONCATANATION OF THE POSTS IN ORDER #
tokenizer = TweetTokenizer()
def count_negative(user_hist):
    #function to count the negative and positive emotions  in
    #the historial of the user 
    count = np.zeros(len(user_hist))

    for i in range(len(user_hist)):
        #a given post from the historial
        user_post= user_hist[i]
        corpus_palabras = []
        #obtienes todas las palabras del documento 
        corpus_palabras += tokenizer.tokenize(user_post)
        
        for word in corpus_palabras:
            if  word.isnumeric() == False:
                count[i] += model.wv.similarity(word, 'negative')
                count[i] += model.wv.similarity(word,'disgust')
                count[i] += model.wv.similarity(word, 'anger')
                count[i] += model.wv.similarity(word, 'fear')
                count[i] += model.wv.similarity(word, 'sadness')
                count[i] -= model.wv.similarity(word, 'joy')
                count[i] -= model.wv.similarity(word, 'positive') 
                count[i] -= model.wv.similarity(word, 'trust')
                count[i] -= model.wv.similarity(word, 'anticipation')
                count[i] -= model.wv.similarity(word, 'surprise')

    user_negative = np.sum(count)
    return count, user_negative

def define_order_post(count, user_hist, version):
    #count is the array with the negative count of each post 
    #we order by the maximun to the minimum
    values_count = np.unique(count)
    #values_count = values_count[::-1] 
    
    if version == 'negative':
        text = ''
        for i in range(values_count.shape[0]):
            index = np.where(count == values_count[i])
            for j in range(index[0].shape[0]):
                ind = index[0][j]
                text = text + user_hist[ind] + '\n'
    
    if version == 'positive':
        text = ''
        v = values_count[::-1]
        for i in range(values_count.shape[0]):

            index = np.where(count == v[i])
            for j in range(index[0].shape[0]):
                ind = index[0][j]
                text = text + user_hist[ind] + '\n'        

    return text 

def get_order_text(user_hist, d_version):
    norm_user = normalize(user_hist)
    count_user, negative_score = count_negative(norm_user)
    text_user = define_order_post(count_user,user_hist, version= d_version)

    return text_user,negative_score

## CONSTRUCTION OF FINAL DICTIONARY 
def get_dictionary(list1, list2, number_user):
    final_dictionary = dict()

    for i in range(len(list1)):
        word = list1[i]
        word_ranking = list2[i]
        #update value 
        if word in final_dictionary: 
            final_dictionary[word] +=  word_ranking
        #add a new word 
        else: 
            final_dictionary[word] = word_ranking
    dictionary = list(final_dictionary.items())
    dic = [(x,y/number_user) for (x,y) in dictionary]
    return dic
        

def get_dict_position(list1,list2):
    #Function that make a dictionary concatenating all the keywords of the posts
    #list1 is the list with all the keywords of each post 
    #list2 has all the positions of this words
    #merge all the sublist in list1 in one list, the same for list2
    x = [item for sublist in list1 for item in sublist]    
    x = [i[0] for i in x]
    y = [item for sublist in list2 for item in sublist]
    # the list contain the word with the rankings from their respective posts

    return x,y




