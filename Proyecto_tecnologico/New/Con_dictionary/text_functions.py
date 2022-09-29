import xml.etree.ElementTree as ET
from os import listdir
from os.path import isfile, join
from more_itertools import locate
import numpy as np
import re 
from nltk import TweetTokenizer
from gensim.models import FastText
# #####--- FUNCTIONS TO READ TRAINING DATA ----########## #
model = FastText.load('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Models/emotions.model')

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

        all_documents.append(document)


    return all_documents


def get_text_chunk(path): 
    all_documents = [] #list with all the documents 

    user_path = get_urls(path)
    for i in range(len(user_path)): 
        subject = user_path[i] #for example test_subjet1005.xml 
        path_subject = path + '/' + subject 
        document = get_text_post_train(path_subject) #get document with all the history of a user 

        all_documents.append(document)
    return all_documents
    
# DICTIONARY OF EMOTIONS





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

    return count


def define_order_post(count, user_hist, version):
    #count is the array with the negative count of each post 
    #we order by the maximun to the minimum
    values_count = np.unique(count)
    values_count = values_count[::-1] 
    
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