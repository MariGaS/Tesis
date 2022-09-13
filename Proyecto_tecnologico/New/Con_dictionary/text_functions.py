import xml.etree.ElementTree as ET
from os import listdir
from os.path import isfile, join
from more_itertools import locate
import numpy as np
import re 
from nltk import TweetTokenizer
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
    all_post = ''
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

        all_post = all_post + p + ' '
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

def normalize(document, all):
    document = [x.lower() for x in document]
    if all == True: 
        document = [re.sub(r'https?:\/\/\S+', '', x) for x in document] #eliminate url
        document = [re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', x) for x in document] #eliminate url 
        document = [re.sub(r'{link}', '', x) for x in document] #eliminate link
        document = [re.sub(r"\[video\]", '', x) for x in document] #eliminate video 
        document = [re.sub(r'\s+', ' ' '', x).strip() for x in document]
        document = [re.sub(r',', ' ' '', x).strip() for x in document]
        document = [x.replace("#","") for x in document]  #eliminate #
        document = [re.subn(r'[^\w\s,]',"", x)[0].strip() for x in document] #eliminate emoticons 
    return document


def get_emotion_from_file(path_corpus): 
    words_list = []
    emotion_list = []
    is_in  = []
    with open(path_corpus, "r") as f: 

        header = 0
        for line in f :
            words = re.split(r'\t+', line)  #cada linea se divide por palabra 
            words_list.append(words[0])
            emotion_list.append(words[1])
            is_in.append(words[2][:-1])
 
          
            
    return words_list, emotion_list, is_in
path_emotions = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'
l1,l2,l3 = get_emotion_from_file(path_emotions)


def get_dic_emotions(l_words,l_emotions,l_is_in):
    dict_emotions = dict()
    cont = 1
    emotions = []
    for i in range(len(l_words)):
        if cont%10 != 0 and i%10 != 9:
            #print(i,cont,'i')
            if l_is_in[i] == '1': 
                emotions.append(l_emotions[i])
            
            cont +=1
            #print(i,cont,'f')
            #print(emotions)
        if cont%10 == 0 and i%10 == 9:
            if l_is_in[i] == '1':
                emotions.append(l_emotions[i])
            if len(emotions)>0:
                dict_emotions[l_words[i]] = emotions
            #print(i,cont,'l')
            #print(emotions)
            emotions = []
            cont = 1
            #print(l_words[i],i)
    return dict_emotions
    
dict_emotions = get_dic_emotions(l1,l2,l3)

#FOR THE CONCATANATION OF THE POSTS IN ORDER #
tokenizer = TweetTokenizer()
def count_negative(dict_emo,user_hist):
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
            if word in dict_emo and word.isnumeric() == False:
                for j in range(len(dict_emo[word])):
                    if dict_emo[word][j] == 'anger':
                        count[i] += 1
                    elif dict_emo[word][j] == 'disgust':
                        count[i]+= 1 
                    elif dict_emo[word][j] == 'fear':
                        count[i] += 1 
                    elif dict_emo[word][j] == 'joy':
                        count[i] -= 1 
                    elif dict_emo[word][j] == 'negative':
                        count[i] += 1 
                    elif dict_emo[word][j] == 'positive':
                        count[i] -= 1 
                    elif dict_emo[word][j] == 'sadness':
                        count[i] += 1 
                    elif dict_emo[word][j] == 'trust':
                        count[i] -= 1 
                    elif dict_emo[word][j] == 'anticipation':
                        count[i] -= 1 
                    elif dict_emo[word][j] == 'surprise':
                        count[i] -= 1  
    return count
def define_order_post(count, user_hist):
    #count is the array with the negative count of each post 
    #we order by the maximun to the minimum
    values_count = np.unique(count)
    values_count = values_count[::-1] 
    text = ''
    for i in range(values_count.shape[0]):
        index = np.where(count == values_count[i])
        for j in range(index[0].shape[0]):
            ind = index[0][j]
            text = text + user_hist[ind] + ' '
    return text 

def get_order_text(user_hist):
    norm_user = normalize(user_hist, True)
    count_user = count_negative(dict_emotions, norm_user)
    text_user = define_order_post(count_user,user_hist)

    return text_user

## CONSTRUCTION OF FINAL DICTIONARY 


def get_dict_position(list1,list2):
    #Function that make a dictionary concatenating all the keywords of the posts
    #list1 is the list with all the keywords of each post 
    #list2 has all the positions of this words
    #merge all the sublist in list1 in one list, the same for list2
    x = [item for sublist in list1 for item in sublist]    
    x = [i[0] for i in x]
    y = [item for sublist in list2 for item in sublist]
    # the list contain the word with the rankings from their respective posts
    position_dic = [[x[i], y[i]] for i in range(0, len(x))]

    return position_dic


def make_final_dic(posi_dict, num_user):
    #posi_dict the dictionary to clean 
    #num_user the number of users to use for this dictionary 
    #contains only words
    words = [posi_dict[i][0] for i in range(len(posi_dict))]
    rankings = [posi_dict[i][1] for i in range(len(posi_dict))]
    #words in a numpy array
    w_array = np.array(words)
    #uniques words in  a numpy array does not preserve original order 
    unique_words = np.unique(w_array)
    #unique_words = set(words)
    
    final_dictionary = []
    #the numpy version of posi_dict
    np_rank = np.array(rankings)
    for i in range(unique_words.shape[0]):
        #we search where this word appears in the posi_dict 
        positions = np.where(w_array == unique_words[i])
        rank_of_word = np_rank[positions[0]].sum()
        final_dictionary.append((unique_words[i],rank_of_word/num_user))
    return final_dictionary   

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