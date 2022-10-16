from grp import struct_group
from turtle import pos
from gensim.parsing.preprocessing import remove_stopwords
from nltk import TweetTokenizer
import nltk
import numpy as np
from time import time
import re
from gensim.models import FastText
import sklearn
from sklearn import svm
from sklearn.metrics import  f1_score
from sklearn.model_selection import GridSearchCV
from gensim.models import FastText
import fasttext
import fasttext.util
import pickle
import logging
from clustering import clutering_addition

tokenizer = TweetTokenizer()
model_anxia = FastText.load(
    '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Models/anxia.model')
print('Load_anorexia')
model_dep = FastText.load(
    '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Models/depresion.model')
print('Load depression')
model_emo= FastText.load(
    '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Models/emotions.model')
print('Load emotions')

print('Load pretrained')
model_pre = fasttext.load_model(
    '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/cc.en.300.bin')


# Pre-processing functions
def normalize(document):

    document = [x.lower() for x in document]
    # eliminate url
    document = [re.sub(r'https?:\/\/\S+', '', x) for x in document]
    # eliminate url
    document = [re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', x)
                for x in document]
    # eliminate link
    document = [re.sub(r'{link}', '', x) for x in document]
    # eliminate video
    document = [re.sub(r"\[video\]", '', x) for x in document]
    document = [re.sub(r',', ' ' '', x).strip()
                for x in document]  # quita comas
    document = [re.sub(r'\s+', ' ' '', x).strip() for x in document]
    # eliminate #
    document = [x.replace("#", "") for x in document]
    # eliminate emoticons
    document = [re.subn(r'[^\w\s,]', "", x)[0].strip() for x in document]

    return document

# FUNCTION TO REMOVE STOP WORDS#
def remove_stop_list(document):
    document = [remove_stopwords(x) for x in document]

    return document

# Dictionary of frequent words #
def sortFreqDict(freqdict):
    aux = [(freqdict[key], key) for key in freqdict]
    aux.sort()
    aux.reverse()
    return aux

# Only obtain the words from the dictionary with the frequency of the words # 
def get_words(fdist_doc):
    words_doc = []
    for i, word in fdist_doc:
        words_doc.append(word)

    return words_doc


def get_fdist(text, num_feat):
    corpus_palabras = []
    for doc in text:
        corpus_palabras += tokenizer.tokenize(doc)
    fdist = nltk.FreqDist(corpus_palabras)
    fdist = sortFreqDict(fdist)
    fdist = fdist[:num_feat]
    return fdist

def get_dictionary_matrix(dictionary, option):
    dictionary_vec = np.zeros((len(set(dictionary)), 300), dtype=float)     
    for i in range(dictionary_vec.shape[0]):
        w1 = dictionary[i]
        if option == 1:
            dictionary_vec[i] = model_anxia.wv[w1]
        if option == 2:
            dictionary_vec[i] = model_dep.wv[w1]
        if option == 3: 
            dictionary_vec[i] = model_pre.get_word_vector(w1)
        if option == 4: 
            dictionary_vec[i] = model_emo.wv[w1]
    return dictionary_vec


def get_fuzzy_rep(words_user, dictionary_vec,epsilon):
    similarity_vocab = sklearn.metrics.pairwise.cosine_similarity(words_user, dictionary_vec)
    # vector de representación
    #vec_representation = np.count_nonzero(similarity_vocab > epsilon, axis=0)
    similarity_rep = np.where(similarity_vocab < epsilon, 0, similarity_vocab)
    #change elements that are greater than epsilon for 1 
    similarity_rep = np.where(similarity_rep >= epsilon, 1, similarity_rep)
    return similarity_rep

def get_sim_rep(words_user, dictionary_vec, epsilon):
    similarity_vocab = sklearn.metrics.pairwise.cosine_similarity(words_user, dictionary_vec)

    vec_representation = np.where(similarity_vocab < epsilon, 0, similarity_vocab)
    return vec_representation

def get_words_from_kw(kw):
    list1 = []
    for i in range(len(kw)):
        list1.append(kw[i][0])
    return list1


def get_list_key(path):
    # function that gets dictionaries from YAKE using pickle function
    with open(path, "rb") as fp:   # Unpickling
        b = pickle.load(fp)
        fp.close()
    return b


def dict_scores(d1, d2, feature1, feature2, no_distintc, con):

    if no_distintc == True:
        # obtain the
        kw1 = d1[:feature1]
        kw2 = d2[:feature2]

        l1 = get_words_from_kw(kw1)
        l2 = get_words_from_kw(kw2)

        return l1, l2

    else:

        dictionary1 = []
        dictionary2 = []
        l1 = get_words_from_kw(d1)
        l2 = get_words_from_kw(d2)
        i = 0
        while len(dictionary1) != feature1:
            w1 = l1[i]
            # revisamos si no está en el diccionario negativo
            if (w1 in l2) == False:
                dictionary1.append(w1)

            else:
                # en donde se encuentra en la lista l2
                indice = l2.index(w1)
                # score en la lista 1
                rel1 = d1[i][1]
                # score en la lista 2
                rel2 = d2[indice][1]
                # si el score de la lista 1 es menor que en la 1 la dejamos en lista 1
                if con == True: 
                    if rel1< rel2:
                        dictionary1.append(w1)
                else: 
                    if rel2 < rel1:
                        dictionary1.append(w1)
            i +=1 
        # dictionario negativo son todas las palabras que no están en dictionariopos
        dictionary2 = [x for x in l2 if x not in dictionary1][:feature2]
    return dictionary1, dictionary2


#POST LEVEL DICTIONARIES 
#---------------------------ANOREXIA-------------------#
#version 1
post_pos_anxia1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Post_dictionary/post_level/anxia_pos_ver1key30'
post_neg_anxia1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Post_dictionary/post_level/anxia_neg_ver1key30'
#version 2
post_pos_anxia2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Post_dictionary/post_level/anxia_pos_ver2key30'
post_neg_anxia2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Post_dictionary/post_level/anxia_neg_ver2key30'

#-------------------------DEPRESSION--------------------#
#version 1 
post_pos_dep1  =  '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Post_dictionary/post_level/dep_pos_ver1key30'
post_neg_dep1  =  '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Post_dictionary/post_level/dep_neg_ver1key30'
#version 2
post_pos_dep2  =  '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Post_dictionary/post_level/dep_pos_ver2key30'
post_neg_dep2  =  '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Post_dictionary/post_level/dep_neg_ver1key30'

##                                         USER VERSION OF THE DICTIONARIES OF THE USERS                        #
#---------------------------ANOREXIA-------------------#
#version 1
user_pos_anxia1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/anxia_pos_ver3'
user_neg_anxia1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/anxia_neg_ver3'
#version 2
user_pos_anxia2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/anxia_pos_ver4'
user_neg_anxia2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/anxia_neg_ver4'

user_pos_anxia3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/anxia_pos_ver5'
user_neg_anxia3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/anxia_neg_ver5'
#version 2
user_pos_anxia4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/anxia_pos_ver6'
user_neg_anxia4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/anxia_neg_ver6'

#---------------------------DEPRESSION-----------------#
#version 1
user_pos_dep1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_pos_ver3'
user_neg_dep1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_neg_ver3'
#version 2
user_pos_dep2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_pos_ver4'
user_neg_dep2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_neg_ver4'

user_pos_dep3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_pos_ver5'
user_neg_dep3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_neg_ver5'
#version 2
user_pos_dep4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_pos_ver6'
user_neg_dep4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_neg_ver6'

##                                                   CONCATANTE ALL THE TEXT 
#version 1
con_pos_anxia1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_pos_ver3'
con_neg_anxia1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_neg_ver3'
#version 2
con_pos_anxia2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_pos_ver4'
con_neg_anxia2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_neg_ver4'

con_pos_anxia3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_pos_ver5'
con_neg_anxia3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_neg_ver5'
#version 2
con_pos_anxia4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_pos_ver6'
con_neg_anxia4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_neg_ver6'

#---------------------------DEPRESSION-----------------#
#version 1
con_pos_dep1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_pos_ver3'
con_neg_dep1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_neg_ver3'
#version 2
con_pos_dep2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_pos_ver4'
con_neg_dep2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_neg_ver4'

con_pos_dep3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_pos_ver5'
con_neg_dep3= '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_neg_ver5'
#version 2
con_pos_dep4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_pos_ver6'
con_neg_dep4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_neg_ver6'


def add_group(score, group, matrix_add):
    #number of elements by group
    num_by_groups = score //group
    if score % group == 0: 
        left = 0
        right = num_by_groups-1

        for i in range(group):
            if i == 0: 
                X_add = np.sum(matrix_add[:, left:right], axis = 1)
            else: 
                X_temp = np.sum(matrix_add[:,left: right], axis = 1)
                X_add = np.column_stack((X_add, X_temp))
            left += num_by_groups
            right += num_by_groups
    else: 
        left = 0
        right = num_by_groups-1
        for i in range(group-1):
            if i == 0: 
                X_add= np.sum(matrix_add[:,left:right], axis = 1)
            else:
                X_temp = np.sum(matrix_add[:,left:right], axis = 1)
                X_add = np.column_stack((X_add, X_temp))

            left += num_by_groups
            right += num_by_groups
        
        right += score%group
        X_temp = np.sum(matrix_add[:,left:right], axis = 1)
        X_add = np.column_stack((X_add, X_temp))
    
    return X_add
                            




def add_groups(add, score1, score2, matrix_add1,matrix_add2, groups, num_col1, num_col2):
    
    if add == 'both':
        #only one number of group
        if len(groups) == 1: 
            add_1 = add_group(score1, groups[0], matrix_add1[:,:num_col1])
            add_2 = add_group(score2, groups[0], matrix_add2[:,:num_col2])

            matrix_add1 = np.column_stack((matrix_add1, add_1))
            matrix_add2 = np.column_stack((matrix_add2, add_2))
        else: 
            temp_1 = matrix_add1[:, :num_col1]
            temp_2 = matrix_add2[:, :num_col2]

            for i in range(len(groups)):
                add_1 = add_group(score1, groups[i], temp_1)
                add_2 = add_group(score2, groups[i], temp_2)

                matrix_add1 = np.column_stack((matrix_add1, add_1))
                matrix_add2 = np.column_stack((matrix_add2, add_2))
    if add == 'negative':
        #only one number of group
        if len(groups) == 1: 
            add_2 = add_group(score2, groups[0], matrix_add2[:,:num_col2])

            matrix_add2 = np.column_stack((matrix_add2, add_2))
        else: 
            temp_2 = matrix_add2[:,:num_col2]

            for i in range(len(groups)):
                add_2 = add_group(score2, groups[i], temp_2)

                matrix_add2 = np.column_stack((matrix_add2, add_2))   
    if add == 'positive':
        #only one number of group
        if len(groups) == 1: 
            add_1 = add_group(score2, groups[0], matrix_add1[:,:num_col1])

            matrix_add1 = np.column_stack((matrix_add1, add_1))
        else: 
            temp_1 = matrix_add1[:,:num_col1]

            for i in range(len(groups)):
                add_1 = add_group(score1, groups[i], temp_1)

                matrix_add1 = np.column_stack((matrix_add1, add_1)) 
    return matrix_add1, matrix_add2              


def get_matrix_cluster(num_cluster, num_dic, chose, positive, ex_type):
    if num_cluster == 5: 
        path_cluster = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Fuzzy_range/5-Cluster'
    if num_cluster == 10: 
        path_cluster = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Fuzzy_range/10-Cluster'
    if num_cluster == 20: 
        path_cluster = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Fuzzy_range/20-Cluster'
    
    if chose == 1: 
        emb = 'anxia'
    if chose == 2: 
        emb = 'dep'
    if chose == 3: 
        emb = 'pre'
    if chose == 4: 
        emb = 'emo'

    if num_dic == 1 or num_dic == 2: 
        path_cluster = path_cluster+ '/post_'
    if num_dic == 3 or num_dic == 4: 
        path_cluster = path_cluster+ '/user_'
    if num_dic == 5 or num_dic == 6: 
        path_cluster = path_cluster+ '/con_' 
    
    if positive == False:
        path_cluster = path_cluster+ 'neg_'
    if ex_type == 'depression':
        if chose == 3:
            path_cluster = path_cluster + 'dep_'
        if chose == 4: 
            path_cluster = path_cluster + 'dep_'

    path_cluster = path_cluster + emb + '_'
    if num_dic == 1 or num_dic == 3 or num_dic == 5: 
        path_cluster = path_cluster + 'uppercase'
    if num_dic == 2 or num_dic == 4 or num_dic == 6:
        path_cluster = path_cluster + 'lowercase'
    with open(path_cluster, "rb") as fp:   # Unpickling
        centers = pickle.load(fp)
        fp.close()    
    
    list_centers = list(centers.values())
    matrix_centers = get_dictionary_matrix(list_centers, option= chose)

    return matrix_centers

    

# function classificator
def classificator_pos_neg(all_path_train, all_path_test, path_tf_train, path_tf_test, num_test, num_train, score1, score2, 
                          di1, di2, tau, chose, groups, add,num_dic, ex_type, clustering, dif=True, 
                          fuzzy=True, compress=True, con = False, tf = True, w_clustering = False):
    # Parameters:
    # all_path_train: positive data from training data
    # all_path_test: negative data fromo training data
    # path_tf_train : the corpus of user with the frequency of each word  
    # score1, score 2: this select the words from the dictionaries
    # tau : tolerance for the similarity between the words
    # compress : if True. compress the matrix for the svm to a two dimensional array
    # dif:  the positive dictioinary and the negative dictionar does not have words in common
    # di1,di2 : the positive and negative dictionary
    #chose word embedding 
    # con : extra for other type of dictionary 
    # tf: if is necesary tf weight 
    # fuzzy : fuzzy weight 
    # add : where to add the groups 
    
    dictionary1, dictionary2 = dict_scores(
        di1, di2, score1, score2, no_distintc=dif, con=con)


    #construct th matrix for the dictionaries
    #positive dictionary 
    emb_dic1 = get_dictionary_matrix(dictionary=dictionary1, option =  chose)
    #ngative_dictionary 
    emb_dic2 = get_dictionary_matrix(dictionary=dictionary2, option =  chose)
    #num of columns in the emb_dic1
    n_col1 = emb_dic1.shape[1]
    n_col2 = emb_dic2.shape[1]

    if w_clustering == True: 
        if add == 'positive':
            for i in range(len(groups)):
                matrix_cluster = get_matrix_cluster(num_cluster=groups[i], num_dic=num_dic, positive=True, ex_type= ex_type,chose = chose)
                emb_dic1 = np.vstack((emb_dic1, matrix_cluster))
        if add == 'negative':
            for i in range(len(groups)):
                matrix_cluster = get_matrix_cluster(num_cluster=groups[i], num_dic=num_dic, positive=False, ex_type= ex_type, chose = chose)
                emb_dic2 = np.vstack((emb_dic2, matrix_cluster))
        if add == 'both':
            for i in range(len(groups)):
                matrix_cluster1 = get_matrix_cluster(num_cluster=groups[i], num_dic=num_dic, positive=True, ex_type= ex_type, chose = chose)
                emb_dic1 = np.vstack((emb_dic1, matrix_cluster1))
                matrix_cluster2 = get_matrix_cluster(num_cluster=groups[i], num_dic=num_dic, positive=False, ex_type= ex_type, chose= chose)
                emb_dic2 = np.vstack((emb_dic2, matrix_cluster2))            



   ### For using svm ####
    X_train1 = np.zeros((num_train, len(emb_dic1)), dtype=float)
    X_train2 = np.zeros((num_train, len(emb_dic2)), dtype=float)
    
    X_test1 = np.zeros((num_test, len(emb_dic1)),
                       dtype=float)  # matriz tipo document-term
    X_test2 = np.zeros((num_test, len(emb_dic2)), dtype=float)

    for i in range(num_train):
        path =all_path_train + '_'+ str(i)
        with open(path, "rb") as fp:   # Unpickling
            b = pickle.load(fp)
            fp.close()

        if fuzzy == True:
            word_repre_user = get_fuzzy_rep(b, emb_dic1, epsilon=tau)
        else:
            word_repre_user = get_sim_rep(b, emb_dic1, epsilon=tau)
        if tf == True:
            path2 =path_tf_train + '_'+ str(i)
            with open(path2, "rb") as fp:   # Unpickling
                corpus = pickle.load(fp)
                fp.close()
            for j in range(len(corpus)):
                frequency = corpus[j][0]
                word_repre_user[j] = frequency* word_repre_user[j]
        final_rep = np.sum(word_repre_user, axis=0)

        X_train1[i] = final_rep
    for i in range(num_train):
        path =all_path_train + '_'+ str(i)
        with open(path, "rb") as fp:   # Unpickling
            b = pickle.load(fp)
            fp.close()        
        if fuzzy == True:
            word_repre_user = get_fuzzy_rep(b, emb_dic2, epsilon=tau)
        else:
            word_repre_user = get_sim_rep(b, emb_dic2, epsilon=tau)
        if tf == True:
            path2 =path_tf_train + '_'+ str(i)
            with open(path2, "rb") as fp:   # Unpickling
                corpus = pickle.load(fp)
                fp.close()
            for j in range(len(corpus)):
                frequency = corpus[j][0]
                word_repre_user[j] = frequency* word_repre_user[j]
        final_rep = np.sum(word_repre_user, axis=0)
        X_train2[i] = final_rep

    # - Construct test representation of positive users
    
    for i in range(num_test):
        path =all_path_test + '_'+ str(i)
        with open(path, "rb") as fp:   # Unpickling
            b = pickle.load(fp)
            fp.close()    
        if fuzzy == True:
            word_repre_user = get_fuzzy_rep(b, emb_dic1, epsilon=tau)
        else:
            word_repre_user = get_sim_rep(b,emb_dic1, epsilon=tau)
        if tf == True:
            path2 =path_tf_test + '_'+ str(i)
            with open(path2, "rb") as fp:   # Unpickling
                corpus = pickle.load(fp)
                fp.close()
            for j in range(len(corpus)):
                frequency = corpus[j][0]
                word_repre_user[j] = frequency* word_repre_user[j]
        final_rep = np.sum(word_repre_user, axis=0)
        X_test1[i] = final_rep


    # Construct test representation of negative users
    for i in range(num_test):
        path =all_path_test + '_'+ str(i)
        with open(path, "rb") as fp:   # Unpickling
            b = pickle.load(fp)
            fp.close()    
        if fuzzy == True:
            word_repre_user = get_fuzzy_rep(b, emb_dic2, epsilon=tau)
        else:
            word_repre_user = get_sim_rep(b, emb_dic2, epsilon=tau)
        if tf == True:
            path2 =path_tf_test + '_'+ str(i)
            with open(path2, "rb") as fp:   # Unpickling
                corpus = pickle.load(fp)
                fp.close()
            for j in range(len(corpus)):
                frequency = corpus[j][0]
                word_repre_user[j] = frequency* word_repre_user[j]
        final_rep = np.sum(word_repre_user, axis=0)
        X_test2[i] = final_rep
## AUGMENTATION OF MATRIX


        


    if compress == True:
        X_test1 = np.sum(X_test1, axis=1)

        X_test2 = np.sum(X_test2, axis=1)

        if clustering == 'simple': 
            X_test1, X_test2 = add_groups(add, score1, score2, X_test1, X_test2, groups, n_col1, n_col2)          
        if clustering == 'clustering': 
            if add == 'both':
                X_test1 = clutering_addition(X_test1,groups,emb_dic1)
                X_test2 = clutering_addition(X_test2,groups,emb_dic2)
            if add == 'positive':
                X_test1 = clutering_addition(X_test1,groups,emb_dic1)
            if add == 'negative':
                X_test2 = clutering_addition(X_test2,groups,emb_dic2)
        

        X_test = np.column_stack((X_test1, X_test2))

        X_train1 = np.sum(X_train1, axis=1)
        X_train2 = np.sum(X_train2, axis=1)
        
        if clustering == 'simple': 
            
            X_train1, X_train2 = add_groups(add, score1, score2, X_train1, X_train2, groups, n_col1, n_col2) 
        if clustering == 'clustering': 
            if add == 'both':
                X_train1 = clutering_addition(X_train1,groups,emb_dic1)
                X_train2 = clutering_addition(X_train2,groups,emb_dic2)
            if add == 'positive':
                X_train1 = clutering_addition(X_train1,groups,emb_dic1)
            if add == 'negative':
                X_train2 = clutering_addition(X_train2,groups,emb_dic2)

        X_train = np.column_stack((X_train1, X_train2))
        print(X_train.shape, X_test.shape)

    else:
        if clustering == False:
            X_test1, X_test2 = add_groups(add, score1, score2, X_test1, X_test2, groups, n_col1, n_col2)            
            X_train1, X_train2 = add_groups(add, score1, score2, X_train1, X_train2, groups, n_col1, n_col2)
        else: 
            if add == 'both':
                X_test1 = clutering_addition(X_test1,groups,emb_dic1)
                X_test2 = clutering_addition(X_test2,groups,emb_dic2)
                X_train1 = clutering_addition(X_train1,groups,emb_dic1)
                X_train2 = clutering_addition(X_train2,groups,emb_dic2)
            if add == 'positive':
                X_train1 = clutering_addition(X_train1,groups,emb_dic1)
                X_test1 = clutering_addition(X_test1,groups,emb_dic1)
            if add == 'negative':
                X_train2 = clutering_addition(X_train2,groups,emb_dic2)
                X_test2 = clutering_addition(X_test2,groups,emb_dic2)
        
        X_train = np.concatenate((X_train1, X_train2), axis=1)
        X_test  = np.concatenate((X_test1, X_test2), axis=1)
        print(X_train.shape, X_test.shape)

    
    return  X_test, X_train



def run_exp_anxia_sim(num_exp, test_labels, train_labels, num_test, num_train,score1, score2,
                      chose, tau, groups, add, clustering,  dif, fuzzy, remove_stop, compress, dic, tf, w_clustering):
    print(num_exp)
    logging.basicConfig(filename="/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Fuzzy_range/w_a2.txt",level=logging.DEBUG)
    logging.debug('\n This is a message from experimet number ' + str(num_exp) )
    logging.info('\n This is a message for info')
    logging.captureWarnings(True)
    logging.warning('\n Warning message for experiment number ' + str(num_exp))


    t_initial = time()
    for_all = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Fuzzy_key_post/Matrix_user/'
    tf_path = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Fuzzy_key_post/corpus_user/'
    if dic == 1 or dic == 3 or dic == 5 or dic == 7 or dic == 9: 
        if remove_stop == False: 
            if chose == 1: 
                path_test = for_all + 'test_anxia_1/anxia_model/corpus_test3'
                path_train = for_all + 'test_anxia_1/anxia_model/corpus_train3'
            if chose == 3: 
                path_test = for_all + 'test_anxia_1/pre_model/corpus_test3'
                path_train = for_all + 'test_anxia_1/pre_model/corpus_train3'
            
            if tf == True: 
                path_tf_train = tf_path + 'train_anxia_1/corpus_train3'
                path_tf_test  = tf_path + 'test_anxia_1/corpus_test3'
        else: 
            if chose == 1: 
                path_test =  for_all +'test_anxia_1/sin_stop/anxia_model/corpus_test7'
                path_train = for_all + 'train_anxia_1/sin_stop/anxia_model/corpus_train7'
            if chose == 3:
                path_test = for_all + 'test_anxia_1/sin_stop/pre_model/corpus_test7'
                path_train = for_all + 'train_anxia_1/sin_stop/pre_model/corpus_train7'
            if tf == True: 
                path_tf_train = tf_path + 'train_anxia_1/sin_stop/corpus_train7'
                path_tf_test  = tf_path + 'test_anxia_1/sin_stop/corpus_test7'
        if dic == 1: 
            con = False
            kw1 = get_list_key(post_pos_anxia1)
            kw2 = get_list_key(post_neg_anxia1)  
            dict_str = 'Level post uppercase'      
        if dic == 3:
            con = False
            kw1 = get_list_key(user_pos_anxia1)
            kw2 = get_list_key(user_neg_anxia1) 
            dict_str = 'Level user uppercase'
        if dic == 5:
            con = True
            kw1 = get_list_key(con_pos_anxia1)
            kw2 = get_list_key(con_neg_anxia1)
            dict_str = 'Level concatenation uppercase'

        if dic == 7: 
            con = False
            kw1 = get_list_key(user_pos_anxia1)
            kw2 = get_list_key(user_neg_anxia3) 
            dict_str = 'Level user upercase v2'
        if dic == 9:
            con = True
            kw1 = get_list_key(con_pos_anxia1)
            kw2 = get_list_key(con_neg_anxia3)
            dict_str = 'Level concatenation upercase v2'            
    if dic == 2 or dic == 4 or dic == 6 or dic == 8 or dic == 10: 

        if remove_stop == False:
            if chose == 1: 
                path_test = for_all + 'test_anxia_2/anxia_model/corpus_test4'
                path_train = for_all + 'test_anxia_2/anxia_model/corpus_train4'
            if chose == 3: 
                path_test = for_all +  'test_anxia_2/pre_model/corpus_test4'
                path_train = for_all + 'test_anxia_2/pre_model/corpus_train4'
            if tf == True: 
                path_tf_train = tf_path + 'train_anxia_2/corpus_train4'
                path_tf_test  = tf_path + 'test_anxia_2/corpus_test4'
        else: 
            if chose == 1: 
                path_test = for_all + 'test_anxia_2/sin_stop/anxia_model/corpus_test8'
                path_train = for_all + 'train_anxia_2/sin_stop/anxia_model/corpus_train8'
            if chose == 3:
                path_test = for_all + 'test_anxia_2/sin_stop/pre_model/corpus_test8'
                path_train = for_all +  'train_anxia_2/sin_stop/pre_model/corpus_train8'
            if tf == True: 
                path_tf_train = tf_path + 'train_anxia_2/sin_stop/corpus_train8'
                path_tf_test  = tf_path + 'test_anxia_2/sin_stop/corpus_test8'
        if dic == 2: 
            con =  False
            kw1 = get_list_key(post_pos_anxia2)
            kw2 = get_list_key(post_neg_anxia2)
            dict_str = 'Level post lowercase'
        if dic == 4: 
            con = False
            kw1 = get_list_key(user_pos_anxia2)
            kw2 = get_list_key(user_neg_anxia2)
            dict_str = 'Level user lowercase'		        
        elif dic == 6: 
            con = True
            kw1 = get_list_key(con_pos_anxia2)
            kw2 = get_list_key(con_neg_anxia2)
            dict_str = 'Level concatenation lowercase'
        if dic == 8: 
            con = False
            kw1 = get_list_key(user_pos_anxia2)
            kw2 = get_list_key(user_neg_anxia4)
            dict_str = 'Level user lowercase v2'		        
        elif dic == 10: 
            con = True
            kw1 = get_list_key(con_pos_anxia2)
            kw2 = get_list_key(con_neg_anxia4)
            dict_str = 'Level concatenation lowercase v2'
    
    seed_val = 42
    np.random.seed(seed_val)
    parameters = {'C': [.05, .12, .25, .5, 1, 2, 4]}

    if tf == False:
        path_tf_train = ''
        path_tf_test = ''

    X_test,X_train = classificator_pos_neg(path_train, path_test, path_tf_train, path_tf_test, num_test, num_train,score1,score2,kw1,kw2, 
        			tau=tau,chose=chose,groups = groups, add= add,num_dic=dic, ex_type='anorexia', clustering = clustering, dif = dif, fuzzy = fuzzy, 
                    compress=compress, con = con, tf = tf, w_clustering= w_clustering)	
        
    if X_test.shape[1] < X_test.shape[0]:
        svr = svm.LinearSVC(class_weight='balanced', dual=False, max_iter = 8000)
        
    else: 
        svr = svm.LinearSVC(class_weight='balanced', dual=True, max_iter = 8000)
        
    grid_anorexia = GridSearchCV(estimator=svr, param_grid=parameters, n_jobs=8, scoring='f1_macro', cv=5)
    grid_anorexia.fit(X_train, train_labels)

    y_pred = grid_anorexia.predict(X_test)
    a= grid_anorexia.best_params_
    f1 = f1_score(test_labels, y_pred)
    if dif == True: 
        dif_str = 'No difference'
    if dif == False:
        dif_str = 'Difference'
    if fuzzy == True:
        fuzzy_str = 'Fuzzy'
    if fuzzy == False:
        fuzzy_str = 'Not Fuzzy'
    if remove_stop == True:
        remove_stop_str = 'Removed stopwords'
    if remove_stop == False:
        remove_stop_str = 'Not removed stopwords'
    if compress == True:
        compress_str ='Compression'
    if compress == False:
        compress_str = 'Full matrix'
    if chose == 1: 
        w_e = 'Anorexia Model'
    if chose == 3: 
        w_e = 'Pre_trained Model'
    if tf == True:
        weight = 'tf'
    else: 
        weight = 'binary'
    if w_clustering == True:
        w_clus = 'add topic'
    else:
        w_clus = 'no_topic'
    str_groups = ''
    for i in range(len(groups)):
        if i != len(groups)-1:
            str_groups += str(groups[i]) +'-'
        else:
            str_groups += str(groups[i])

    f = open('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Fuzzy_range/f1_anorexia3.txt','a')
    f.write('\n' + str(num_exp) + ',' + str(score1) + ',' + str(score2) +',' + str(tau) +',' + dif_str 
            +','+ fuzzy_str+','+ remove_stop_str +','+ compress_str+ ','+ clustering+ ',' + w_clus+',' + add + ',' + str_groups
            +','+ w_e + ','+ dict_str + ','+ weight + ',' + str(f1) + ',' + str(a)) 
    f.close()   
        
    print("done in %fs" % (time() - t_initial))
        
    return f1_score(test_labels, y_pred)

def run_exp_dep_sim(num_exp,  test_labels, train_labels,num_test,num_train, score1, score2,
                    chose, tau,add,clustering,  groups, dif, fuzzy, remove_stop, compress, dic, tf, w_clustering ):

    logging.basicConfig(filename="/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Fuzzy_range/w_dep2.txt",level=logging.DEBUG)
    logging.debug('\n This is a message from experimet number ' + str(num_exp) )
    logging.info('\n This is a message for info')
    logging.captureWarnings(True)
    logging.warning('\nWarning message for experiment number ' + str(num_exp))
    
    t_initial = time()
    for_all = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Fuzzy_key_post/Matrix_user/'
    tf_path = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Fuzzy_key_post/corpus_user/'
    if dic == 1 or dic == 3 or dic == 5 or dic == 7 or dic == 9: 
        if remove_stop == False:
            if chose == 2: 
                path_test = for_all +'test_dep_1/dep_model/corpus_test3'
                path_train = for_all + 'train_dep_1/dep_model/corpus_train3'
            if chose == 3: 
                path_test = for_all + 'test_dep_1/pre_model/corpus_test3'
                path_train = for_all + 'train_dep_1/pre_model/corpus_train3'
            if tf == True: 
                path_tf_train = tf_path + 'train_dep_1/corpus_train3'
                path_tf_test  = tf_path + 'test_dep_1/corpus_test3'
        else: 
            if chose == 2: 
                path_test = for_all + 'test_dep_1/sin_stop/dep_model/corpus_test7'
                path_train = for_all +'train_dep_1/sin_stop/dep_model/corpus_train7'
            if chose == 3:
                path_test = for_all + 'test_dep_1/sin_stop/pre_model/corpus_test7'
                path_train =for_all + 'train_dep_1/sin_stop/pre_model/corpus_train7'
            if tf == True: 
                path_tf_train = tf_path + 'train_dep_1/sin_stop/corpus_train7'
                path_tf_test  = tf_path + 'test_dep_1/sin_stop/corpus_test7'
        
        if dic == 1:
            con = False
            kw1 = get_list_key(post_pos_dep1)
            kw2 = get_list_key(post_neg_dep1)
            dict_str = 'Level post upercase'  
        if dic == 3: 
            con = False
            kw1 = get_list_key(user_pos_dep1)
            kw2 = get_list_key(user_neg_dep1) 
            dict_str = 'Level user upercase'  
        if dic == 5: 
            con = True
            kw1 = get_list_key(con_pos_dep1)
            kw2 = get_list_key(con_neg_dep1)
            dict_str = 'Level concatenation upercase'  
        if dic == 7: 
            con = False
            kw1 = get_list_key(user_pos_dep1)
            kw2 = get_list_key(user_neg_dep3) 
            dict_str = 'Level user upercase v2'  
        if dic == 9: 
            con = True
            kw1 = get_list_key(con_pos_dep1)
            kw2 = get_list_key(con_neg_dep4)
            dict_str = 'Level concatenation upercase v2'  
    if dic == 2 or dic == 4 or dic == 6 or dic == 8 or dic == 10:         
        if remove_stop == False:
            if chose == 2: 
                path_test = for_all + 'test_dep_2/dep_model/corpus_test4'
                path_train = for_all + 'train_dep_2/dep_model/corpus_train4'
            if tf == True: 
                path_tf_train = tf_path + 'train_dep_2/corpus_train4'
                path_tf_test  = tf_path + 'test_dep_2/corpus_test4'
            if chose == 3: 
                path_test = for_all + 'test_dep_2/pre_model/corpus_test4'
                path_train = for_all +  'train_dep_2/pre_model/corpus_train4'
        else: 
            if chose == 2: 
                path_test = for_all + 'test_dep_2/sin_stop/dep_model/corpus_test8'
                path_train = for_all + 'train_dep_2/sin_stop/dep_model/corpus_train8'
            if chose == 3:
                path_test = for_all + 'test_dep_2/sin_stop/pre_model/corpus_test8'
                path_train =for_all + 'train_dep_2/sin_stop/pre_model/corpus_train8'
            if tf == True: 
                path_tf_train = tf_path + 'train_dep_2/sin_stop/corpus_train8'
                path_tf_test  = tf_path + 'test_dep_2/sin_stop/corpus_test8'
        if dic == 2: 
            con = False
            kw1 = get_list_key(post_pos_dep2)
            kw2 = get_list_key(post_neg_dep2)
            dict_str = 'Level post lowercase'  
        if dic == 4: 
            con = False
            kw1 = get_list_key(user_pos_dep2)
            kw2 = get_list_key(user_neg_dep2)
            dict_str = 'Level user lowercase'  
        if  dic == 6: 
            con = True
            kw1 = get_list_key(con_pos_dep2)
            kw2 = get_list_key(con_neg_dep2)
            dict_str = 'Level concatenation lowercase'    
        if dic == 8: 
            con = False
            kw1 = get_list_key(user_pos_dep2)
            kw2 = get_list_key(user_neg_dep3)
            dict_str = 'Level user lowercase v2'  
        if  dic == 10: 
            con = True
            kw1 = get_list_key(con_pos_dep2)
            kw2 = get_list_key(con_neg_dep4)
            dict_str = 'Level concatenation lowercase v2'        
    
    seed_val = 42
    np.random.seed(seed_val)
    parameters = {'C': [.05, .12, .25, .5, 1, 2, 4]}
    if tf == False:
        path_tf_train = ''
        path_tf_test = ''
    if dif == True: 
        dif_str = 'No difference'
    if dif == False:
        dif_str = 'Difference'
    if fuzzy == True:
        fuzzy_str = 'Fuzzy'
    if fuzzy == False:
        fuzzy_str = 'Not Fuzzy'
    if remove_stop == True:
        remove_stop_str = 'Removed stopwords'
    if remove_stop == False:
        remove_stop_str = 'Not removed stopwords'
    if compress == True:
        compress_str ='Compression'
    if compress == False:
        compress_str = 'Full matrix'
    if chose == 2: 
        w_e = 'Depression Model'
    if chose == 3: 
        w_e = 'Pre_trained Model'
    if tf == True:
        weight = 'tf'
    else: 
        weight = 'binary'
    if clustering == 'None':
        clus = 'None'
    if clustering == True:
        clus = 'Clustering'
    else:
        clus = 'Simple_compression'

    str_groups = ''
    for i in range(len(groups)):
        if i != len(groups)-1:
            str_groups += str(groups[i]) +'-'
        else:
            str_groups += str(groups[i])

    X_test,X_train = classificator_pos_neg(path_train, path_test,path_tf_train, path_tf_test,  num_test, num_train,score1,score2,kw1,kw2, 
        			tau=tau,chose=chose,add = add, num_dic=dic, ex_type='depression', clustering=clustering, groups = groups,dif = dif, 
                    fuzzy = fuzzy, compress = compress, con= con, tf = tf, w_clustering= w_clustering)	
        
    if X_test.shape[1] < X_test.shape[0]:
        svr = svm.LinearSVC(class_weight='balanced', dual=False, max_iter = 6000)
        
    else: 
        svr = svm.LinearSVC(class_weight='balanced', dual=True, max_iter = 6000)
        
    grid_dep = GridSearchCV(estimator=svr, param_grid=parameters, n_jobs=8, scoring='f1_macro', cv=5)
    grid_dep.fit(X_train, train_labels)

    y_pred = grid_dep.predict(X_test)
    a= grid_dep.best_params_
    f1 = f1_score(test_labels, y_pred)
    f = open('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Fuzzy_range/f1_dep3.txt','a')
    f.write('\n' + str(num_exp) + ',' + str(score1) + ',' + str(score2) +',' + str(tau) +',' + dif_str 
            +','+ fuzzy_str+','+ remove_stop_str +','+ compress_str+ ','+ clus +',' +add + ','+ str_groups+ ',' + w_e + ','+ dict_str + ','+ weight + ',' + str(f1) + ',' + str(a)) 
    f.close()       

    print('The time for this experiment was %fs' % (time() - t_initial))

    return f1
        



