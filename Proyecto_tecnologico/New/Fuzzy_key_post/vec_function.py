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


from text_functions import (get_text_labels,
                            get_text_test)
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


def get_fuzzy_rep(words_user, dictionary, option, epsilon):

    dictionary_vec = np.zeros((len(set(dictionary)), 300), dtype=float)     
    for i in range(dictionary_vec.shape[0]):
        w1 = dictionary[i]
        if option == 1:
            dictionary_vec[i] = model_anxia.wv[w1]
        if option == 2:
            dictionary_vec[i] = model_dep.wv[w1]
        if option == 3: 
            dictionary_vec[i] = model_pre.get_word_vector(w1)
    similarity_vocab = sklearn.metrics.pairwise.cosine_similarity(words_user, dictionary_vec)
    # vector de representación
    vec_representation = np.count_nonzero(similarity_vocab > epsilon, axis=0)
    return vec_representation

def get_sim_rep(words_user, dictionary,option, epsilon):

    dictionary_vec = np.zeros((len(set(dictionary)),300) ,dtype=float)
     
    for i in range(dictionary_vec.shape[0]):
        w1 = dictionary[i]
        if option == 1:
            dictionary_vec[i] = model_anxia.wv[w1]
        if option == 2:
            dictionary_vec[i] = model_dep.wv[w1]
        if option == 3: 
            dictionary_vec[i] = model_pre.get_word_vector(w1)
            
    similarity_vocab = sklearn.metrics.pairwise.cosine_similarity(words_user, dictionary_vec)


    vec_representation = np.sum( similarity_vocab, axis=0, where= similarity_vocab > epsilon)
     
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
    t_initial = time()
    # we upload the keywords
    #d1 = get_list_key(k1)
    #d2 = get_list_key(k2)

    # sort by the score
    #d1.sort(key=lambda y: y[1])
    #d2.sort(key=lambda y: y[1])

    if no_distintc == True:
        # obtain the
        kw1 = d1[:feature1]
        kw2 = d2[:feature2]

        l1 = get_words_from_kw(kw1)
        l2 = get_words_from_kw(kw2)

        return l1, l2

    else:
        # select by the score, when the score is low the importance is greater
        #kw1 = [x for x in d1 if x[1] > score1]
        #kw2 = [x for x in d2 if x[1] > score2]

        dictionary1 = []
        dictionary2 = []

        # get only the words
        #l1 = get_words_from_kw(kw1)
        #l2 = get_words_from_kw(kw2)
        l1 = get_words_from_kw(d1)
        l2 = get_words_from_kw(d2)

        #len1 = len(l1)
        #len2 = len(l2)
        #if len1 < len2:
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

        '''
        else:
            for i in range(len2):
                w2 = l2[i]
                # revisamos si no está en el diccionario positivo
                if (w2 in l1) == False:
                    dictionary2.append(w2)
                # en caso de que lo esté revisamos cual tiene score más pequeño
                else:
                    # en donde se encuentra en la lista l1
                    indice = l1.index(w2)
                    # score en la lista 1
                    rel1 = kw1[indice][1]
                    # score en la lista 2
                    rel2 = kw2[i][1]
                    # si el score de la lista 1 es menor que en la 1 la dejamos en lista 1
                    if rel2 > rel1:
                        dictionary2.append(w2)
            # dictionario positivo son todas las palabras que no están en dictionario negativo
            dictionary1 = [x for x in l1 if x not in dictionary2]
        '''
    #print("make dictionaries done in %fs" % (time() - t_initial))
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

#USER VERSION OF THE DICTIONARIES OF THE USERS
#---------------------------ANOREXIA-------------------#
#version 1
user_pos_anxia1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/anxia_pos_ver3'
user_neg_anxia1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/anxia_neg_ver3'
#version 2
user_pos_anxia2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/anxia_pos_ver4'
user_neg_anxia2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/anxia_neg_ver4'

#---------------------------DEPRESSION-----------------#
#version 1
user_pos_dep1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_pos_ver3'
user_neg_dep1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_neg_ver3'
#version 2
user_pos_dep2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_pos_ver4'
user_neg_dep2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_neg_ver4'

#CONCATANTE ALL THE TEXT 
#version 1
con_pos_anxia1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_pos_ver3'
con_neg_anxia1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_neg_ver3'
#version 2
con_pos_anxia2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_pos_ver4'
con_neg_anxia2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_neg_ver4'

#---------------------------DEPRESSION-----------------#
#version 1
con_pos_dep1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_pos_ver3'
con_neg_dep1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_neg_ver3'
#version 2
con_pos_dep2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_pos_ver4'
con_neg_dep2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_neg_ver4'




# function classificator
def classificator_pos_neg(all_path_train, all_path_test, num_test, num_train, score1, score2, di1, di2, tau, chose, dif=True,
                          fuzzy=True, compress=True, con = False):
    # Parameters:
    # pos_data: positive data from training data
    # neg_data: negative data fromo training data
    # test : test data for the constructions of the test matrix for the classification
    # score1, score 2: this select the words from the dictionaries
    # tau : tolerance for the similarity between the words
    # remove_stop: remove the stop words from the text
    # train_data:  if True, construct the matrix of training for the SVM classificator
    # compress : if True. compress the matrix for the svm to a two dimensional array
    # dif:  the positive dictioinary and the negative dictionar does not have words in common
    # di1,di2 : the positive and negative dictionary
    
    dictionary1, dictionary2 = dict_scores(
        di1, di2, score1, score2, no_distintc=dif, con=con)
    X_test1 = np.zeros((num_test, len(dictionary1)),
                       dtype=float)  # matriz tipo document-term
    X_test2 = np.zeros((num_test, len(dictionary2)), dtype=float)


   ### For using svm ####
    X_train1 = np.zeros((num_train, len(dictionary1)), dtype=float)
    X_train2 = np.zeros((num_train, len(dictionary2)), dtype=float)
    
    for i in range(num_train):
        path =all_path_train + '_'+ str(i)
        with open(path, "rb") as fp:   # Unpickling
            b = pickle.load(fp)
            fp.close()

        if fuzzy == True:
            word_repre_user = get_fuzzy_rep(
            b, dictionary1, option=chose, epsilon=tau)
        else:
            word_repre_user = get_sim_rep(
            b, dictionary1, option=chose, epsilon=tau)
        X_train1[i] = word_repre_user
    #print("termina construcción de train data pos in %fs" % (time() - t_initial1))
    t_initial2 = time()
    for i in range(num_train):
        path =all_path_train + '_'+ str(i)
        with open(path, "rb") as fp:   # Unpickling
            b = pickle.load(fp)
            fp.close()        
        if fuzzy == True:
            word_repre_user = get_fuzzy_rep(
                b, dictionary2, option=chose, epsilon=tau)
        else:
            word_repre_user = get_sim_rep(
                b, dictionary2, option=chose, epsilon=tau)
        X_train2[i] = word_repre_user
    #print("termina construcción de train data neg in %fs" % (time() - t_initial2))
    if compress == True:
        X_train1 = np.sum(X_train1, axis=1)
        X_train2 = np.sum(X_train2, axis=1)

    # - Construct test representation of positive users
    
    for i in range(num_test):
        path =all_path_test + '_'+ str(i)
        with open(path, "rb") as fp:   # Unpickling
            b = pickle.load(fp)
            fp.close()    
        if fuzzy == True:
            word_repre_user = get_fuzzy_rep(
                    b, dictionary1, option=chose, epsilon=tau)
        else:
            word_repre_user = get_sim_rep(
                    b, dictionary1, option=chose, epsilon=tau)
        X_test1[i] = word_repre_user


    # Construct test representation of negative users
    for i in range(num_test):
        path =all_path_test + '_'+ str(i)
        with open(path, "rb") as fp:   # Unpickling
            b = pickle.load(fp)
            fp.close()    
        if fuzzy == True:
            word_repre_user = get_fuzzy_rep(
                b, dictionary2, option=chose, epsilon=tau)
        else:
            word_repre_user = get_sim_rep(
                b, dictionary2, option=chose, epsilon=tau)
        X_test2[i] = word_repre_user



    if compress == True:
        X_test1 = np.sum(X_test1, axis=1)
        X_test2 = np.sum(X_test2, axis=1)
        #X_train = np.concatenate((X_train1, X_train2), axis = 1)
        #results = np.concatenate((X_test1, X_test2), axis = 1)
        X_test = np.concatenate((X_test1, X_test2)).reshape((-1, 2), order='F')
        X_train = np.concatenate((X_train1, X_train2)).reshape((-1, 2), order='F')
        print(X_train.shape, X_test.shape)
        return X_test, X_train, len(dictionary1), len(dictionary2)

    else:
        X_train = np.concatenate((X_train1, X_train2), axis=1)
        X_test  = np.concatenate((X_test1, X_test2), axis=1)
        print(X_train.shape, X_test.shape)

        return  X_test, X_train, len(dictionary1), len(dictionary2)



def run_exp_anxia_sim(num_exp, test_labels, train_labels, num_test, num_train,score1, score2,
                      chose, tau, dif, fuzzy, remove_stop, compress, dic):
    print(num_exp)
    logging.basicConfig(filename="/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Fuzzy_key_post/warnings_anorexia.txt",level=logging.DEBUG)
    logging.debug('\n This is a message from experimet number ' + str(num_exp) )
    logging.info('\n This is a message for info')
    logging.captureWarnings(True)
    logging.warning('\n Warning message for experiment number ' + str(num_exp))


    t_initial = time()
    for_all = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Fuzzy_key_post/Matrix_user/'
    if dic == 1 or dic == 3 or dic == 5: 
        if remove_stop == False: 
            if chose == 1: 
                path_test = for_all + 'test_anxia_1/anxia_model/corpus_test3'
                path_train = for_all + 'test_anxia_1/anxia_model/corpus_train3'
            if chose == 3: 
                path_test = for_all + 'test_anxia_1/pre_model/corpus_test3'
                path_train = for_all + 'test_anxia_1/pre_model/corpus_train3'
        else: 
            if chose == 1: 
                path_test =  for_all +'test_anxia_1/sin_stop/anxia_model/corpus_test7'
                path_train = for_all + 'train_anxia_1/sin_stop/anxia_model/corpus_train7'
            if chose == 3:
                path_test = for_all + 'test_anxia_1/sin_stop/pre_model/corpus_test7'
                path_train = for_all + 'train_anxia_1/sin_stop/pre_model/corpus_train7'
        if dic == 1: 
            con = False
            kw1 = get_list_key(post_pos_anxia1)
            kw2 = get_list_key(post_neg_anxia1)            
        if dic == 3:
            con = False
            kw1 = get_list_key(user_pos_anxia1)
            kw2 = get_list_key(user_neg_anxia1) 
        if dic == 5:
            con = True
            kw1 = get_list_key(con_pos_anxia1)
            kw2 = get_list_key(con_neg_anxia1)

    if dic == 2 or dic == 4 or dic == 6: 

        if remove_stop == False:
            if chose == 1: 
                path_test = for_all + 'test_anxia_2/anxia_model/corpus_test4'
                path_train = for_all + 'test_anxia_2/anxia_model/corpus_train4'
            if chose == 3: 
                path_test = for_all +  'test_anxia_2/pre_model/corpus_test4'
                path_train = for_all + 'test_anxia_2/pre_model/corpus_train4'
        else: 
            if chose == 1: 
                path_test = for_all + 'test_anxia_2/sin_stop/anxia_model/corpus_test8'
                path_train = for_all + 'train_anxia_2/sin_stop/anxia_model/corpus_train8'
            if chose == 3:
                path_test = for_all + 'test_anxia_2/sin_stop/pre_model/corpus_test8'
                path_train = for_all +  'train_anxia_2/sin_stop/pre_model/corpus_train8'
        if dic == 2: 
            con =  False
            kw1 = get_list_key(post_pos_anxia2)
            kw2 = get_list_key(post_neg_anxia2)
        if dic == 4: 
            con = False
            kw1 = get_list_key(user_pos_anxia2)
            kw2 = get_list_key(user_neg_anxia2)
    		        
        elif dic == 6: 
            con = True
            kw1 = get_list_key(con_pos_anxia2)
            kw2 = get_list_key(con_neg_anxia2)

    seed_val = 42
    np.random.seed(seed_val)
    parameters = {'C': [.05, .12, .25, .5, 1, 2, 4]}
        
    X_test,X_train,len_dic1,len_dic2 = classificator_pos_neg(path_train, path_test, num_test, num_train,score1,score2,kw1,kw2, 
        			tau=tau,chose=chose,dif = dif, fuzzy = fuzzy, compress=compress, con = con)	
        
    if X_test.shape[1] < X_test.shape[0]:
        svr = svm.LinearSVC(class_weight='balanced', dual=False, max_iter = 2000)
        
    else: 
        svr = svm.LinearSVC(class_weight='balanced', dual=True, max_iter = 2000)
        
    grid_anorexia = GridSearchCV(estimator=svr, param_grid=parameters, n_jobs=8, scoring='f1_macro', cv=5)
    grid_anorexia.fit(X_train, train_labels)

    y_pred = grid_anorexia.predict(X_test)
    a= grid_anorexia.best_params_
    f1 = f1_score(test_labels, y_pred)
    f = open('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Fuzzy_key_post/f1_anorexia2.txt','a')
    f.write('\n' + str(num_exp) + ',' + str(score1) + ',' + str(score2) +',' + str(tau) +',' + str(dif) 
            +','+ str(fuzzy) +','+ str(remove_stop) +','+ str(compress)+  ',' + str(chose) + ','+ str(dic) + ',' + str(f1) + ',' + str(a)) 
    f.close()   
        
    print("done in %fs" % (time() - t_initial))
        
    return f1_score(test_labels, y_pred)

def run_exp_dep_sim(num_exp,  test_labels, train_labels,num_test,num_train, score1, score2,
                    chose, tau, dif, fuzzy, remove_stop, compress, dic):

    logging.basicConfig(filename="/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Fuzzy_key_post/warnings_dep.txt",level=logging.DEBUG)
    logging.debug('\n This is a message from experimet number ' + str(num_exp) )
    logging.info('\n This is a message for info')
    logging.captureWarnings(True)
    logging.warning('\nWarning message for experiment number ' + str(num_exp))
    
    t_initial = time()
    for_all = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Fuzzy_key_post/Matrix_user/'
    if dic == 1 or dic == 3 or dic == 5: 
        if remove_stop == False:
            if chose == 2: 
                path_test = for_all +'test_dep_1/dep_model/corpus_test3'
                path_train = for_all + 'train_dep_1/dep_model/corpus_train3'
            if chose == 3: 
                path_test = for_all + 'test_dep_1/pre_model/corpus_test3'
                path_train = for_all + 'train_dep_1/pre_model/corpus_train3'
        else: 
            if chose == 2: 
                path_test = for_all + 'test_dep_1/sin_stop/dep_model/corpus_test7'
                path_train = for_all +'train_dep_1/sin_stop/dep_model/corpus_train7'
            if chose == 3:
                path_test = for_all + 'test_dep_1/sin_stop/pre_model/corpus_test7'
                path_train =for_all + 'train_dep_1/sin_stop/pre_model/corpus_train7'
        
        if dic == 1:
            con = False
            kw1 = get_list_key(post_pos_dep1)
            kw2 = get_list_key(post_neg_dep1)
        if dic == 3: 
            con = False
            kw1 = get_list_key(user_pos_dep1)
            kw2 = get_list_key(user_neg_dep1) 
        if dic == 5: 
            con = True
            kw1 = get_list_key(con_pos_dep1)
            kw2 = get_list_key(con_neg_dep1)

    if dic == 2 or dic == 4 or dic == 6:         
        if remove_stop == False:
            if chose == 2: 
                path_test = for_all + 'test_dep_2/dep_model/corpus_test4'
                path_train = for_all + 'train_dep_2/dep_model/corpus_train4'
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
        if dic == 2: 
            con = False
            kw1 = get_list_key(post_pos_dep2)
            kw2 = get_list_key(post_neg_dep2)

        if dic == 4: 
            con = False
            kw1 = get_list_key(user_pos_dep2)
            kw2 = get_list_key(user_neg_dep2)
        if  dic == 6: 
            con = True
            kw1 = get_list_key(con_pos_dep2)
            kw2 = get_list_key(con_neg_dep2)
      
    
    seed_val = 42
    np.random.seed(seed_val)
    parameters = {'C': [.05, .12, .25, .5, 1, 2, 4]}
        
    X_test,X_train,len_dic1,len_dic2 = classificator_pos_neg(path_train, path_test,num_test, num_train,score1,score2,kw1,kw2, 
        			tau=tau,chose=chose,dif = dif, fuzzy = fuzzy, compress = compress, con= con)	
        
    if X_test.shape[1] < X_test.shape[0]:
        svr = svm.LinearSVC(class_weight='balanced', dual=False, max_iter = 6000)
        
    else: 
        svr = svm.LinearSVC(class_weight='balanced', dual=True, max_iter = 6000)
        
    grid_dep = GridSearchCV(estimator=svr, param_grid=parameters, n_jobs=8, scoring='f1_macro', cv=5)
    grid_dep.fit(X_train, train_labels)

    y_pred = grid_dep.predict(X_test)
    a= grid_dep.best_params_
    f1 = f1_score(test_labels, y_pred)
    f = open('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Fuzzy_key_post/f1_dep2.txt','a')
    f.write('\n' + str(num_exp) + ',' + str(score1) + ',' + str(score2) +',' + str(tau) +',' + str(dif) 
            +','+ str(fuzzy) +','+ str(remove_stop) +','+ str(compress)+  ',' + str(chose) + ','+ str(dic) + ',' + str(f1) + ',' + str(a)) 
    f.close()       

    print('The time for this experiment was %fs' % (time() - t_initial))

    return f1,a, len_dic1, len_dic2
        



