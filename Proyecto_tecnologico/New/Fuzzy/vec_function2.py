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


def get_fuzzy_rep(words_doc, dictionary, option, epsilon):
    words_user = np.zeros((len(words_doc), 300), dtype=float)
    dictionary_vec = np.zeros((len(set(dictionary)), 300), dtype=float)

    for i in range(words_user.shape[0]):
        w1 = words_doc[i]
        if option == 1:
            words_user[i] = model_anxia.wv[w1]
        if option == 2:
            words_user[i] = model_dep.wv[w1]
        if option == 3:
            words_user[i] = model_pre.get_word_vector(w1)
    for i in range(dictionary_vec.shape[0]):
        w1 = dictionary[i]
        if option == 1:
            dictionary_vec[i] = model_anxia.wv[w1]
        if option == 2:
            dictionary_vec[i] = model_dep.wv[w1]
        if option == 3:
            dictionary_vec[i] = model_pre.get_word_vector(w1)
    similarity_vocab = sklearn.metrics.pairwise.cosine_similarity(
        words_user, dictionary_vec)
    # change the similarities that are lower than the threshold for 0
    similarity_rep = np.where(similarity_vocab < epsilon, 0, similarity_vocab)
    #change elements that are greater than epsilon for 1 
    similarity_rep = np.where(similarity_rep >= epsilon, 1, similarity_rep)
    

    return similarity_rep

def get_sim_rep(words_doc, dictionary, option, epsilon):
    words_user = np.zeros((len(words_doc), 300), dtype=float)
    dictionary_vec = np.zeros((len(set(dictionary)), 300), dtype=float)

    for i in range(words_user.shape[0]):
        w1 = words_doc[i]
        if option == 1:
            words_user[i] = model_anxia.wv[w1]
        if option == 2:
            words_user[i] = model_dep.wv[w1]
        if option == 3:
            words_user[i] = model_pre.get_word_vector(w1)
    for i in range(dictionary_vec.shape[0]):
        w1 = dictionary[i]
        if option == 1:
            dictionary_vec[i] = model_anxia.wv[w1]
        if option == 2:
            dictionary_vec[i] = model_dep.wv[w1]
        if option == 3:
            dictionary_vec[i] = model_pre.get_word_vector(w1)

    similarity_vocab = sklearn.metrics.pairwise.cosine_similarity(
        words_user, dictionary_vec)

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


def make_vec_rep(index, dict_data, dict_words, user_corpus, freq_words,  feature_dic,fuzzy, tf, tau, word_embedding):
    #index for the loop
    #dict_data is the directory with the embedings
    #dict_words contains the words from all the corpus in the dataset 
    #user corpus contains the words used by the given user (list)
    # freq_words contains the frequency of the words (dictionary) 
    #feature_dic about the type of dictionary 
    #tf weight 
    #tau tolarance for the similarity 
    #model for word embedding
    if index == 0: 
        dict_words = user_corpus
        if fuzzy == True:
            word_repre_user = get_fuzzy_rep(user_corpus, feature_dic, option=word_embedding, epsilon=tau)
            dict_data = {user_corpus[i] : word_repre_user[i] for i in range(len(user_corpus))}
            if tf == True:
                for j in range(len(user_corpus)):
                    frequency = freq_words[j][0]
                    word_repre_user[j] = frequency * word_repre_user[j]
                word_repre_user = np.sum(word_repre_user, axis = 0)
            else:
                word_repre_user = np.sum(word_repre_user,axis = 0) 
        else: 
            word_repre_user = get_sim_rep(user_corpus, feature_dic, option=word_embedding, epsilon=tau)
            dict_data = {user_corpus[i] : word_repre_user[i] for i in range(len(user_corpus))}
            if tf == True:
                for j in range(len(user_corpus)):
                    frequency = freq_words[j][0]
                    word_repre_user[j] = frequency * word_repre_user[j]
                word_repre_user = np.sum(word_repre_user, axis = 0)
            else:
                word_repre_user = np.sum(word_repre_user,axis = 0)    
    if index > 0:
        words_in = [x for x in user_corpus if x in dict_words]
        words_not = [x for x in user_corpus if x not in dict_words]       

        if fuzzy == True:
            #only calculate the words not in dict_words
            word_repre_user = get_fuzzy_rep(words_not, feature_dic, option = word_embedding, epsilon=tau)
            #update the dictionaries with tensors 
            dict_data.update({words_not[j]:  word_repre_user[j] for j in range(len(words_not))}) 
            #now update the dictionary of words
            dict_words = dict_words + words_not
            if tf == True:
                #now we add the term frequency 
                for j in range(len(words_not)):
                    word = words_not[j]        
                    #original index in the user corpus 
                    index_word = user_corpus.index(word)
                    #frequency word
                    frequency = freq_words[index_word][0]
                    word_repre_user[j] = frequency * word_repre_user[j]
                #now the words that are alredy calculated 
                for j in range(len(words_in)):
                    word = words_in[j]
                    index_word = user_corpus.index(word)
                    frequency = freq_words[index_word][0]
                    #vector calculated previously 
                    vec_cal = dict_data[word]
                    tf_vec_cal = frequency*vec_cal
                    #add the row 
                    word_repre_user = np.row_stack((word_repre_user, tf_vec_cal))
                word_repre_user = np.sum(word_repre_user, axis = 0)
            else:
                for j in range(len(words_in)):
                    word = words_in[j]
                    index_word = user_corpus.index(word)
                    frequency = freq_words[index_word][0]
                    #vector calculated previously 
                    vec_cal = dict_data[word]
                    word_repre_user = np.row_stack((word_repre_user, vec_cal))
                word_repre_user = np.sum(word_repre_user,axis = 0)                 
        else: 
            #only calculate the words not in dict_words
            word_repre_user = get_sim_rep(words_not, feature_dic, option = word_embedding, epsilon=tau)
            #update the dictionaries with tensors 
            dict_data.update({words_not[j]:  word_repre_user[j] for j in range(len(words_not))}) 
            #now update the dictionary of words
            dict_words = dict_words + words_not
            if tf == True:
                #now we add the term frequency 
                for j in range(len(words_not)):
                    word = words_not[j]        
                    #original index in the user corpus 
                    index_word = user_corpus.index(word)
                    #frequency word
                    frequency = freq_words[index_word][0]
                    word_repre_user[j] = frequency * word_repre_user[j]
                #now the words that are alredy calculated 
                for j in range(len(words_in)):
                    word = words_in[j]
                    index_word = user_corpus.index(word)
                    frequency = freq_words[index_word][0]
                    #vector calculated previously 
                    vec_cal = dict_data[word]
                    tf_vec_cal = frequency*vec_cal
                    #add the row 
                    word_repre_user = np.row_stack((word_repre_user, tf_vec_cal))
                word_repre_user = np.sum(word_repre_user, axis = 0)
            else:
                for j in range(len(words_in)):
                    word = words_in[j]
                    index_word = user_corpus.index(word)
                    frequency = freq_words[index_word][0]
                    #vector calculated previously 
                    vec_cal = dict_data[word]
                    word_repre_user = np.row_stack((word_repre_user, vec_cal))
                word_repre_user = np.sum(word_repre_user,axis = 0)
    return word_repre_user, dict_data, dict_words

# function classificator
def classificator_pos_neg(pos_data, neg_data, test, score1, score2, di1, di2, tau, word_emb, dif,
                          fuzzy, remove_stop, compress, con, cap, tf):
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


    if remove_stop == True:
        print("Quitando stopwords")
        pos_data = remove_stop_list(pos_data)
        neg_data = remove_stop_list(neg_data)
        test = remove_stop_list(test)
        print('End')
    pos_data = normalize(pos_data)
    neg_data = normalize(neg_data)
    test = normalize(test)

    num_test = len(test)

    dictionary1, dictionary2 = dict_scores(
        di1, di2, score1, score2, no_distintc=dif, con=con)
    X_test1 = np.zeros((num_test, len(dictionary1)),
                       dtype=float)  # matriz tipo document-term
    X_test2 = np.zeros((num_test, len(dictionary2)), dtype=float)
    num_train = len(pos_data) + len(neg_data)

   ### For using svm ####
    train = [*pos_data, *neg_data]
    X_train1 = np.zeros((num_train, len(dictionary1)), dtype=float)
    X_train2 = np.zeros((num_train, len(dictionary2)), dtype=float)
    
    words_train = []
    dict_train_pos = dict()
    for i in range(num_train):
        doc = train[i]
        #if cap == True then we don't lower the text 
        if cap == True:
            corpus_palabras = tokenizer.tokenize(doc)
        else:
            corpus_palabras = tokenizer.tokenize(doc.lower())
        fdist = nltk.FreqDist(corpus_palabras)
        v = sortFreqDict(fdist)
        words_doc = get_words(v)
        word_repre_user, dict_train_pos, words_train = make_vec_rep(i,dict_train_pos,words_train,
                                                                    words_doc,v,dictionary1,fuzzy=fuzzy,
                                                                    tf = tf,tau = tau, word_embedding= word_emb) 
        X_train1[i] = word_repre_user
    
    words_train_neg = []
    dict_train_neg = dict()
    for i in range(num_train):
        doc = train[i]
        #if cap == True then we don't lower the text 
        if cap == True:
            corpus_palabras = tokenizer.tokenize(doc)
        else:
            corpus_palabras = tokenizer.tokenize(doc.lower())
        fdist = nltk.FreqDist(corpus_palabras)
        v = sortFreqDict(fdist)
        words_doc = get_words(v)
        word_repre_user, dict_train_neg, words_train_neg = make_vec_rep(i,dict_train_neg,words_train_neg,
                                                                    words_doc,v,dictionary2,fuzzy=fuzzy,
                                                                    tf = tf,tau = tau, word_embedding= word_emb) 

        X_train2[i] = word_repre_user
    #print("termina construcción de train data neg in %fs" % (time() - t_initial2))
    if compress == True:
        X_train1 = np.sum(X_train1, axis=1)
        X_train2 = np.sum(X_train2, axis=1)

    # - Construct test representation of positive users
    dict_test_pos = dict()
    words_test_pos = []
    for i in range(num_test):
        doc = test[i]
        #if cap == True then we don't lower the text 
        if cap == True:
            corpus_palabras = tokenizer.tokenize(doc)
        else:
            corpus_palabras = tokenizer.tokenize(doc.lower())
        fdist = nltk.FreqDist(corpus_palabras)
        v = sortFreqDict(fdist)
        words_doc = get_words(v)
        word_repre_user, dict_test_pos, words_test_pos = make_vec_rep(i,dict_test_pos,words_test_pos,
                                                                    words_doc,v,dictionary1,fuzzy=fuzzy,
                                                                    tf = tf,tau = tau, word_embedding= word_emb) 
        X_test1[i] = word_repre_user

    dict_test_neg = dict()
    words_test_neg = []
    # Construct test representation of negative users
    for i in range(num_test):
        doc = test[i]
        #if cap == True then we don't lower the text 
        if cap == True:
            corpus_palabras = tokenizer.tokenize(doc)
        else:
            corpus_palabras = tokenizer.tokenize(doc.lower())
        fdist = nltk.FreqDist(corpus_palabras)
        v = sortFreqDict(fdist)
        words_doc = get_words(v)
        word_repre_user, dict_test_neg, words_test_neg = make_vec_rep(i,dict_test_neg,words_test_neg,
                                                                    words_doc,v,dictionary2,fuzzy=fuzzy,
                                                                    tf = tf,tau = tau, word_embedding= word_emb) 
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



def run_exp_anxia_sim(num_exp, pos_data, neg_data, test, test_labels, train_labels,score1, score2,
                      chose, tau, dif, tf, fuzzy, remove_stop, compress, dic):
    print(num_exp)
    logging.basicConfig(filename="/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Fuzzy/w_a.txt",level=logging.DEBUG)
    logging.debug('\n This is a message from experimet number ' + str(num_exp) )
    logging.info('\n This is a message for info')
    logging.captureWarnings(True)
    logging.warning('\n Warning message for experiment number ' + str(num_exp))


    t_initial = time()

    if dic == 1: 
        con = False
        cap = True
        kw1 = get_list_key(post_pos_anxia1)
        kw2 = get_list_key(post_neg_anxia1)   
        dict_str = 'Level Post Upercase'

    if dic == 3:
        con = False
        cap = True
        kw1 = get_list_key(user_pos_anxia1)
        kw2 = get_list_key(user_neg_anxia1) 
        dict_str = 'Level User Upercase'
    if dic == 5:
        con = True
        cap = True
        kw1 = get_list_key(con_pos_anxia1)
        kw2 = get_list_key(con_neg_anxia1)
        dict_str = 'Level Concatenation Upercase'
    if dic == 2: 
        con =  False
        cap = False
        kw1 = get_list_key(post_pos_anxia2)
        kw2 = get_list_key(post_neg_anxia2)
        dict_str = 'Level post lowercase'
    if dic == 4: 
        con = False
        cap = False
        kw1 = get_list_key(user_pos_anxia2)
        kw2 = get_list_key(user_neg_anxia2)
        dict_str = 'Level user lowercase'
    elif dic == 6: 
        con = True
        cap = False
        kw1 = get_list_key(con_pos_anxia2)
        kw2 = get_list_key(con_neg_anxia2)
        dict_str = 'Level concatenation lowercase'
    seed_val = 42
    np.random.seed(seed_val)
    parameters = {'C': [.05, .12, .25, .5, 1, 2, 4]}
        
    X_test,X_train,len_dic1,len_dic2 = classificator_pos_neg(pos_data, neg_data,test, score1,score2,kw1,kw2, 
        			tau=tau,word_emb=chose,dif = dif, fuzzy = fuzzy, remove_stop=remove_stop,  compress=compress, con = con,cap = cap, tf = tf)	
        
    if X_test.shape[1] < X_test.shape[0]:
        svr = svm.LinearSVC(class_weight='balanced', dual=False, max_iter = 2000)
        
    else: 
        svr = svm.LinearSVC(class_weight='balanced', dual=True, max_iter = 2000)
        
    grid_anorexia = GridSearchCV(estimator=svr, param_grid=parameters, n_jobs=8, scoring='f1_macro', cv=5)
    grid_anorexia.fit(X_train, train_labels)

    y_pred = grid_anorexia.predict(X_test)
    a= grid_anorexia.best_params_
    f1 = f1_score(test_labels, y_pred)
    f = open('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Fuzzy/f1_anorexia.txt','a')

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
    

    f.write('\n' + str(num_exp) + ',' + str(score1) + ',' + str(score2) +',' + str(tau) +',' + dif_str 
            +','+ fuzzy_str+','+ remove_stop_str +','+ compress_str+  ',' + w_e + ','+ dict_str + ',' + str(f1) + ',' + str(a)) 
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
        



