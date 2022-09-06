from gensim.parsing.preprocessing import remove_stopwords
from nltk import TweetTokenizer
import nltk
import numpy as np
from time import time
import re
from gensim.models import FastText
import sklearn
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from gensim.models import utils
from gensim.models import FastText
import gensim
import gensim.downloader
from gensim.test.utils import get_tmpfile, datapath
#from gensim.models import fasttext
import fasttext
import fasttext.util
import pickle
import warnings 
import logging
from statistics import stdev, variance


tokenizer = TweetTokenizer()

print('Load_anxia')
model_anxia = FastText.load('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Models/anxiety.model')
print('Load_dep')
model_dep   = FastText.load('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Models/depresion.model')
print('Load pretrained')
model_pre = fasttext.load_model('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/cc.en.300.bin')


# Pre-processing functions
def normalize(document):

    document = [x.lower()  for x in document]
    # eliminate url
    document = [re.sub(r'https?:\/\/\S+', '', x) for x in document]
    # eliminate url
    document = [re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', x) for x in document]
    # eliminate link
    document = [re.sub(r'{link}', '', x) for x in document]
    # eliminate video
    document = [re.sub(r"\[video\]", '', x) for x in document]
    document = [re.sub(r',', ' ' '', x).strip() for x in document] #quita comas
    document = [re.sub(r'\s+', ' ' '', x).strip() for x in document]
    # eliminate #
    document = [x.replace("#","") for x in document]
    # eliminate emoticons
    document = [re.subn(r'[^\w\s,]',"", x)[0].strip() for x in document]

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
        if option ==2:
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
    similarity_vocab = sklearn.metrics.pairwise.cosine_similarity(words_user, dictionary_vec)
    # vector de representación
    vec_representation = np.count_nonzero(similarity_vocab > epsilon, axis=0)
    return vec_representation

def get_sim_rep(words_doc, dictionary,option, epsilon):
    words_user = np.zeros((len(words_doc),300) , dtype=float)
    dictionary_vec = np.zeros((len(set(dictionary)),300) ,dtype=float)

    for i in range(words_user.shape[0]):
        w1 = words_doc[i]
        if option == 1:
            words_user[i] = model_anxia.wv[w1]
        if option ==2:
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
            
    similarity_vocab = sklearn.metrics.pairwise.cosine_similarity(words_user, dictionary_vec)


    vec_representation = np.sum( similarity_vocab, axis=0, where= similarity_vocab > epsilon)
     
    return vec_representation
    
def get_words_from_kw(kw):
    list1 = []
    for i in range(len(kw)):
        list1.append(kw[i][0])
    return list1


def get_list_key(path):
    #function that gets dictionaries from YAKE using pickle function 
    with open(path, "rb") as fp:   # Unpickling
        b = pickle.load(fp)
        fp.close()
    return b
    
    
def dict_scores(d1,d2, score1, score2, no_distintc):
    #we upload the keywords
    #d1 = get_list_key(k1)
    #d2 = get_list_key(k2)

    #sort by the score 
    d1.sort(key=lambda y: y[1])    
    d2.sort(key=lambda y: y[1])
   
    if no_distintc == True: 
        # obtain the 
        kw1 = [x for x in d1 if x[1] < score1]
        kw2 = [x for x in d2 if x[1] < score2]

        l1 = get_words_from_kw(kw1)
        l2 = get_words_from_kw(kw2)

        return l1,l2

    else:
        #select by the score, when the score is low the importance is greater 
        kw1 = [x for x in d1 if x[1] < score1]
        kw2 = [x for x in d2 if x[1] < score2]

        dictionary1 = []
        dictionary2 = []

        #get only the words
        l1 = get_words_from_kw(kw1)
        l2 = get_words_from_kw(kw2)
        
        len1 = len(l1)
        len2 = len(l2)
        if len1 < len2: 
            for i in range(len1):
                w1 = l1[i]
                #revisamos si no está en el diccionario negativo
                if (w1 in l2) == False: 
                    dictionary1.append(w1)
                #en caso de que lo esté revisamos cual tiene score más pequeño   
                else:
                    #en donde se encuentra en la lista l2 
                    indice = l2.index(w1)
                    #score en la lista 1
                    rel1 = kw1[i][1]
                    #score en la lista 2
                    rel2 = kw2[indice][1]
                    #si el score de la lista 1 es menor que en la 1 la dejamos en lista 1
                    if rel2 > rel1: 
                        dictionary1.append(w1)
            #dictionario negativo son todas las palabras que no están en dictionariopos 
            dictionary2 = [x for x in l2 if x not  in dictionary1]
        else:
            for i in range(len2):
                w2 = l2[i]
                #revisamos si no está en el diccionario positivo
                if (w2 in l1) == False: 
                    dictionary2.append(w2)
                #en caso de que lo esté revisamos cual tiene score más pequeño   
                else:
                    #en donde se encuentra en la lista l1 
                    indice = l1.index(w2)
                    #score en la lista 1
                    rel1 = kw1[indice][1]
                    #score en la lista 2
                    rel2 = kw2[i][1]
                    #si el score de la lista 1 es menor que en la 1 la dejamos en lista 1
                    if rel2 < rel1: 
                        dictionary2.append(w2)
            #dictionario positivo son todas las palabras que no están en dictionario negativo
            dictionary1 = [x for x in l1 if x not  in dictionary2]


    return  dictionary1, dictionary2
                    


                
path_pos_anxia = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/train_yake/dic_pos_anxia_1'
path_neg_anxia = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/train_yake/dic_neg_anxia_1'
path_pos_dep = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/train_yake/dic_pos_dep_1'
path_neg_dep = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/train_yake/dic_neg_dep_1'

#get all the list of the concatenate dictionaries 
k1 = get_list_key(path_pos_anxia)
k2 = get_list_key(path_neg_anxia)
k3 = get_list_key(path_pos_dep)
k4 = get_list_key(path_neg_dep)

#get all the list of the first dictionaries from Sim-Key 
#Keywords for anxia 
path_1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Shuffle_yake/anxia_pos'
path_2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Shuflle_yake/anxia_neg'
#Keywords for depression
path_3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Shuffle_yake/dep_pos'
path_4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Shuffle_yake/dep_neg'

k5 = get_list_key(path_1)
k6 = get_list_key(path_2)
k7 = get_list_key(path_3)
k8 = get_list_key(path_4)


#function classificator
def classificator_pos_neg(pos_data, neg_data, test, score1, score2,di1,di2, tau,chose,dif = True, 
                          fuzzy = True, remove_stop = False):
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
        print("quitando stopwords")
        pos_data = remove_stop_list(pos_data)
        neg_data = remove_stop_list(neg_data)
        test     = remove_stop_list(test)

    pos_data = normalize(pos_data)
    neg_data = normalize(neg_data)
    test = normalize(test)


    num_test = len(test)
    	
    dictionary1,dictionary2 = dict_scores(di1,di2,score1, score2, no_distintc = dif)
    X_test1 = np.zeros((num_test,len(dictionary1)) ,dtype=float) #matriz tipo document-term
    X_test2 = np.zeros((num_test,len(dictionary2)) ,dtype=float)
    num_feat1 = len(dictionary1)
    num_feat2 = len(dictionary2)

   ### For using svm #### 
    
    train = [*pos_data, *neg_data]
    num_train = len(train)
    X_train1 = np.zeros((num_train,len(dictionary1)), dtype= float)
    X_train2 = np.zeros((num_train,len(dictionary2)), dtype= float)
    for i in range(num_train): 
        doc = train[i]
        corpus_palabras = tokenizer.tokenize(doc.lower())
        fdist = nltk.FreqDist(corpus_palabras)
        v = sortFreqDict(fdist)
        words_doc =  get_words(v) 
        if fuzzy == True: 
            word_repre_user = get_fuzzy_rep(words_doc, dictionary1,option = chose, epsilon = tau)
        else: 
            word_repre_user = get_sim_rep(words_doc, dictionary1,option =chose, epsilon = tau)
        X_train1[i] = word_repre_user
    for i in range(num_train): 
        doc = train[i]
        corpus_palabras = tokenizer.tokenize(doc.lower())
        fdist = nltk.FreqDist(corpus_palabras)
        v = sortFreqDict(fdist)
        words_doc =  get_words(v) 
        if fuzzy == True: 
            word_repre_user = get_fuzzy_rep(words_doc, dictionary2, option = chose,epsilon = tau)
        else: 
            word_repre_user = get_sim_rep(words_doc, dictionary2, option = chose,epsilon = tau)
        X_train2[i] = word_repre_user
        
    
    X_train3 = np.sum(X_train1, axis = 1)  
    X_train4 = np.sum(X_train2, axis = 1)  

    #- Construct test representation of positive users
    for i in range(num_test): 
        doc = test[i]
        corpus_palabras = tokenizer.tokenize(doc.lower())
        fdist = nltk.FreqDist(corpus_palabras)
        v = sortFreqDict(fdist)
        words_doc =  get_words(v) 
        if fuzzy == True: 
            word_repre_user = get_fuzzy_rep(words_doc, dictionary1, option= chose,epsilon = tau)
        else: 
            word_repre_user = get_sim_rep(words_doc, dictionary1, option=chose,epsilon = tau)
        X_test1[i] = word_repre_user

    #Construct test representation of negative users
    for i in range(num_test): 
        doc = test[i]
        corpus_palabras = tokenizer.tokenize(doc.lower())
        fdist = nltk.FreqDist(corpus_palabras)
        v = sortFreqDict(fdist)
        words_doc =  get_words(v) 
        if fuzzy == True: 
            word_repre_user = get_fuzzy_rep(words_doc, dictionary2,option= chose, epsilon = tau)
        else: 
            word_repre_user = get_sim_rep(words_doc, dictionary2,option = chose, epsilon = tau)
        X_test2[i] = word_repre_user
    #print("Vectorization for test data: Done")

        
    
    X_test3 = np.sum(X_test1, axis = 1)  
    X_test4 = np.sum(X_test2, axis = 1)  


    X_train = np.concatenate((X_train1, X_train2), axis = 1)
    X_train = np.c_[X_train, X_train3]
    X_train = np.c_[X_train, X_train4]
    X_test =  np.concatenate((X_test1, X_test2), axis = 1)
    X_test = np.c_[X_test, X_test3]
    X_test = np.c_[X_test, X_test4]
    
    print(X_train.shape, X_test.shape)
        
    return X_test,X_train, len(dictionary1),len(dictionary2)



def run_exp_anxia_sim(num_exp, pos_data, neg_data, test_data, test_labels, train_labels,score1,score2, 
                      chose,tau, dif,fuzzy, remove_stop, concatenate):
    print(num_exp)
    logging.basicConfig(filename="/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Key_Con/warnings_anorexia.txt",level=logging.DEBUG)
    logging.debug('\n This is a message from experimet number ' + str(num_exp) )
    logging.info('\n This is a message for info')
    logging.captureWarnings(True)
    logging.warning('\n Warning message for experiment number ' + str(num_exp))


    t_initial = time()
    if concatenate == True: 
        kw1 = k5
        kw2 = k6
    else: 
        kw1 = k1
        kw2 = k2
    		        
    
    seed_val = 42
    np.random.seed(seed_val)
    parameters = {'C': [.05, .12, .25, .5, 1, 2, 4]}
        
    X_test,X_train,len_dic1,len_dic2 = classificator_pos_neg(pos_data, neg_data, test_data,score1,score2,kw1,kw2, 
        			tau=tau,chose=chose,dif = dif, fuzzy = fuzzy,remove_stop=remove_stop)	
        
    if X_test.shape[1] < X_test.shape[0]:
        svr = svm.LinearSVC(class_weight='balanced', dual=False, max_iter = 2000)
        
    else: 
        svr = svm.LinearSVC(class_weight='balanced', dual=True, max_iter = 2000)
        
    grid_anorexia = GridSearchCV(estimator=svr, param_grid=parameters, n_jobs=8, scoring='f1_macro', cv=5)
    grid_anorexia.fit(X_train, train_labels)

    y_pred = grid_anorexia.predict(X_test)
    a= grid_anorexia.best_params_
    f1 = f1_score(test_labels, y_pred)
    f = open('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Key_Con/f1_anorexia.txt','a')
    f.write('\n' + str(num_exp) + ',' + str(score1) + ',' + str(score2) +',' + str(tau) +',' + str(dif) 
            +','+ str(fuzzy) +','+ str(remove_stop) + ',' + str(chose)+ ',' + str(f1) + ',' + str(a)) 
    f.close()
        
    print("done in %fs" % (time() - t_initial))
        
    return f1_score(test_labels, y_pred),a, len_dic1, len_dic2
        
        
            
            
    


def run_exp_dep_sim(num_exp, pos_data, neg_data, test_data, test_labels, train_labels,score1,score2, 
                      chose,tau, dif, fuzzy, remove_stop,concatenate):
    logging.basicConfig(filename="/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Key_Con/warnings_dep.txt",level=logging.DEBUG)
    logging.debug('\n This is a message from experimet number ' + str(num_exp) )
    logging.info('\n This is a message for info')
    logging.captureWarnings(True)
    logging.warning('\nWarning message for experiment number ' + str(num_exp))

    
    print(num_exp)
    t_initial = time()
    if concatenate == True: 
        kw1 = k7
        kw2 = k8
    else: 
        kw1 = k3
        kw2 = k4    
    
    seed_val = 42
    np.random.seed(seed_val)
    parameters = {'C': [.05, .12, .25, .5, 1, 2, 4]}
        
    X_test,X_train,len_dic1,len_dic2 = classificator_pos_neg(pos_data, neg_data, test_data,score1,score2,kw1,kw2, 
        			tau=tau,chose=chose,dif = dif, fuzzy = fuzzy,remove_stop=remove_stop)	
        
    if X_test.shape[1] < X_test.shape[0]:
        svr = svm.LinearSVC(class_weight='balanced', dual=False, max_iter = 6000)
        
    else: 
        svr = svm.LinearSVC(class_weight='balanced', dual=True, max_iter = 6000)
        
    grid_dep = GridSearchCV(estimator=svr, param_grid=parameters, n_jobs=8, scoring='f1_macro', cv=5)
    grid_dep.fit(X_train, train_labels)

    y_pred = grid_dep.predict(X_test)
    a= grid_dep.best_params_
    f1 = f1_score(test_labels, y_pred)
    f = open('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Key_Con/f1_dep.txt','a')
    f.write('\n' + str(num_exp) + ',' + str(score1) + ',' + str(score2) +',' + str(tau) +',' + str(dif) 
            +','+ str(fuzzy) +','+ str(remove_stop) + ',' + str(chose)+ ',' + str(f1) + ',' + str(a)) 
    f.close()       
    
    print("done in %fs" % (time() - t_initial))
    return f1,a, len_dic1, len_dic2
        

