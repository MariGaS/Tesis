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

from text_functions import (get_text_labels,
                            get_text_test)
tokenizer = TweetTokenizer()

print('Load_anxia')
model_anxia = FastText.load('/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Model/anxiety.model')
print('Load_dep')
model_dep   = FastText.load('/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Model/depresion.model')
print('Load pretrained')
model_pre = fasttext.load_model('/home/est_posgrado_maria.garcia/Proyecto_tecnologico/cc.en.300.bin')


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

def get_fuzzy_rep(words_doc, dictionary, option, epsilon, dict_key, weight = True):
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
    if weight: 
        new_vec_rep = np.zeros((vec_representation.shape), dtype = float)
        #dict_key is the dictionary containing the words with their scores
        #tem contains only the words from dict_key 
        tem = get_words_from_kw(dict_key) 

        #multiply the tf weight by the score of thwe word
        for i in range(len(dictionary)):
            word_target = dictionary[i] #word to obtain it's score 
            #index of the word in dict_key 
            indice = tem.index(word_target)
            score = dict_key[indice][1] #score of the word in the dictionary 
            #new weight 
            t = float(vec_representation[i])
            
            new_vec_rep[i] = t*score 
        
        vec_representation = new_vec_rep     
    return vec_representation

def get_sim_rep(words_doc, dictionary,option, epsilon, dict_key,weight = True):
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
    if weight : 
        new_vec_rep = np.zeros((vec_representation.shape), dtype = float)
        #dict_key is the dictionary containing the words with their scores
        #tem contains only the words from dict_key 
        tem = get_words_from_kw(dict_key) 

        #multiply the tf weight by the score of thwe word
        for i in range(len(dictionary)):
            word_target = dictionary[i] #word to obtain it's score 
            #index of the word in dict_key 
            indice = tem.index(word_target)
            score = dict_key[indice][1] #score of the word in the dictionary 
            #new weight 
            t = float(vec_representation[i])
            
            new_vec_rep[i] = t*score 
        
        vec_representation = new_vec_rep 
     
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
                    


                
path_pos_anxia = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/train_yake/dic_pos_anxia'
path_neg_anxia = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/train_yake/dic_neg_anxia'
path_pos_dep = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/train_yake/dic_pos_dep'
path_neg_dep = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/train_yake/dic_neg_dep'

#get all the list of the concatenate dictionaries 
k1 = get_list_key(path_pos_anxia)
k2 = get_list_key(path_neg_anxia)
k3 = get_list_key(path_pos_dep)
k4 = get_list_key(path_neg_dep)

#get all the list of the first dictionaries from Sim-Key 
#Keywords for anxia 
path_1 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/keywords/key1'
path_2 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/keywords/key2'
#Keywords for depression
path_3 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/keywords/key5'
path_4 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/keywords/key6'

k5 = get_list_key(path_1)
k6 = get_list_key(path_2)
k7 = get_list_key(path_3)
k8 = get_list_key(path_4)


#function classificator
def classificator_pos_neg(pos_data, neg_data, test, score1, score2,di1,di2, tau,chose,w, dif = True, 
                          fuzzy = True, remove_stop = False, train_data = False,
                          compress = True):
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
    if train_data== True: 
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
                word_repre_user = get_fuzzy_rep(words_doc, dictionary1,option = chose, epsilon = tau, dict_key = di1, weight = w)
            else: 
                word_repre_user = get_sim_rep(words_doc, dictionary1,option =chose, epsilon = tau, dict_key = di1,weight = w)
            X_train1[i] = word_repre_user
        for i in range(num_train): 
            doc = train[i]
            corpus_palabras = tokenizer.tokenize(doc.lower())
            fdist = nltk.FreqDist(corpus_palabras)
            v = sortFreqDict(fdist)
            words_doc =  get_words(v) 
            if fuzzy == True: 
                word_repre_user = get_fuzzy_rep(words_doc, dictionary2, option = chose,epsilon = tau, dict_key = di2,weight = w)
            else: 
                word_repre_user = get_sim_rep(words_doc, dictionary2, option = chose,epsilon = tau,dict_key = di2, weight = w)
            X_train2[i] = word_repre_user
        
        if compress == True:
            X_train1 = np.sum(X_train1, axis = 1)  
            X_train2 = np.sum(X_train2, axis = 1)  

    #- Construct test representation of positive users
    for i in range(num_test): 
        doc = test[i]
        corpus_palabras = tokenizer.tokenize(doc.lower())
        fdist = nltk.FreqDist(corpus_palabras)
        v = sortFreqDict(fdist)
        words_doc =  get_words(v) 
        if fuzzy == True: 
            word_repre_user = get_fuzzy_rep(words_doc, dictionary1, option= chose,epsilon = tau,dict_key = di1,weight = w)
        else: 
            word_repre_user = get_sim_rep(words_doc, dictionary1, option=chose,epsilon = tau,dict_key = di1, weight = w)
        X_test1[i] = word_repre_user

    #Construct test representation of negative users
    for i in range(num_test): 
        doc = test[i]
        corpus_palabras = tokenizer.tokenize(doc.lower())
        fdist = nltk.FreqDist(corpus_palabras)
        v = sortFreqDict(fdist)
        words_doc =  get_words(v) 
        if fuzzy == True: 
            word_repre_user = get_fuzzy_rep(words_doc, dictionary2,option= chose, epsilon = tau, dict_key = di2,weight = w)
        else: 
            word_repre_user = get_sim_rep(words_doc, dictionary2,option = chose, epsilon = tau,dict_key = di2, weight = w)
        X_test2[i] = word_repre_user
    #print("Vectorization for test data: Done")

    
    if train_data == True:
        
        if compress == True: 
            X_test1 = np.sum(X_test1, axis = 1)  
            X_test2 = np.sum(X_test2, axis = 1)  
            #X_train = np.concatenate((X_train1, X_train2), axis = 1)
            #results = np.concatenate((X_test1, X_test2), axis = 1)
            results = np.concatenate((X_test1, X_test2)).reshape((-1,2),order = 'F')
            X_train = np.concatenate((X_train1,X_train2)).reshape((-1, 2), order='F')
            print(X_train.shape, results.shape)
        
        else: 
            total_feat = num_feat1+num_feat2
            X_train = np.concatenate((X_train1, X_train2), axis = 1)
            results = np.concatenate((X_test1, X_test2), axis = 1)
            print(X_train.shape, results.shape)
        
        return results, X_test1, X_test2,X_train,dictionary1,dictionary2
         


    if train_data == False: 
        X_test1 = np.sum(X_test1, axis = 1)  
        X_test2 = np.sum(X_test2, axis = 1)  

        results = np.zeros((num_test), dtype = int)
        for i in range(num_test):
            if X_test1[i] > X_test2[i]:
                results[i] = 1
            else: 
                results[i] = 0
        print(results.shape)
        return results, X_test1, X_test2, dictionary1,dictionary2



def run_exp_anxia_sim(num_exp, pos_data, neg_data, test_data, test_labels, train_labels,score1,score2, 
                      chose,tau,w, dif,fuzzy, remove_stop, train_data, compress, concatenate):
    print(num_exp)
    t_initial = time()
    if concatenate == True: 
    	kw1 = k5
    	kw2 = k6
    else: 
    	kw1 = k1
    	kw2 = k2
    		          
    if train_data == False:
        results,x , y,dic1,dic2 = classificator_pos_neg(pos_data, neg_data, test_data,score1,score2,kw1,kw2, 
        			tau=tau,chose=chose,w = w,dif = dif, fuzzy = fuzzy,remove_stop=remove_stop, train_data =train_data,compress = False)
        result_name = 'result_anxia_key' + str(num_exp) + '.txt'
        path_name =  '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Anorexia_w_key'+ '/' + result_name
        
        with open(path_name, "w") as f:
        	f.write("Experimento de anorexia número: " + str(num_exp) + '\n')
        	f.write("Confusion matrix: \n")
        	f.write(str(confusion_matrix(test_labels, results)) + '\n')

        	f.write('Metrics classification \n')
        	f.write(str(metrics.classification_report(test_labels, results)) + '\n')
        	f.write('F1-score:\n')
        	f.write(str(f1_score(test_labels, results)))

        	f.close()
        path_name_dic1 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Anorexia_w_key/dic1' + '/dictionary1_' + str(num_exp) + '.txt'
        path_name_dic2 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Anorexia_w_key/dic2' + '/dictionary2_' + str(num_exp) + '.txt'
        
        with open(path_name_dic1, "w") as f:
        	for i in range(len(dic1)):
            		f.write(str(dic1[i]) + '\n')
        	f.close()
        with open(path_name_dic2, "w") as f:
        	for i in range(len(dic2)):
            		f.write(str(dic2[i]) + '\n')
        	f.close()
        print("done in %fs" % (time() - t_initial))
        return f1_score(test_labels, results)
        
    if train_data == True:
        seed_val = 42
        np.random.seed(seed_val)
        parameters = {'C': [.05, .12, .25, .5, 1, 2, 4]}
        
        results, x, y,z,dic1,dic2 = classificator_pos_neg(pos_data, neg_data, test_data,score1,score2,kw1,kw2, 
        			tau=tau,chose=chose,w = w,dif = dif, fuzzy = fuzzy,remove_stop=remove_stop, train_data =train_data,compress = compress)
		
        if compress == True: 
        	svr = svm.LinearSVC(class_weight='balanced', dual=False)
        else: 
        	svr = svm.LinearSVC(class_weight='balanced', dual=True)
        
        grid_anorexia = GridSearchCV(estimator=svr, param_grid=parameters, n_jobs=8, scoring='f1_macro', cv=5)
        grid_anorexia.fit(z, train_labels)

        y_pred = grid_anorexia.predict(results)
        a1 = grid_anorexia.best_params_

        result_name = 'result_anxia_key' + str(num_exp) + '.txt'
        path_name = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Anorexia_w_key' + '/' + result_name
        
        
        with open(path_name, "w") as f:
            f.write("Experimento de anorexia número: " + str(num_exp) + '\n')
            f.write("Confusion matrix: \n")
            f.write(str(confusion_matrix(test_labels, y_pred)) + '\n')

            f.write('Metrics classification \n')
            f.write(str(metrics.classification_report(test_labels, y_pred)) + '\n')

            f.write('Best parameter:\n')
            f.write(str(a1))
            f.write('\n')
            f.write('F1-score:\n')
            f.write(str(f1_score(test_labels, y_pred)))
            f.close()
        
        path_name_dic1 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Anorexia_w_key/dic1' + '/dictionary1_' + str(num_exp) + '.txt'
        path_name_dic2 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Anorexia_w_key/dic2' + '/dictionary2_' + str(num_exp) + '.txt'
        
        with open(path_name_dic1, "w") as f:
        	for i in range(len(dic1)):
            		f.write(str(dic1[i]) + '\n')
        	f.close()
        
        with open(path_name_dic2, "w") as f:
        	for i in range(len(dic2)):
            		f.write(str(dic2[i]) + '\n')
        	f.close()
        print("done in %fs" % (time() - t_initial))
        
        return f1_score(test_labels, y_pred),a1
        
        
            
            
    


def run_exp_dep_sim(num_exp, pos_data, neg_data, test_data, test_labels, train_labels,score1,score2, 
                      chose,tau, w,dif, fuzzy, remove_stop, train_data, compress, concatenate):
                      
    
    print(num_exp)
    t_initial = time()
    if concatenate == True: 
    	kw1 = k7
    	kw2 = k8
    else: 
    	kw1 = k3
    	kw2 = k4    
    
    
    if train_data == False:
        results,x , y,dic1,dic2 = classificator_pos_neg(pos_data, neg_data, test_data,score1,score2,kw1,kw2, 
        			tau=tau,chose=chose,w = w,dif = dif, fuzzy = fuzzy,remove_stop=remove_stop, train_data =train_data,compress = False)
        result_name = 'result_dep_key' + str(num_exp) + '.txt'
        path_name =  '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Depresion_w_key'+ '/' + result_name
        
        with open(path_name, "w") as f:
        	f.write("Experimento de depresión número: " + str(num_exp) + '\n')
        	f.write("Confusion matrix: \n")
        	f.write(str(confusion_matrix(test_labels, results)) + '\n')

        	f.write('Metrics classification \n')
        	f.write(str(metrics.classification_report(test_labels, results)) + '\n')
        	f.write(str(f1_score(test_labels, results)))
        	f.write('F1-score:\n')
        	f.write(str(f1_score(test_labels, results)))
            
        	f.close()
        path_name_dic1 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Depresion_w_key/dic1' + '/dictionary1_' + str(num_exp) + '.txt'
        path_name_dic2 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Depresion_w_key/dic2' + '/dictionary2_' + str(num_exp) + '.txt'
        
        with open(path_name_dic1, "w") as f:
        	for i in range(len(dic1)):
            		f.write(str(dic1[i]) + '\n')
        	f.close()
        
        with open(path_name_dic2, "w") as f:
        	for i in range(len(dic2)):
            		f.write(str(dic2[i]) + '\n')
        	f.close()
        print("done in %fs" % (time() - t_initial))
        return f1_score(test_labels, results)

    else:
        seed_val = 42
        np.random.seed(seed_val)
        parameters = {'C': [.05, .12, .25, .5, 1, 2, 4]}
        results, x, y,z,dic1,dic2 = classificator_pos_neg(pos_data, neg_data, test_data,score1,score2,kw1,kw2,
        			tau=tau,chose=chose,w = w,dif = dif, fuzzy = fuzzy,remove_stop=remove_stop, train_data =train_data,compress = compress)

        if compress == True: 
        	svr = svm.LinearSVC(class_weight='balanced', dual=False)
        else: 
        	svr = svm.LinearSVC(class_weight='balanced', dual=True)
        		
        grid_anorexia = GridSearchCV(estimator=svr, param_grid=parameters, n_jobs=8, scoring='f1_macro', cv=5)
        grid_anorexia.fit(z, train_labels)

        y_pred = grid_anorexia.predict(results)
        a1 = grid_anorexia.best_params_

        p, r, f, _ = precision_recall_fscore_support(test_labels, y_pred, average='macro', pos_label=1)
        result_name = 'result_dep_key' + str(num_exp) + '.txt'
        path_name = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Depresion_w_key' + '/' + result_name

        with open(path_name, "w") as f:
            f.write("Experimento de depresión número: " + str(num_exp) + '\n')
            f.write("Confusion matrix: \n")
            f.write(str(confusion_matrix(test_labels, y_pred)) + '\n')

            f.write('Metrics classification \n')
            f.write(str(metrics.classification_report(test_labels, y_pred)) + '\n')

            f.write('Best parameter:\n')
            f.write(str(a1))
            f.write('\n')
            
            
            f.write('F1-score:\n')
            f.write(str(f1_score(test_labels, y_pred)))
            
            f.close()
        path_name_dic1 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Depresion_w_key/dic1' + '/dictionary1_' + str(num_exp) + '.txt'
        path_name_dic2 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Depresion_w_key/dic2' + '/dictionary2_' + str(num_exp) + '.txt'
        
        with open(path_name_dic1, "w") as f:
        	for i in range(len(dic1)):
            		f.write(str(dic1[i]) + '\n')
        	f.close()
        with open(path_name_dic2, "w") as f:
        	for i in range(len(dic2)):
            		f.write(str(dic2[i]) + '\n')
        	f.close()
        print("done in %fs" % (time() - t_initial))
        return f1_score(test_labels, y_pred),a1
        
