from gensim.parsing.preprocessing import remove_stopwords
from nltk import TweetTokenizer
import nltk
import numpy as np
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


def remove_stop_list(document):
    document = [remove_stopwords(x) for x in document]

    return document

def sortFreqDict(freqdict):
    aux = [(freqdict[key], key) for key in freqdict]
    aux.sort()
    aux.reverse()
    return aux


def get_words(fdist_doc):
    words_doc = []
    for i, word in fdist_doc:
        words_doc.append(word)

    return words_doc


def get_dictionary_from_list(fdist1):
    words1 = get_words(fdist1)
    vocab_plus = set(words1)
    value = [x for x in range(len(vocab_plus))]
    dictionary = dict(zip(value, words1)) # nuevo diccionario

    return dictionary

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
    
def get_pos_neg(path1, path2):
    #function that gets dictionaries from YAKE using pickle function 
    #b is for pos dictionary 
    # c is for negative dictionary 
    with open(path1, "rb") as fp:   # Unpickling
        b = pickle.load(fp)
        fp.close()
    with open(path2, "rb") as fp:   # Unpickling
        c = pickle.load(fp)
        fp.close()
    return b,c

def get_dictionaries(kw1, kw2, num_features1, num_features2, no_distintc):
    if no_distintc == True: 
        l1 = get_words_from_kw(kw1)
        l2 = get_words_from_kw(kw2)

        dictionary1 =l1[:num_features1]
        dictionary2 = l2[:num_features2]
        return dictionary1,dictionary2

    else:
        dic1 = kw1
        dic2 = kw2

        dictionary1 = []
        dictionary2 = []
        i= 0

        l1 = get_words_from_kw(dic1)
        l2 = get_words_from_kw(dic2)
        #print(len(l1), len(l2))
        while len(dictionary2) != num_features2:
            #print(i)
            mem1 = l1[i]
            mem2 = l2[i]
            #print(len(l2),len(l1))
            w1 = dic1[i][0]
            w2 = dic2[i][0]

            if (w2 in l1) == False: 
                dictionary2.append(w2)
            else: 
                indice = l1.index(w2) #buscamos donde se encuentra un miembro de dic1 en dic2
                relevance1 = dic1[indice][1]
                relevance2 = dic2[i][1]
                if relevance1 > relevance2: 
                    dictionary2.append(w2)
                    

            i = i+1
        i = 0
        while len(dictionary1) != num_features1:
            #print(i)
            mem1 = l1[i]
            mem2 = l2[i]
            #print(len(l2),len(l1))
            w1 = dic1[i][0]
            w2 = dic2[i][0]

            if (w1 in l2) == False: 
                dictionary1.append(w1)
            else: 
                indice = l2.index(w1) #buscamos donde se encuentra un miembro de dic1 en dic2
                relevance1 = dic1[i][1]
                relevance2 = dic2[indice][1]
                if relevance1 < relevance2: 
                    dictionary1.append(w1)
            i = i+1

    return  dictionary1, dictionary2
                
path_1 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/key9'
path_2 ='/home/est_posgrado_maria.garcia/Proyecto_tecnologico/key10'

k1,k2 = get_pos_neg(path_1,path_2)

def classificator_pos_neg(pos_data, neg_data, test, num_feat1, num_feat2, tau,chose,
                          fuzzy = True, remove_stop = False, train_data = False,
                          compress = True):
    if remove_stop == True: 
        print("quitando stopwords")
        pos_data = remove_stop_list(pos_data)
        neg_data = remove_stop_list(neg_data)
        test     = remove_stop_list(test)

    pos_data = normalize(pos_data)
    neg_data = normalize(neg_data)
    test = normalize(test)


    num_test = len(test)
    
    fdist_pos = get_fdist(pos_data, num_feat1)
    fdist_neg = get_fdist(neg_data, num_feat2)

    #dictionary1 = get_dictionary_from_list(fdist_pos)
    #dictionary2 = get_dictionary_from_list(fdist_neg)
    dictionary1, dictionary2 = get_dictionaries(k1,k2,num_feat1,num_feat2,True)
    X_test1 = np.zeros((num_test,len(dictionary1)) ,dtype=float) #matriz tipo document-term
    X_test2 = np.zeros((num_test,len(dictionary2)) ,dtype=float)

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

    
    if train_data == True:
        
        if compress == True: 
            X_test1 = np.sum(X_test1, axis = 1)  
            X_test2 = np.sum(X_test2, axis = 1)  

            results = np.concatenate((X_test1, X_test2)).reshape((-1,2),order = 'F')
            X_train = np.concatenate((X_train1,X_train2)).reshape((-1, 2), order='F')
        
        else: 
            total_feat = num_feat1+num_feat2
            X_train = np.concatenate((X_train1,X_train2)).reshape((-1, total_feat), order='F')
                    
            results = np.concatenate((X_test1, X_test2)).reshape((-1,total_feat),order = 'F')
        
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

        return results, X_test1, X_test2, dictionary1,dictionary2



def run_exp_anxia_sim(num_exp, pos_data, neg_data, test_data, test_labels, train_labels,num_feat1,num_feat2, 
                      chose,tau, fuzzy, remove_stop, train_data, compress):
    if train_data == False:
        results,x , y,dic1,dic2 = classificator_pos_neg(pos_data, neg_data, test_data,num_feat1,num_feat2, 
        			tau=tau,chose=chose,fuzzy = fuzzy,remove_stop=remove_stop, train_data =train_data,compress = False)
        result_name = 'result_anxia_key' + str(num_exp) + '.txt'
        path_name =  '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Anorexia_key'+ '/' + result_name
        
        with open(path_name, "w") as f:
        	f.write("Experimento de anorexia número: " + str(num_exp) + '\n')
        	f.write("Confusion matrix: \n")
        	f.write(str(confusion_matrix(test_labels, results)) + '\n')

        	f.write('Metrics classification \n')
        	f.write(str(metrics.classification_report(test_labels, results)) + '\n')

        	f.close()
        path_name_dic1 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Anorexia_key' + '/dictionary1_' + str(num_exp) + '.txt'
        path_name_dic2 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Anorexia_key' + '/dictionary2_' + str(num_exp) + '.txt'
        
        with open(path_name_dic1, "w") as f:
        	for i in range(len(dic1)):
            		f.write(str(dic1[i]) + '\n')
        	f.close()
        with open(path_name_dic2, "w") as f:
        	for i in range(len(dic2)):
            		f.write(str(dic2[i]) + '\n')
        	f.close()
        return f1_score(test_labels, results)
        
    if train_data == True:
        seed_val = 42
        np.random.seed(seed_val)
        parameters = {'C': [.05, .12, .25, .5, 1, 2, 4]}
        
        results, x, y,z,dic1,dic2 = classificator_pos_neg(pos_data, neg_data, test_data,num_feat1,num_feat2, 
        			tau=tau,chose=chose,fuzzy = fuzzy,remove_stop=remove_stop, train_data =train_data,compress = compress)

        svr = svm.LinearSVC(class_weight='balanced', dual=False)
        grid_anorexia = GridSearchCV(estimator=svr, param_grid=parameters, n_jobs=8, scoring='f1_macro', cv=5)
        grid_anorexia.fit(z, train_labels)

        y_pred = grid_anorexia.predict(results)
        a1 = grid_anorexia.best_params_

        p, r, f, _ = precision_recall_fscore_support(test_labels, y_pred, average='macro', pos_label=1)
        result_name = 'result_anxia_key' + str(num_exp) + '.txt'
        path_name = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Anorexia_key' + '/' + result_name
        
        
        with open(path_name, "w") as f:
            f.write("Experimento de anorexia número: " + str(num_exp) + '\n')
            f.write("Confusion matrix: \n")
            f.write(str(confusion_matrix(test_labels, y_pred)) + '\n')

            f.write('Metrics classification \n')
            f.write(str(metrics.classification_report(test_labels, y_pred)) + '\n')

            f.write('Best parameter:\n')
            f.write(str(a1))
            f.write('\n')
            f.close()
        
        path_name_dic1 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Anorexia_key' + '/dictionary1_' + str(num_exp) + '.txt'
        path_name_dic2 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Anorexia_key' + '/dictionary2_' + str(num_exp) + '.txt'
        
        with open(path_name_dic1, "w") as f:
        	for i in range(len(dic1)):
            		f.write(str(dic1[i]) + '\n')
        	f.close()
        
        with open(path_name_dic2, "w") as f:
        	for i in range(len(dic2)):
            		f.write(str(dic2[i]) + '\n')
        	f.close()
        return f1_score(test_labels, y_pred),a1
        
        
            
            
    


def run_exp_dep_sim(num_exp, pos_data, neg_data, test_data, test_labels, train_labels,num_feat1,num_feat2, 
                      chose,tau, fuzzy, remove_stop, train_data, compress):
    print(num_exp, num_feat1,num_feat2, chose,tau, fuzzy, remove_stop, train_data, compress)
    if train_data == False:
        results,x , y,dic1,dic2 = classificator_pos_neg(pos_data, neg_data, test_data,num_feat1,num_feat2, 
        			tau=tau,chose=chose,fuzzy = fuzzy,remove_stop=remove_stop, train_data =train_data,compress = False)
        result_name = 'result_dep_key' + str(num_exp) + '.txt'
        path_name =  '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Depresion_key'+ '/' + result_name
        
        with open(path_name, "w") as f:
        	f.write("Experimento de depresión número: " + str(num_exp) + '\n')
        	f.write("Confusion matrix: \n")
        	f.write(str(confusion_matrix(test_labels, results)) + '\n')

        	f.write('Metrics classification \n')
        	f.write(str(metrics.classification_report(test_labels, results)) + '\n')
        	f.write(str(f1_score(test_labels, results)))
        	f.close()
        path_name_dic1 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Depresion_key/dic1' + '/dictionary1_' + str(num_exp) + '.txt'
        path_name_dic2 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Depresion_key/dic2' + '/dictionary2_' + str(num_exp) + '.txt'
        
        with open(path_name_dic1, "w") as f:
        	for i in range(len(dic1)):
            		f.write(str(dic1[i]) + '\n')
        	f.close()
        
        with open(path_name_dic2, "w") as f:
        	for i in range(len(dic2)):
            		f.write(str(dic2[i]) + '\n')
        	f.close()
        return f1_score(test_labels, results)

    else:
        seed_val = 42
        np.random.seed(seed_val)
        parameters = {'C': [.05, .12, .25, .5, 1, 2, 4]}
        results, x, y,z,dic1,dic2 = classificator_pos_neg(pos_data, neg_data, test_data,num_feat1,num_feat2, 
        			tau=tau,chose=chose,fuzzy = fuzzy,remove_stop=remove_stop, train_data =train_data,compress = compress)

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
        path_name = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Depresion_key' + '/' + result_name

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
        path_name_dic1 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Depresion_key/dic1' + '/dictionary1_' + str(num_exp) + '.txt'
        path_name_dic2 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Depresion_key/dic2' + '/dictionary2_' + str(num_exp) + '.txt'
        
        with open(path_name_dic1, "w") as f:
        	for i in range(len(dic1)):
            		f.write(str(dic1[i]) + '\n')
        	f.close()
        with open(path_name_dic2, "w") as f:
        	for i in range(len(dic2)):
            		f.write(str(dic2[i]) + '\n')
        	f.close()
        return f1_score(test_labels, y_pred),a1
        

