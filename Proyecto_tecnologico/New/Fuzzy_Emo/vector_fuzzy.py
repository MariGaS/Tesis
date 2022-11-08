import nltk
import gensim
import re
import os
import numpy as np
import sklearn
from gensim.models import FastText
from gensim.parsing.preprocessing import remove_stopwords
from nltk import TweetTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from gensim.models import utils
import pickle
from gensim.models import FastText
import gensim.downloader
from gensim.test.utils import get_tmpfile, datapath
#from gensim.models import fasttext
import fasttext
import fasttext.util
from sklearn.naive_bayes import MultinomialNB
from time import time

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

# Only obtain the words from the dictionary with the frequency of the words # 
def get_words(fdist_doc):
    words_doc = []
    for i, word in fdist_doc:
        words_doc.append(word)

    return words_doc


def sortFreqDict(freqdict):
    aux = [(freqdict[key], key) for key in freqdict]
    aux.sort()
    aux.reverse()
    return aux



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


def get_words_dictionary(path_name):
    vocab = []
    with open(path_name, 'r') as f:
        for twitt in f:
            vocab.append(twitt[:-1].replace(" ", "_"))
        f.close()
    vocab.pop(0)
    vocab_plus = set(vocab)
    value = [x for x in range(len(vocab_plus))]
    dictionary = dict(zip(value, set(vocab_plus)))

    return dictionary

def get_dictionary(name_dic):

    if name_dic == 'dict1':
        path_name = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Results/Anorexia/features_words_3.txt'
        dictionary = get_words_dictionary(path_name)

    elif name_dic == 'dict2':
        path_name = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Results/Anorexia/features_words_4.txt'
        dictionary = get_words_dictionary(path_name)

    elif name_dic == 'dict3':
        path_name = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Results/Anorexia/features_words_6.txt'
        dictionary = get_words_dictionary(path_name)

    elif name_dic == 'dict4':
        path_name = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Results/Anorexia/features_words_8.txt'
        dictionary = get_words_dictionary(path_name)

    elif name_dic == 'dict5':
        path_name = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Results/Depresion/features_words_final_1.txt' #1
        dictionary = get_words_dictionary(path_name)

    elif name_dic == 'dict6':
        path_name = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Results/Depresion/features_words_final_2.txt'#3
        dictionary = get_words_dictionary(path_name)

    elif name_dic == 'dict7':
        path_name = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Results/Depresion/features_words_final_3.txt' #4
        dictionary = get_words_dictionary(path_name)

    elif name_dic == 'dict8':
        path_name = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Results/Depresion/features_words_final_4.txt'#5
        dictionary = get_words_dictionary(path_name)

    return dictionary


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


def classificator_fuzzy(all_path_train, all_path_test, path_tf_train, path_tf_test, num_test, num_train,
                        name_dic, tau, chose,fuzzy = True,tf= True, only_bow = False):

    if only_bow == True: 
        dictionary_emo = dict()
        dictionary_emo[0] = 'anger'
        dictionary_emo[1] = 'anticipation'
        dictionary_emo[2] = 'disgust'
        dictionary_emo[3] = 'fear'
        dictionary_emo[4] = 'joy'
        dictionary_emo[5] = 'negative'
        dictionary_emo[6] = 'positive'
        dictionary_emo[7] = 'sadness'
        dictionary_emo[8] = 'surprise'
        dictionary_emo[9] = 'trust'
        dictionary = dictionary_emo
        emb_dic = get_dictionary_matrix(dictionary, option = chose)
    # obtenemos dictionario de features
    else: 
        dictionary = get_dictionary(name_dic)
        n = len(dictionary)
        dictionary[n] = 'anger'
        dictionary[n+1] = 'anticipation'
        dictionary[n+2] = 'disgust'
        dictionary[n+3] = 'fear'
        dictionary[n+4] = 'joy'
        dictionary[n+5] = 'negative'
        dictionary[n+6] = 'positive'
        dictionary[n+7] = 'sadness'
        dictionary[n+8] = 'surprise'
        dictionary[n+9] = 'trust'

        emb_dic = get_dictionary_matrix(dictionary, option = chose)
    	
    print(len(dictionary))

    X_train = np.zeros((num_train, len(dictionary)), dtype=float)
    X_test = np.zeros((num_test, len(dictionary)), dtype=float)  # matriz tipo document-term
    print(X_train.shape)
    print(X_test.shape)
    for i in range(num_train):
        path =all_path_train + '_'+ str(i)
        with open(path, "rb") as fp:   # Unpickling
            b = pickle.load(fp)
            fp.close()

        if fuzzy == True:
            word_repre_user = get_fuzzy_rep(b, emb_dic, epsilon=tau)
        else:
            word_repre_user = get_sim_rep(b, emb_dic, epsilon=tau)
        if tf == True:
            path2 =path_tf_train + '_'+ str(i)
            with open(path2, "rb") as fp:   # Unpickling
                corpus = pickle.load(fp)
                fp.close()
            for j in range(len(corpus)):
                frequency = corpus[j][0]
                word_repre_user[j] = frequency* word_repre_user[j]
        final_rep = np.sum(word_repre_user, axis=0)

        X_train[i] = final_rep
    

    print("Vectorization for train data: Done")
    for i in range(num_test):
        path =all_path_test + '_'+ str(i)
        with open(path, "rb") as fp:   # Unpickling
            b = pickle.load(fp)
            fp.close()    
        if fuzzy == True:
            word_repre_user = get_fuzzy_rep(b, emb_dic, epsilon=tau)
        else:
            word_repre_user = get_sim_rep(b,emb_dic, epsilon=tau)
        if tf == True:
            path2 =path_tf_test + '_'+ str(i)
            with open(path2, "rb") as fp:   # Unpickling
                corpus = pickle.load(fp)
                fp.close()
            for j in range(len(corpus)):
                frequency = corpus[j][0]
                word_repre_user[j] = frequency* word_repre_user[j]
        final_rep = np.sum(word_repre_user, axis=0)
        X_test[i] = final_rep
    print("Vectorization for test data: Done")

    return X_test, X_train, len(dictionary)

def run_exp_anxia_sim(num_exp, test_labels, train_labels, num_test, num_train,name_dic,
                    chose, tau,fuzzy, remove_stop,only_bow, tf, classificator):

    print(num_exp)


    t_initial = time()
    for_all = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Fuzzy_key_post/Matrix_user/'
    tf_path = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Fuzzy_key_post/corpus_user/'
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

    seed_val = 42
    np.random.seed(seed_val)
    parameters = {'C': [.05, .12, .25, .5, 1, 2, 4]}

    if tf == False:
        path_tf_train = ''
        path_tf_test = ''

    X_test,X_train,len_dict = classificator_fuzzy(all_path_train = path_train, all_path_test =path_test,
                    path_tf_train = path_tf_train, path_tf_test= path_tf_test,
                    num_test = num_test, num_train = num_train,
                    name_dic = name_dic, tau = tau, chose=chose,fuzzy = fuzzy,tf= tf, only_bow=only_bow)	
    if classificator == 'SVM':
            
        if X_test.shape[1] < X_test.shape[0]:
            svr = svm.LinearSVC(class_weight='balanced', dual=False, max_iter = 8000)
            
        else: 
            svr = svm.LinearSVC(class_weight='balanced', dual=True, max_iter = 8000)
            
        grid_anorexia = GridSearchCV(estimator=svr, param_grid=parameters, n_jobs=8, scoring='f1_macro', cv=5)
        grid_anorexia.fit(X_train, train_labels)

        y_pred = grid_anorexia.predict(X_test)
        a= grid_anorexia.best_params_
        f1 = f1_score(test_labels, y_pred)
    if classificator == 'NB':
        model = MultinomialNB()
        model.fit(X_train, train_labels)
        y_pred = model.predict(X_test)
        f1 = f1_score(test_labels, y_pred)

    if fuzzy == True:
        fuzzy_str = 'Fuzzy'
    if fuzzy == False:
        fuzzy_str = 'Not Fuzzy'
    if remove_stop == True:
        remove_stop_str = 'Removed stopwords'
    if remove_stop == False:
        remove_stop_str = 'Not removed stopwords'
    if chose == 1: 
        w_e = 'Anorexia Model'
    if chose == 3: 
        w_e = 'Pre_trained Model'
    if tf == True:
        weight = 'tf'
    else: 
        weight = 'binary'

    if only_bow: 
        b = 'Only emotions'
    if only_bow == False:
        b = 'Emotions and words'  

    if classificator == 'NB':
        a = 'None'
    f = open('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Fuzzy_Emo/f1_anorexia.txt','a')
    f.write('\n' + str(num_exp) + ',' + str(tau)  
            +','+ fuzzy_str+','+ remove_stop_str 
            +','+ w_e + ','+ weight + ',' + name_dic+  ',' + str(len_dict) +  ',' + b  +  ',' + classificator + ','+ str(f1) + ',' + str(a)) 
    f.close()   
        
    print("done in %fs" % (time() - t_initial))
        
    return f1_score(test_labels, y_pred)

def run_exp_dep_sim(num_exp, test_labels, train_labels, num_test, num_train,name_dic,
                    chose, tau,fuzzy, remove_stop, tf, only_bow, classificator):
    
    t_initial = time()
    for_all = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Fuzzy_key_post/Matrix_user/'
    tf_path = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Fuzzy_key_post/corpus_user/'

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
    if tf == False:
        path_tf_train = ''
        path_tf_test = ''
    seed_val = 42
    np.random.seed(seed_val)
    parameters = {'C': [.05, .12, .25, .5, 1, 2, 4]}
    if fuzzy == True:
        fuzzy_str = 'Fuzzy'
    if fuzzy == False:
        fuzzy_str = 'Not Fuzzy'
    if remove_stop == True:
        remove_stop_str = 'Removed stopwords'
    if remove_stop == False:
        remove_stop_str = 'Not removed stopwords'
    if chose == 2: 
        w_e = 'Depression Model'
    if chose == 3: 
        w_e = 'Pre_trained Model'
    if tf == True:
        weight = 'tf'
    else: 
        weight = 'binary'
    if only_bow: 
        b = 'Only emotions'
    if only_bow == False:
        b = 'Emotions and words'  

    X_test,X_train, len_dic =classificator_fuzzy(all_path_train = path_train, all_path_test =path_test,
                    path_tf_train = path_tf_train, path_tf_test= path_tf_test,
                    num_test = num_test, num_train = num_train,
                    name_dic = name_dic, tau = tau, chose=chose,fuzzy = fuzzy,tf= tf, only_bow=only_bow)	
    if classificator == 'SVM':
        if X_test.shape[1] < X_test.shape[0]:
            svr = svm.LinearSVC(class_weight='balanced', dual=False, max_iter = 6000)
            
        else: 
            svr = svm.LinearSVC(class_weight='balanced', dual=True, max_iter = 6000)
            
        grid_dep = GridSearchCV(estimator=svr, param_grid=parameters, n_jobs=8, scoring='f1_macro', cv=5)
        grid_dep.fit(X_train, train_labels)

        y_pred = grid_dep.predict(X_test)
        a= grid_dep.best_params_
        f1 = f1_score(test_labels, y_pred)
    
    if classificator == 'NB':
        model = MultinomialNB()
        model.fit(X_train, train_labels)
        y_pred = model.predict(X_test)
        f1 = f1_score(test_labels, y_pred)
        a = 'None'

    f = open('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Fuzzy_Emo/f1_dep.txt','a')
    f.write('\n' + str(num_exp) +',' + str(tau)
            +','+ fuzzy_str+','+ remove_stop_str
            +','+ w_e + ','+ weight + ',' + name_dic + ',' + str(len_dic)+ ',' +b+ ',' + classificator+','  + str(f1) + ',' + str(a)) 
    f.close()       

    print('The time for this experiment was %fs' % (time() - t_initial))

    return f1
        