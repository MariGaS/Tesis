from time import time
import numpy as np
from numpy import array, asarray, zeros
import re
import sklearn
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler
import nltk
from nltk import TweetTokenizer
from gensim.models import FastText
import gensim.downloader
from gensim.test.utils import get_tmpfile, datapath
#from gensim.models import fasttext
import fasttext
import fasttext.util

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
    document = [re.sub(r'\s+', ' ' '', x).strip() for x in document]
    # eliminate #
    document = [x.replace("#","") for x in document]
    # eliminate emoticons
    document = [re.subn(r'[^\w\s,]',"", x)[0].strip() for x in document]

    return document


tokenizer = TweetTokenizer()


def sortFreqDict(freqdict):
    aux = [(freqdict[key], key) for key in freqdict]
    aux.sort()
    aux.reverse()
    return aux

#  ---------- CONSTRUCT Glove REPRESENTATION -------------------
#embeddings_file_path = 'glove.6B.300d.txt'
#embeddings_dict_w2v = dict()
#w2v_file = open(embeddings_file_path)

#for line in w2v_file:
#    records = line.split() # Turn into array with word on first position and embeddings as rest of line.
#    word = records[0]
#    vector_dim = asarray(records[1:], dtype='float32') # Take rest of embeddings out.
#    # add to embeddings_dict as word:embeddings.
#    embeddings_dict_w2v[word] = vector_dim


#load models
print('Load_anxia')
model_anxia = FastText.load('/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Model/anxiety.model')
print('Load_dep')
model_dep   = FastText.load('/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Model/depresion.model')
print('Load pretrained')
model_pre = fasttext.load_model('/home/est_posgrado_maria.garcia/Proyecto_tecnologico/cc.en.300.bin')
#model_pre = fasttext.load_facebook_vectors(datapath("./wiki.en/wiki.en.bin"))

def get_words_dic(dictionary_list):
    '''
    For each word of dictionary it check if
    the words has GloVe representation

    Return List with words with representation in GloVe
    '''
    vocab_emb = []
    for word in dictionary_list:
        # revisa si la palabra está en la matriz de glove
        #embeddings_vector = embeddings_dict_w2v.get(word)
        #if embeddings_vector is not None:
        vocab_emb.append(word)
    return vocab_emb


def get_words(fdist_doc):
    '''
    For each user we have the dictionary of frequency of their words
    we have a reduced list by check is their words have a representation
    in GloVe

    Return: List
    '''
    words_doc = []
    for i, word in fdist_doc:
        #if word.isnumeric() == False:
        #   embeddings_vector = embeddings_dict_w2v.get(word)
        #  if embeddings_vector is not None:
         words_doc.append(word)

    return words_doc


def get_dictionary(name_dic):
    if name_dic == 'dict29':
        vocab = []
        with open('/home/est_posgrado_maria.garcia/Proyecto_tecnologico/vocabulary_ex.txt', 'r') as f:
            for twitt in f:
                vocab.append(twitt[:-1])
            f.close()
        vocab_plus = set(vocab)
        vocab_emb = get_words_dic(vocab_plus)
        value = [x for x in range(len(vocab_emb))]
        dictionary = dict(zip(value, set(vocab_emb)))  # nuevo diccionario

    elif name_dic == 'dict29_ex':
        vocab = []
        with open('/home/est_posgrado_maria.garcia/Proyecto_tecnologico/vocabulary_extending.txt', 'r') as f:
            for twitt in f:
                vocab.append(twitt[:-1])
            f.close()
        vocab_plus = set(vocab)
        vocab_emb = get_words_dic(vocab_plus)
        value = [x for x in range(len(vocab_emb))]
        dictionary = dict(zip(value, set(vocab_emb)))  # nuevo diccionario

    elif name_dic == 'dict51':
        vocab = []
        with open('/home/est_posgrado_maria.garcia/Proyecto_tecnologico/vocab_extending51.txt', 'r') as f:
            for twitt in f:
                vocab.append(twitt[:-1])
            f.close()
        vocab_plus = set(vocab)
        vocab_emb = get_words_dic(vocab_plus)
        value = [x for x in range(len(vocab_emb))]
        dictionary = dict(zip(value, set(vocab_emb)))  # nuevo diccionario

    elif name_dic == 'dict51_ex':
        vocab = []
        with open('/home/est_posgrado_maria.garcia/Proyecto_tecnologico/vocab_ex51.txt', 'r') as f:
            for twitt in f:
                vocab.append(twitt[:-1])
            f.close()
        vocab_plus = set(vocab)
        vocab_emb = get_words_dic(vocab_plus)
        value = [x for x in range(len(vocab_emb))]
        dictionary = dict(zip(value, set(vocab_emb)))  # nuevo diccionario

    return dictionary


def get_vec_rep(words_doc, dictionary, option, norm = False, param_norm = {}):
    words_user = np.zeros((len(words_doc),300) , dtype=float)

    dictionary_vec = np.zeros((len(set(dictionary)),300) ,dtype=float)
    for i in range(words_user.shape[0]):
        w1 = words_doc[i]
        
        #words_user[i] = embeddings_dict_w2v[w1]
        if option == 1: 
        	words_user[i] = model_anxia.wv[w1]
        if option == 2: 
        	words_user[i] = model_dep.wv[w1]
        if option == 3: 
        	words_user[i] = model_pre.get_word_vector(w1)

    for i in range(dictionary_vec.shape[0]):
        w1 = dictionary[i]
        #dictionary_vec[i] = embeddings_dict_w2v[w1]
        if option == 1: 
        	dictionary_vec[i] = model_anxia.wv[w1]
        if option == 2: 
        	dictionary_vec[i] = model_anxia.wv[w1]
        if option == 3: 
        	dictionary_vec[i] = model_pre.get_word_vector(w1)

    similarity_vocab = sklearn.metrics.pairwise.cosine_similarity(words_user, dictionary_vec)

    if norm == True:
        norm_type = param_norm['type']
        if norm_type == 'avg':
            vec_rep = np.sum(similarity_vocab, axis=0) /len(words_doc)
            
        elif norm_type == 'norm':
            vec_rep = np.sum(similarity_vocab, axis=0)
            vec_rep =  preprocessing.normalize([vec_rep])
            
    else:
        vec_rep = np.sum(similarity_vocab, axis=0)
       
    return vec_rep



def classificator_vectors(data, test, name_dic, option,norm_vec=False, norm_data=False, sub_param={}, param_norm={}):
    num_doc = len(data)  # número de sujetos
    num_test = len(test)

    dictionary = get_dictionary(name_dic)
    

    X_train = np.zeros((num_doc, len(dictionary)), dtype=float)
    X_test = np.zeros((num_test, len(dictionary)), dtype=float)  # matriz tipo document-term
    data = normalize(data)
    test = normalize(test)

    for i in range(num_doc):
        doc = data[i]
        corpus_palabras = tokenizer.tokenize(doc.lower())
        fdist = nltk.FreqDist(corpus_palabras)
        v = sortFreqDict(fdist)
        words_doc = get_words(v)  # lista v reducida con las palabras que si tienen representación GloVe
        word_repre_user = get_vec_rep(words_doc, dictionary, option,norm=norm_vec,
                                      param_norm=param_norm)
        X_train[i] = word_repre_user

    print("Vectorization for train data: Done")
    for i in range(num_test):
        doc = test[i]
        corpus_palabras = tokenizer.tokenize(doc.lower())
        fdist = nltk.FreqDist(corpus_palabras)
        v = sortFreqDict(fdist)
        words_doc = get_words(v)  # lista v reducida con las palabras que si tienen representación GloVe
        word_repre_user = get_vec_rep(words_doc, dictionary, option,norm=norm_vec,
                                      param_norm=param_norm)
        X_test[i] = word_repre_user

    print("Vectorization for test data: Done")

    if norm_data == True:
        norm_type = sub_param['type']
        if norm_type == 'normalize':
            X_train = preprocessing.normalize(X_train)
            X_test = preprocessing.normalize(X_test)

        elif norm_type == 'standard':
            X_train = StandardScaler().fit_transform(X_train)
            X_test = StandardScaler().fit_transform(X_test)

    return X_train, X_test


def run_exp_anorexia(num_exp, train_data, test_data, y_train, y_test,name_dict, option,norm_data, norm_vec, sub_param = {}, param_norm={}):
    parameters = {'C': [.05, .12, .25, .5, 1, 2, 4]}
    seed_val = 42
    np.random.seed(seed_val)	

    x_train, x_test = classificator_vectors(train_data, test_data, name_dic=name_dict, option = option,norm_vec=norm_vec,
                                              norm_data=norm_data, sub_param=sub_param,param_norm = param_norm)

    svr = svm.LinearSVC(class_weight='balanced', dual = False)
    grid_anorexia = GridSearchCV(estimator=svr, param_grid=parameters, n_jobs=8, scoring='f1_macro', cv=5)
    grid_anorexia.fit(x_train, y_train)

    y_pred = grid_anorexia.predict(x_test)
    a1 = grid_anorexia.best_params_

    # print("Best paramter: ", a1)

    p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', pos_label=1)
    result_name = 'result_anorexia_vec_' + str(num_exp) + '.txt'
    path_name = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Anorexia_vec' + '/' + result_name


    with open(path_name, "w") as f:
        f.write("Experimento de anorexia número: " + str(num_exp) + '\n')
        f.write("Confusion matrix: \n")
        f.write(str(confusion_matrix(y_test, y_pred)))

        f.write('Metrics classification \n')
        f.write(str(metrics.classification_report(y_test, y_pred)))

        f.write('Best parameter:\n')
        f.write(str(a1))
        f.write('\n')
        # f.write('Precission, recall and F1-score:')
        # f.write(p,r,f)
        f.close()


    return f1_score(y_test, y_pred),a1


def run_exp_dep(num_exp, train_data, test_data, y_train, y_test,name_dict, option,norm_data, norm_vec, sub_param = {}, param_norm={}):
    parameters = {'C': [.05, .12, .25, .5, 1, 2, 4]}
    seed_val = 42
    np.random.seed(seed_val)	

    x_train, x_test = classificator_vectors(train_data, test_data, name_dic=name_dict, option = option,norm_vec=norm_vec,
                                              norm_data=norm_data, sub_param=sub_param,param_norm = param_norm)

    svr = svm.LinearSVC(class_weight='balanced')
    grid_anorexia = GridSearchCV(estimator=svr, param_grid=parameters, n_jobs=8, scoring='f1_macro', cv=5)
    grid_anorexia.fit(x_train, y_train)

    y_pred = grid_anorexia.predict(x_test)
    a1 = grid_anorexia.best_params_

    # print("Best paramter: ", a1)

    p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', pos_label=1)
    result_name = 'result_depression_vec_' + str(num_exp) + '.txt'
    path_name = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Depresion_vec' + '/' + result_name


    with open(path_name, "w") as f:
        f.write("Experimento de depresión número: " + str(num_exp) + '\n')
        f.write("Confusion matrix: \n")
        f.write(str(confusion_matrix(y_test, y_pred)))

        f.write('Metrics classification \n')
        f.write(str(metrics.classification_report(y_test, y_pred)))

        f.write('Best parameter:\n')
        f.write(str(a1))
        f.write('\n')
        # f.write('Precission, recall and F1-score:')
        # f.write(p,r,f)
        f.close()

    return f1_score(y_test, y_pred), a1
