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
tokenizer = TweetTokenizer()

from gensim.models import FastText
import gensim.downloader
from gensim.test.utils import get_tmpfile, datapath
#from gensim.models import fasttext
import fasttext
import fasttext.util


print('Load_anxia')
model_anxia = FastText.load('/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Model/anxiety.model')
print('Load_dep')
model_dep   = FastText.load('/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Model/depresion.model')
print('Load pretrained')
model_pre = fasttext.load_model('/home/est_posgrado_maria.garcia/Proyecto_tecnologico/cc.en.300.bin')
print('Load emo')
model_emo = FastText.load('/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Model/emotions.model')
#model_pre = fasttext.load_facebook_vectors(datapath("./wiki.en/wiki.en.bin"))

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


def get_words(fdist_doc):
    '''
    For each user we have the dictionary of frequency of their words
    we have a reduced list by check is their words have a representation
    in GloVe

    Return: List
    '''
    words_doc = []
    for i, word in fdist_doc:
        # if word.isnumeric() == False:
        # embeddings_vector = model.wv[word]
        # if embeddings_vector is not None:
        words_doc.append(word)

    return words_doc


def sortFreqDict(freqdict):
    aux = [(freqdict[key], key) for key in freqdict]
    aux.sort()
    aux.reverse()
    return aux


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
        if option == 4:
            words_user[i] = model_emo.wv[w1]           
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
    similarity_vocab = sklearn.metrics.pairwise.cosine_similarity(words_user, dictionary_vec)
    # vector de representación
    vec_representation = np.count_nonzero(similarity_vocab > epsilon, axis=0)

    return vec_representation


def remove_stop_list(document):
    document = [remove_stopwords(x) for x in document]

    return document

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
        path_name = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Anorexia/features_words_3.txt'
        dictionary = get_words_dictionary(path_name)

    elif name_dic == 'dict2':
        path_name = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Anorexia/features_words_4.txt'
        dictionary = get_words_dictionary(path_name)

    elif name_dic == 'dict3':
        path_name = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Anorexia/features_words_6.txt'
        dictionary = get_words_dictionary(path_name)

    elif name_dic == 'dict4':
        path_name = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Anorexia/features_words_8.txt'
        dictionary = get_words_dictionary(path_name)

    elif name_dic == 'dict5':
        path_name = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Depresion/features_words_15.txt' #1
        dictionary = get_words_dictionary(path_name)

    elif name_dic == 'dict6':
        path_name = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Depresion/features_words_1.txt'#3
        dictionary = get_words_dictionary(path_name)

    elif name_dic == 'dict7':
        path_name = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Depresion/features_words_5.txt' #4
        dictionary = get_words_dictionary(path_name)

    elif name_dic == 'dict8':
        path_name = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Depresion/features_words_3.txt'#5
        dictionary = get_words_dictionary(path_name)

    return dictionary


def classificator_fuzzy(data, test, name_dic, tau, chose,remove_stop=False, norm_data=False, sub_param={}):
    # número de sujetos
    num_doc = len(data)
    num_test = len(test)

    # obtenemos dictionario de features
    dictionary = get_dictionary(name_dic)
    #print(len(dictionary))

    X_train = np.zeros((num_doc, len(dictionary)), dtype=float)
    X_test = np.zeros((num_test, len(dictionary)), dtype=float)  # matriz tipo document-term

    if remove_stop == True:
        print("Quitando stopwords")
        data = remove_stop_list(data)
        test = remove_stop_list(test)
        print("Terminado")

    data = normalize(data)
    test = normalize(test)

    for i in range(num_doc):
        doc = data[i]
        corpus_palabras = tokenizer.tokenize(doc.lower())
        fdist = nltk.FreqDist(corpus_palabras)
        v = sortFreqDict(fdist)
        words_doc = get_words(v)
        word_repre_user = get_fuzzy_rep(words_doc, dictionary, option=chose, epsilon=tau)
        X_train[i] = word_repre_user

    print("Vectorization for train data: Done")
    for i in range(num_test):
        doc = test[i]
        corpus_palabras = tokenizer.tokenize(doc.lower())
        fdist = nltk.FreqDist(corpus_palabras)
        v = sortFreqDict(fdist)
        words_doc = get_words(v)
        word_repre_user = get_fuzzy_rep(words_doc, dictionary, option = chose,epsilon=tau)
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

    return X_train, X_test, len(dictionary)

def run_exp_anorexia(num_exp, train_data, test_data, y_train, y_test, chose,tau, remove_stop, name_dict, norm_data, sub_param = {}):
    seed_val = 42
    np.random.seed(seed_val)
    parameters = {'C': [.05, .12, .25, .5, 1, 2, 4]}
    x_train, x_test, l_d = classificator_fuzzy(train_data, test_data,tau=tau,chose=chose,remove_stop=remove_stop,
                                               name_dic=name_dict, norm_data=norm_data, sub_param=sub_param)

    svr = svm.LinearSVC(class_weight='balanced', dual=False)
    grid_anorexia = GridSearchCV(estimator=svr, param_grid=parameters, n_jobs=8, scoring='f1_macro', cv=5)
    grid_anorexia.fit(x_train, y_train)

    y_pred = grid_anorexia.predict(x_test)
    a1 = grid_anorexia.best_params_

    p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', pos_label=1)
    result_name = 'result_anxia_fuzzy' + str(num_exp) + '.txt'
    path_name = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Anorexia_fuzzy' + '/' + result_name

    with open(path_name, "w") as f:
        f.write("Experimento de anorexia número: " + str(num_exp) + '\n')
        f.write("Confusion matrix: \n")
        f.write(str(confusion_matrix(y_test, y_pred)) + '\n')

        f.write('Metrics classification \n')
        f.write(str(metrics.classification_report(y_test, y_pred)) + '\n')

        f.write('Best parameter:\n')
        f.write(str(a1))
        f.write('\n')
        # f.write('Precission, recall and F1-score:')
        # f.write(p,r,f)
        f.close()


    return f1_score(y_test, y_pred),a1, l_d


def run_exp_depresion(num_exp, train_data, test_data, y_train, y_test, chose, tau, remove_stop, name_dict, norm_data, sub_param={}):
    parameters = {'C': [.05, .12, .25, .5, 1, 2, 4]}
    seed_val = 42
    np.random.seed(seed_val)
    x_train, x_test, l_d = classificator_fuzzy(train_data, test_data, tau=tau, chose=chose, remove_stop=remove_stop,
                                               name_dic=name_dict, norm_data=norm_data,sub_param=sub_param)

    svr = svm.LinearSVC(class_weight='balanced', dual=False)
    grid = GridSearchCV(estimator=svr, param_grid=parameters, n_jobs=8, scoring='f1_macro', cv=5)
    grid.fit(x_train, y_train)

    y_pred = grid.predict(x_test)
    a1 = grid.best_params_

    p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', pos_label=1)
    result_name = 'result_dep_fuzzy' + str(num_exp) + '.txt'
    path_name = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Depresion_fuzzy' + '/' + result_name

    with open(path_name, "w") as f:
        f.write("Experimento de depresión número: " + str(num_exp) + '\n')
        f.write("Confusion matrix: \n")
        f.write(str(confusion_matrix(y_test, y_pred)) + '\n')

        f.write('Metrics classification \n')
        f.write(str(metrics.classification_report(y_test, y_pred)) + '\n')

        f.write('Best parameter:\n')
        f.write(str(a1))
        f.write('\n')
        f.close()

    return f1_score(y_test, y_pred), a1, l_d

