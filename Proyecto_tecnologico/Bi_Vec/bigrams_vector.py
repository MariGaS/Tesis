from gensim.parsing.preprocessing import remove_stopwords
from nltk import TweetTokenizer
import nltk
import numpy as np
import re
from gensim.models import FastText
from nltk.util import ngrams
import collections
import sklearn
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
tokenizer = TweetTokenizer()

print('Load_anxia')
model_anxia = FastText.load('Model/anxiety.model')
print('Load_dep')
model_dep = FastText.load('Model/depresion.model')


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

def get_corpus(text):
    corpus_palabras = []
    for doc in text:
        corpus_palabras += tokenizer.tokenize(doc)
    return corpus_palabras


# from https://www.kaggle.com/code/rtatman/tutorial-getting-n-grams
def get_bigrams(data, num_bigrams):
    corpus = get_corpus(data)
    esBigrams = ngrams(corpus, 2)
    esBigramFreq = collections.Counter(esBigrams)
    l = esBigramFreq.most_common(num_bigrams)

    bigrams = []
    for i in range(len(l)):
        w1 = l[i][0][0]
        w2 = l[i][0][1]

        w = w1 + '_' + w2
        bigrams.append(w)

    return bigrams


def get_words(fdist_doc):
    words_doc = []
    for i, word in fdist_doc:
        words_doc.append(word)

    return words_doc


def get_dic_bigram(fdist1, bigrams):
    words1 = get_words(fdist1)
    words = words1 + bigrams
    vocab_plus = set(words)

    value = [x for x in range(len(vocab_plus))]
    dictionary = dict(zip(value, vocab_plus))  # nuevo diccionario

    return dictionary


def get_fuzzy_rep(words_doc, dictionary, option,epsilon):
    words_user = np.zeros((len(words_doc), 300), dtype=float)
    dictionary_vec = np.zeros((len(set(dictionary)), 300), dtype=float)

    for i in range(words_user.shape[0]):
        w1 = words_doc[i]
        if option == 1:
            words_user[i] = model_anxia.wv[w1]
        else:
            words_user[i] = model_dep.wv[w1]
    for i in range(dictionary_vec.shape[0]):
        w1 = dictionary[i]
        if option == 1:
            dictionary_vec[i] = model_anxia.wv[w1]
        else:
            dictionary_vec[i] = model_dep.wv[w1]

    similarity_vocab = sklearn.metrics.pairwise.cosine_similarity(words_user, dictionary_vec)
    # vector de representación
    vec_representation = np.count_nonzero(similarity_vocab > epsilon, axis=0)

    return vec_representation

def get_bigrams_for_doc(data, num_bigrams):
    corpus_palabras = []
    corpus_palabras += tokenizer.tokenize(data)
    esBigrams = ngrams(corpus_palabras, 2)
    esBigramFreq = collections.Counter(esBigrams)
    l = esBigramFreq.most_common(num_bigrams)

    bigrams = []
    for i in range(len(l)):
        w1 = l[i][0][0]
        w2 = l[i][0][1]

        w = w1 + '_' + w2
        bigrams.append(w)

    return bigrams

def get_fdist(text, num_feat):
    corpus_palabras = []
    for doc in text:
        corpus_palabras += tokenizer.tokenize(doc)
    fdist = nltk.FreqDist(corpus_palabras)
    fdist = sortFreqDict(fdist)
    fdist = fdist[:num_feat]
    return fdist


def classificator_pos_neg(pos_data, neg_data, test, num_feat, num_bigrams,chose,tau, remove_stop=False):
    if remove_stop == True:
        print("Quitando stopwords")
        pos_data = remove_stop_list(pos_data)
        neg_data = remove_stop_list(neg_data)
        test = remove_stop_list(test)
        print("Finalizado")
    pos_data = normalize(pos_data)
    neg_data = normalize(neg_data)
    test = normalize(test)

    num_test = len(test)

    fdist_pos = get_fdist(pos_data, num_feat)
    fdist_neg = get_fdist(neg_data, num_feat)

    bigrams1 = get_bigrams(pos_data, num_bigrams)
    bigrams2 = get_bigrams(neg_data, num_bigrams)
    dictionary1 = get_dic_bigram(fdist_pos, bigrams1)
    dictionary2 = get_dic_bigram(fdist_neg, bigrams2)


    X_test1 = np.zeros((num_test, len(dictionary1)), dtype=float)  # matriz tipo document-term
    X_test2 = np.zeros((num_test, len(dictionary2)), dtype=float)
    print(len(dictionary1))
    print(len(dictionary2))
    print("Iniciando vectorización positiva")
    for i in range(num_test):
        doc = test[i]
        corpus_palabras = tokenizer.tokenize(doc.lower())
        fdist = nltk.FreqDist(corpus_palabras)
        v = sortFreqDict(fdist)
        words_doc = get_words(v)
        bigrams_test1 = get_bigrams_for_doc(doc, num_bigrams)
        words_doc = words_doc + bigrams_test1
        word_repre_user = get_fuzzy_rep(words_doc, dictionary1, option=chose, epsilon=tau)
        X_test1[i] = word_repre_user
    print('end')
    print("Iniciando vectorización negativa")
    for i in range(num_test):
        doc = test[i]
        corpus_palabras = tokenizer.tokenize(doc.lower())
        fdist = nltk.FreqDist(corpus_palabras)
        v = sortFreqDict(fdist)
        words_doc = get_words(v)
        bigrams_test2 = get_bigrams_for_doc(doc, num_bigrams)
        words_doc = words_doc + bigrams_test2
        word_repre_user = get_fuzzy_rep(words_doc, dictionary2,option=chose, epsilon=tau)
        X_test2[i] = word_repre_user
    print('end')
    X_test1 = np.sum(X_test1, axis=1)
    X_test2 = np.sum(X_test2, axis=1)

    results = np.zeros((num_test), dtype=int)
    for i in range(num_test):
        if X_test1[i] > X_test2[i]:
            results[i] = 1
        else:
            results[i] = 0
    return results, dictionary1, dictionary2, bigrams1, bigrams2


def run_exp_anxia_bi(num_exp, pos_data, neg_data, test_data, test_labels, num_feat, num_bigrams,tau, remove_stop):
    results, dic1, dic2, bi1, bi2 = classificator_pos_neg(pos_data, neg_data, test_data,num_feat, num_bigrams, tau=tau,chose=1,
                                                 remove_stop=remove_stop)


    result_name = 'result_anxia_bi' + str(num_exp) + '.txt'
    path_name = 'Results/Anorexia_bi' + '/' + result_name

    with open(path_name, "w") as f:
        f.write("Experimento de anorexia número: " + str(num_exp) + '\n')
        f.write("Confusion matrix: \n")
        f.write(str(confusion_matrix(test_labels, results)) + '\n')

        f.write('Metrics classification \n')
        f.write(str(metrics.classification_report(test_labels, results)) + '\n')

        f.close()

    path_name_dic1 = 'Results/Anorexia_bi' + '/dictionary1_' + str(num_exp) + '.txt'
    path_name_dic2 = 'Results/Anorexia_bi' + '/dictionary2_' + str(num_exp) + '.txt'

    with open(path_name_dic1, "w") as f:
        for i in range(len(dic1)):
            f.write(str(dic1[i]) + '\n')
        f.close()

    with open(path_name_dic2, "w") as f:
        for i in range(len(dic2)):
            f.write(str(dic2[i]) + '\n')
        f.close()

    path_name_bi1 = 'Results/Anorexia_bi' + '/bigram1_' + str(num_exp) + '.txt'
    path_name_bi2 = 'Results/Anorexia_bi' + '/bigram2_' + str(num_exp) + '.txt'

    with open(path_name_bi1, "w") as f:
        for i in range(len(bi1)):
            f.write(str(bi1[i]) + '\n')
        f.close()

    with open(path_name_bi2, "w") as f:
        for i in range(len(bi2)):
            f.write(str(bi2[i]) + '\n')
        f.close()

    return f1_score(test_labels, results)


def run_exp_dep_bi(num_exp, pos_data, neg_data, test_data, test_labels, num_feat, num_bigrams,tau, remove_stop):
    results, dic1, dic2, bi1,bi2= classificator_pos_neg(pos_data, neg_data, test_data,num_feat,num_bigrams, tau=tau,chose=2,
                                                 remove_stop=remove_stop)


    result_name = 'result_dep_bi' + str(num_exp) + '.txt'
    path_name = 'Results/Depresion_bi' + '/' + result_name

    with open(path_name, "w") as f:
        f.write("Experimento de depresión número: " + str(num_exp) + '\n')
        f.write("Confusion matrix: \n")
        f.write(str(confusion_matrix(test_labels, results)) + '\n')

        f.write('Metrics classification \n')
        f.write(str(metrics.classification_report(test_labels, results)) + '\n')

        f.close()

    path_name_dic1 = 'Results/Depresion_bi' + '/dictionary1_' + str(num_exp) + '.txt'
    path_name_dic2 = 'Results/Depresion_bi' + '/dictionary2_' + str(num_exp) + '.txt'

    with open(path_name_dic1, "w") as f:
        for i in range(len(dic1)):
            f.write(str(dic1[i]) + '\n')
        f.close()

    with open(path_name_dic2, "w") as f:
        for i in range(len(dic2)):
            f.write(str(dic2[i]) + '\n')
        f.close()
    path_name_bi1 = 'Results/Depresion_bi' + '/bigram1_' + str(num_exp) + '.txt'
    path_name_bi2 = 'Results/Depresion_bi' + '/bigram2_' + str(num_exp) + '.txt'

    with open(path_name_bi1, "w") as f:
        for i in range(len(bi1)):
            f.write(str(bi1[i]) + '\n')
        f.close()

    with open(path_name_bi2, "w") as f:
        for i in range(len(bi2)):
            f.write(str(bi2[i]) + '\n')
        f.close()
    return f1_score(test_labels, results)




