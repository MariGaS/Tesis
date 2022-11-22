from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import scipy.sparse as sp
from time import time
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
import re
from nltk import TweetTokenizer
import nltk
from sklearn.metrics import f1_score
import ekphrasis
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from sklearn.naive_bayes import MultinomialNB
import emoji
import string
import random 
tokenizer = TweetTokenizer()
# nltk.download('punkt')
stemmer = nltk.stem.porter.PorterStemmer()
tt = nltk.tokenize.TweetTokenizer()
hashtag_segmenter = TextPreProcessor(segmenter="twitter", unpack_hashtags=True)
punct_set = set(string.punctuation + '''…'"`’”“''')
##  constructing BOW ##

tt = nltk.tokenize.TweetTokenizer()


def emoji_split(e, joiner = '\u200d', 
                variation_selector=b'\xef\xb8\x8f'.decode('utf-8'),
                return_special_chars = False):
  parts = []
  for part in e:
    if part == joiner:
      if return_special_chars:
        parts.append(":joiner:")
    elif part == variation_selector:
      if return_special_chars:
        parts.append(":variation:")
    else:
      parts.append(part)
  return parts

def my_preprocessor(text,
                         hashtag_segmenter=hashtag_segmenter,
                         punct_set=punct_set):

  # lowercase
  text = text.lower()
  # tokenize
  tokens = tokenizer.tokenize(text)
  updated_tokens = []
  # set different behavior for different kinds of tokens
  for t in tokens:
    # split emoji into components
    if t in emoji.UNICODE_EMOJI:
      updated_tokens += emoji_split(t)
    # keep original hashtags and split them into words
    elif t.startswith('#'):
      updated_tokens += [t]
      updated_tokens += hashtag_segmenter.pre_process_doc(t).split()
    # remove user mentions
    elif t.startswith('@'):
      pass
    # remove urls because we will get them from the expanded_urls field anyways
    # and remove single punctuation markers
    elif t.startswith('http') or t in punct_set:
      pass
    # skip stopwords and empty strings, include anything else
    elif t:
      updated_tokens += [t]
  return ' '.join(updated_tokens)

def building_bow(data, labels, ntrain, min=1, max=1, num_feat=3000, binary=False, tf=False, tf_idf=False,
                 stopwords=False, tf_stop=False, verbose=True, analyzer_char=False):
    documents = data


    # split the data
    # x_train, x_val, y_train, y_val = train_test_split(documents, labels, test_size= split, random_state=42)
    x_train = documents[:ntrain]
    x_val = documents[ntrain:]
    y_train = labels[:ntrain]
    t_initial = time()

    analyzer_type = 'word'  # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    if analyzer_char:
        analyzer_type = 'char'
        
    if binary:
        vectorizer = CountVectorizer(ngram_range=(min, max), binary=True, analyzer=analyzer_type)
        #features_name = vectorizer.get_feature_names_out()
        #stop_words= []
    elif stopwords:
        vectorizer = TfidfVectorizer(ngram_range=(min, max), stop_words='english',
                                     analyzer=analyzer_type, sublinear_tf=True)
        #stop_words= vectorizer.get_stop_words()
        #features_name = vectorizer.get_feature_names_out()
    elif tf:
        vectorizer = TfidfVectorizer(ngram_range=(min, max),
                                     analyzer=analyzer_type, sublinear_tf=True, use_idf=False)
        #features_name = vectorizer.get_feature_names_out()
        #stop_words = []
    elif tf_stop:
        vectorizer = TfidfVectorizer(ngram_range=(min, max), stop_words='english',
                                     analyzer=analyzer_type, sublinear_tf=True, use_idf=False)
        #features_name = vectorizer.get_feature_names_out()
        #stop_words = vectorizer.get_stop_words()

    elif tf_idf:
        vectorizer = TfidfVectorizer(ngram_range=(min, max), sublinear_tf=True, analyzer=analyzer_type)
        #features_name = vectorizer.get_feature_names_out()
        #stop_words = []


    X_train = vectorizer.fit_transform(x_train)
    X_val = vectorizer.transform(x_val)

    if verbose:
        print("done in %fs" % (time() - t_initial), X_train.shape, X_val.shape)

    y = np.array(y_train)

    if num_feat < X_train.shape[1]:
        t0 = time()
        ch2 = SelectKBest(chi2, k=num_feat)
        X_train = ch2.fit_transform(X_train, y)
        X_test = ch2.transform(X_val)
        assert sp.issparse(X_train)
        stop_words    = vectorizer.get_stop_words()
        feature_names = vectorizer.get_feature_names_out()
        feature_names = [feature_names[i] for i
                         in ch2.get_support(indices=True)]
    else:
        ch2 = 0
        X_test = X_val
	
    if verbose:
        print("Extracting best features by a chi-squared test.. ", X_train.shape, X_test.shape)
    return X_train, y, X_test, ch2, feature_names


def run_experiment_anorexia(data, label, ntrain, test_labels, 
num_exp, min, max, num_feat, weight, classifier):

    seed_val = 42
    np.random.seed(seed_val)
    if weight == 'binary':
        x_train, y, x_test, chi, features_name = building_bow(data=data, labels=label, ntrain=ntrain, min=min, max=max,
                                                              num_feat=num_feat,
                                                              binary=True, verbose=False)
    elif weight == 'tf_stop':
        x_train, y, x_test, chi, features_name = building_bow(data=data, labels=label, ntrain=ntrain, min=min, max=max,
                                                              num_feat=num_feat,
                                                              tf=True, stopwords=True, verbose=False)
    elif weight == 'tf':
        x_train, y, x_test, chi, features_name = building_bow(data=data, labels=label, ntrain=ntrain, min=min, max=max,
                                                              num_feat=num_feat,
                                                              tf=True,  verbose=False)
    elif weight == 'stopwords':
        x_train, y, x_test, chi, features_name = building_bow(data=data, labels=label, ntrain=ntrain, min=min, max=max,
                                                              num_feat=num_feat,
                                                              stopwords=True,verbose=False)
    elif weight == 'tf_idf':
        x_train, y, x_test, chi, features_name = building_bow(data=data, labels=label, ntrain=ntrain, min=min, max=max,
                                                              num_feat=num_feat,
                                                              tf_idf=True,  verbose=False)
    for i in range(5):
        seeds = []
        f_scores = []
        seed_value = random.randrange(1000)
        np.random.seed(seed_value)
        seeds.append(seed_value)    
        if classifier == 'svm':
            parameters = {'C': [.05, .12, .25, .5, 1, 2, 4]}
            svr = svm.LinearSVC(class_weight='balanced')
            grid_dep = GridSearchCV(estimator=svr, param_grid=parameters, n_jobs=8, scoring='f1_macro', cv=5)       
            grid_dep.fit(x_train, y)
            y_pred = grid_dep.predict(x_test)
            a1 = grid_dep.best_params_

        if classifier == 'NB':
            model = MultinomialNB()
            model.fit(x_train, y)
            y_pred = model.predict(x_test)
            a1 = 'None'
        
        f1 = f1_score(test_labels, y_pred)
        f_scores.append(f1)

        f = open('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Baseline/test_anxia.txt','a')
        f.write('\n' + str(num_exp) + ',' + str(num_feat) + ',' + str(min) +',' + str(max) +',' + str(weight) + ',' + str(classifier)+ 
                                '.' + str(seed_value) + ',' + str(f1) + ',' + str(a1))
        f.close() 
        f.close()
        if i == 4:

            f = open('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Baseline/anxia_var.txt','a')
            f.write('\n' + str(num_exp) + ',' + str(seeds) + ',' +  str(np.var(f_scores)) + ',' + str(np.std(f_scores))) 
            f.close()





def run_experiment_depression(data, label, ntrain, test_labels, 
num_exp, min, max, num_feat, weight, classifier):
    seed_val = 42
    np.random.seed(seed_val)
    if weight == 'binary':
        x_train, y, x_test, chi, features_name= building_bow(data=data, labels=label, ntrain=ntrain, min=min, max=max, num_feat=num_feat,
                                          binary=True, verbose=False)
    elif weight == 'tf_stop':
        x_train, y, x_test, chi, features_name= building_bow(data=data, labels=label, ntrain=ntrain, min=min, max=max, num_feat=num_feat,
                                          tf=True, stopwords=True, verbose=False)
    elif weight == 'tf':
        x_train, y, x_test, chi, features_name= building_bow(data=data, labels=label, ntrain=ntrain, min=min, max=max, num_feat=num_feat,
                                          tf=True, verbose=False)
    elif weight == 'stopwords':
        x_train, y, x_test, chi, features_name = building_bow(data=data, labels=label, ntrain=ntrain, min=min, max=max, num_feat=num_feat,
                                          stopwords=True, verbose=False)
    elif weight == 'tf_idf':
        x_train, y, x_test, chi, features_name = building_bow(data=data, labels=label, ntrain=ntrain, min=min, max=max, num_feat=num_feat,
                                          tf_idf=True,  verbose=False)
    else:
        print("Include the name of the weighting")
        exit()
       
    for i in range(5):
        seeds = []
        f_scores = []
        seed_value = random.randrange(1000)
        np.random.seed(seed_value)
        seeds.append(seed_value)    
        if classifier == 'svm':
            parameters = {'C': [.05, .12, .25, .5, 1, 2, 4]}
            svr = svm.LinearSVC(class_weight='balanced')
            grid_dep = GridSearchCV(estimator=svr, param_grid=parameters, n_jobs=8, scoring='f1_macro', cv=5)       
            grid_dep.fit(x_train, y)
            y_pred = grid_dep.predict(x_test)
            a1 = grid_dep.best_params_

        if classifier == 'NB':
            model = MultinomialNB()
            model.fit(x_train, y)
            y_pred = model.predict(x_test)
            a1 = 'None'
        
        f1 = f1_score(test_labels, y_pred)
        f_scores.append(f1)

        f = open('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Baseline/test_dep.txt','a')

        f.write('\n' + str(num_exp) + ',' + str(num_feat) + ',' + str(min) +',' + str(max) +',' + str(weight) + ',' + str(classifier)+ 
                                '.' + str(seed_value) + ',' + str(f1) + ',' + str(a1))
        f.close()
        if i == 4:

            f = open('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Baseline/dep_var.txt','a')
            f.write('\n' + str(num_exp) + ',' + str(seeds) + ',' +  str(np.var(f_scores)) + ',' + str(np.std(f_scores))) 
            f.close()


