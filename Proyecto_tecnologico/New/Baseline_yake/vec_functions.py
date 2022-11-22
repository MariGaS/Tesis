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
import pickle
from nltk.util import ngrams
import itertools


tokenizer = TweetTokenizer()
# nltk.download('punkt')
stemmer = nltk.stem.porter.PorterStemmer()
tt = nltk.tokenize.TweetTokenizer()
hashtag_segmenter = TextPreProcessor(segmenter="twitter", unpack_hashtags=True)
punct_set = set(string.punctuation + '''…'"`’”“''')
##  constructing BOW ##

tt = nltk.tokenize.TweetTokenizer()
text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'date', 'hashtag'],
    # terms that will be annotated
    annotate={},
    fix_html=True,  # fix HTML tokens
    
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter", 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    corrector="twitter", 
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)

# adding a simple wrapper that we can use later if we want
def ekphrasis_processor(text, text_processor=text_processor):
  return ' '.join(text_processor.pre_process_doc(text))

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

##                                         USER VERSION OF THE DICTIONARIES OF THE USERS                        #
#---------------------------ANOREXIA-------------------#
#version 1
user_pos_anxia1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/anxia_pos_ver3'
user_neg_anxia1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/anxia_neg_ver3'
#version 2
user_pos_anxia2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/anxia_pos_ver4'
user_neg_anxia2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/anxia_neg_ver4'

user_pos_anxia3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/anxia_pos_ver5'
user_neg_anxia3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/anxia_neg_ver5'
#version 2
user_pos_anxia4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/anxia_pos_ver6'
user_neg_anxia4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/anxia_neg_ver6'

#---------------------------DEPRESSION-----------------#
#version 1
user_pos_dep1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_pos_ver3'
user_neg_dep1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_neg_ver3'
#version 2
user_pos_dep2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_pos_ver4'
user_neg_dep2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_neg_ver4'

user_pos_dep3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_pos_ver5'
user_neg_dep3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_neg_ver5'
#version 2
user_pos_dep4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_pos_ver6'
user_neg_dep4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_neg_ver6'

##                                                   CONCATANTE ALL THE TEXT 
#version 1
con_pos_anxia1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_pos_ver3'
con_neg_anxia1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_neg_ver3'
#version 2
con_pos_anxia2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_pos_ver4'
con_neg_anxia2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_neg_ver4'

con_pos_anxia3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_pos_ver5'
con_neg_anxia3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_neg_ver5'
#version 2
con_pos_anxia4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_pos_ver6'
con_neg_anxia4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_neg_ver6'

#---------------------------DEPRESSION-----------------#
#version 1
con_pos_dep1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_pos_ver3'
con_neg_dep1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_neg_ver3'
#version 2
con_pos_dep2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_pos_ver4'
con_neg_dep2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_neg_ver4'

con_pos_dep3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_pos_ver5'
con_neg_dep3= '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_neg_ver5'
#version 2
con_pos_dep4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_pos_ver6'
con_neg_dep4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_neg_ver6'

def get_bigrams(corpus):

    dictionary = ngrams(corpus, 2)
    dictionary = list(dictionary)
    bigrams = []
    for i in range(len(dictionary)):
        w1 = dictionary[i][0][0]
        w2 = dictionary[i][0][1]

        w = w1 + ' ' + w2
        bigrams.append(w)

    return list(dictionary)



def building_bow(data, labels, ntrain, d1, d2, dif, con, min=1, max=1, num_feat1=100, num_feat2 = 100,  low = True,binary=False, tf=False, tf_idf=False,
                 stopwords=False, tf_stop=False, verbose=True, analyzer_char=False, bi = False):
    documents = data

    dictionary1, dictionary2 = dict_scores(
        d1, d2, num_feat1, num_feat2, no_distintc=dif, con=con)
    dictionary = dictionary1+dictionary2

    
    
    if bi == True: 
        bi1 = get_bigrams(dictionary1)
        bi2 = get_bigrams(dictionary2)
        bigrams = bi1 + bi2
        
    if bi == False:
        dictionary = set(dictionary)
    else:
        dictionary = bigrams+dictionary
        dictionary = set(dictionary)        
        
        
    dictionary = list(dictionary)
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
        vectorizer = CountVectorizer(ngram_range=(min, max), binary=True, analyzer=analyzer_type, vocabulary = dictionary, lowercase = low)
        #features_name = vectorizer.get_feature_names_out()
        #stop_words= []
    elif stopwords:
        vectorizer = TfidfVectorizer(ngram_range=(min, max), stop_words='english',
                                     analyzer=analyzer_type, sublinear_tf=True, vocabulary = dictionary, lowercase = low)
        #stop_words= vectorizer.get_stop_words()
        #features_name = vectorizer.get_feature_names_out()
    elif tf:
        vectorizer = TfidfVectorizer(ngram_range=(min, max),
                                     analyzer=analyzer_type, sublinear_tf=True, use_idf=False, vocabulary = dictionary, lowercase= low)
        #features_name = vectorizer.get_feature_names_out()
        #stop_words = []
    elif tf_stop:
        vectorizer = TfidfVectorizer(ngram_range=(min, max), stop_words='english',
                                     analyzer=analyzer_type, sublinear_tf=True, use_idf=False, vocabulary = dictionary, lowercase = low)
        #features_name = vectorizer.get_feature_names_out()
        #stop_words = vectorizer.get_stop_words()

    elif tf_idf:
        vectorizer = TfidfVectorizer(ngram_range=(min, max), sublinear_tf=True, analyzer=analyzer_type, vocabulary = dictionary, lowercase = low)
        #features_name = vectorizer.get_feature_names_out()
        #stop_words = []


    X_train = vectorizer.fit_transform(x_train)
    X_val = vectorizer.transform(x_val)
    
    
    if verbose:
        print("done in %fs" % (time() - t_initial), X_train.shape, X_val.shape)

    y = np.array(y_train)
    num_feat =num_feat1 + num_feat2
    
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
    if bi == False:
        feature_names = dictionary
    return X_train, y, X_test, len(feature_names)



def run_experiment_anorexia(data, label, ntrain, test_labels, 
num_exp, min, max, num_feat1, num_feat2, dic, dif, weight, classifier, processor, bi):
    if dic == 1: 
        con = False
        kw1 = get_list_key(post_pos_anxia1)
        kw2 = get_list_key(post_neg_anxia1)  
        dict_str = 'Level post uppercase'      
    if dic == 3:
        con = False
        kw1 = get_list_key(user_pos_anxia1)
        kw2 = get_list_key(user_neg_anxia1) 
        dict_str = 'Level user uppercase'
    if dic == 5:
        con = True
        kw1 = get_list_key(con_pos_anxia1)
        kw2 = get_list_key(con_neg_anxia1)
        dict_str = 'Level concatenation uppercase'

    if dic == 7: 
        con = False
        kw1 = get_list_key(user_pos_anxia1)
        kw2 = get_list_key(user_neg_anxia3) 
        dict_str = 'Level user upercase v2'
    if dic == 9:
        con = True
        kw1 = get_list_key(con_pos_anxia1)
        kw2 = get_list_key(con_neg_anxia3)
        dict_str = 'Level concatenation upercase v2'      
    if dic == 2: 
        con =  False
        kw1 = get_list_key(post_pos_anxia2)
        kw2 = get_list_key(post_neg_anxia2)
        dict_str = 'Level post lowercase'
    if dic == 4: 
        con = False
        kw1 = get_list_key(user_pos_anxia2)
        kw2 = get_list_key(user_neg_anxia2)
        dict_str = 'Level user lowercase'		        
    elif dic == 6: 
        con = True
        kw1 = get_list_key(con_pos_anxia2)
        kw2 = get_list_key(con_neg_anxia2)
        dict_str = 'Level concatenation lowercase'
    if dic == 8: 
        con = False
        kw1 = get_list_key(user_pos_anxia2)
        kw2 = get_list_key(user_neg_anxia4)
        dict_str = 'Level user lowercase v2'		        
    elif dic == 10: 
        con = True
        kw1 = get_list_key(con_pos_anxia2)
        kw2 = get_list_key(con_neg_anxia4)
        dict_str = 'Level concatenation lowercase v2'

    if dic ==2 or dic == 4 or dic == 6 or dic == 8 or dic ==10: 
        low = True
    else:
        low = False

    if weight == 'binary':
        x_train, y, x_test, l_dic = building_bow(data=data, labels=label, ntrain=ntrain,d1=kw1, d2=kw2,
                                                              num_feat1=num_feat1, low = low, min=min, max=max,
                                                              num_feat2=num_feat2, con=con, dif = dif,
                                                              binary=True, verbose=False, bi = bi)
    elif weight == 'tf_stop':
        x_train, y, x_test, l_dic = building_bow(data=data, labels=label, ntrain=ntrain,d1=kw1, d2=kw2,
                                                              num_feat1=num_feat1, low = low,  min=min, max=max,
                                                              num_feat2=num_feat2, con = con, dif = dif, 
                                                              tf=True, stopwords=True, verbose=False, bi = bi)
    elif weight == 'tf':
        x_train, y, x_test, l_dic = building_bow(data=data, labels=label, ntrain=ntrain,d1=kw1, d2=kw2,
                                                              num_feat1=num_feat1, low = low, min=min, max=max,
                                                              num_feat2=num_feat2, con = con, dif = dif,
                                                              tf=True,  verbose=False, bi = bi)
    elif weight == 'stopwords':
        x_train, y, x_test, l_dic= building_bow(data=data, labels=label, ntrain=ntrain,d1=kw1, d2=kw2,
                                                              num_feat1=num_feat1, low = low, min=min, max=max,
                                                              num_feat2=num_feat2, con = con, dif= dif,
                                                              stopwords=True,verbose=False, bi = bi)
    elif weight == 'tf_idf':
        x_train, y, x_test, l_dic= building_bow(data=data, labels=label, ntrain=ntrain,d1=kw1, d2=kw2,
                                                              num_feat1=num_feat1, low = low, min=min, max=max,
                                                              num_feat2=num_feat2, con = con, dif = dif, 
                                                              tf_idf=True,  verbose=False, bi = bi)
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

        if dif == True: 
            dif_str = 'No difference'
        if dif == False:
            dif_str = 'Difference'
        f1 = f1_score(test_labels, y_pred)
        f_scores.append(f1)
        
        if max == 1:
            f = open('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Baseline_yake/test_anxia.txt','a')
            f.write('\n' + str(num_exp) + ',' + str(num_feat1)  + ',' + str(num_feat2) + ',' + str(min) +',' + str(max) 
                                +',' + str(weight) + ',' + dict_str+ ',' + str(l_dic) +  ',' + dif_str+ ',' + str(classifier)+ 
                                    ',' + str(seed_value) + ',' + str(f1) + ',' + str(a1) +  ','+ processor)
            f.close() 
            f.close()
            if i == 4:

                f = open('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Baseline_yake/anxia_var.txt','a')
                f.write('\n' + str(num_exp) + ',' + str(seeds) + ',' +  str(np.var(f_scores)) + ',' + str(np.std(f_scores))) 
                f.close()           
            
            
        if max == 2: 
            f = open('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Baseline_yake/anxia_bi.txt','a')
            f.write('\n' + str(num_exp) + ',' + str(num_feat1)  + ',' + str(num_feat2) + ',' + str(min) +',' + str(max) 
                                +',' + str(weight) + ',' + dict_str+ ',' + str(l_dic) +  ',' + dif_str+ ',' + str(classifier)+ 
                                    ',' + str(seed_value) + ',' + str(f1) + ',' + str(a1) +  ','+ processor)
            f.close() 
            f.close()
            if i == 4:

                f = open('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Baseline_yake/anxia_var_bi.txt','a')
                f.write('\n' + str(num_exp) + ',' + str(seeds) + ',' +  str(np.var(f_scores)) + ',' + str(np.std(f_scores))) 
                f.close()

        




def run_experiment_depression(data, label, ntrain, test_labels, 
num_exp, min, max, num_feat1, num_feat2, dic, dif, weight, classifier):

    if dic == 2: 
        con = False
        kw1 = get_list_key(post_pos_dep2)
        kw2 = get_list_key(post_neg_dep2)
        dict_str = 'Level post lowercase'  
    if dic == 4: 
        con = False
        kw1 = get_list_key(user_pos_dep2)
        kw2 = get_list_key(user_neg_dep2)
        dict_str = 'Level user lowercase'  
    if  dic == 6: 
        con = True
        kw1 = get_list_key(con_pos_dep2)
        kw2 = get_list_key(con_neg_dep2)
        dict_str = 'Level concatenation lowercase'    
    if dic == 8: 
        con = False
        kw1 = get_list_key(user_pos_dep2)
        kw2 = get_list_key(user_neg_dep3)
        dict_str = 'Level user lowercase v2'  
    if  dic == 10: 
        con = True
        kw1 = get_list_key(con_pos_dep2)
        kw2 = get_list_key(con_neg_dep4)
        dict_str = 'Level concatenation lowercase v2' 

    if dic == 1:
        con = False
        kw1 = get_list_key(post_pos_dep1)
        kw2 = get_list_key(post_neg_dep1)
        dict_str = 'Level post upercase'  
    if dic == 3: 
        con = False
        kw1 = get_list_key(user_pos_dep1)
        kw2 = get_list_key(user_neg_dep1) 
        dict_str = 'Level user upercase'  
    if dic == 5: 
        con = True
        kw1 = get_list_key(con_pos_dep1)
        kw2 = get_list_key(con_neg_dep1)
        dict_str = 'Level concatenation upercase'  
    if dic == 7: 
        con = False
        kw1 = get_list_key(user_pos_dep1)
        kw2 = get_list_key(user_neg_dep3) 
        dict_str = 'Level user upercase v2'  
    if dic == 9: 
        con = True
        kw1 = get_list_key(con_pos_dep1)
        kw2 = get_list_key(con_neg_dep4)
        dict_str = 'Level concatenation upercase v2'  

    if dic ==2 or dic == 4 or dic == 6 or dic == 8 or dic ==10: 
        low = True
    else:
        low = False
        
    if weight == 'binary':
        x_train, y, x_test, l_dic= building_bow(data=data, labels=label, ntrain=ntrain,d1=kw1, d2=kw2,
                                                              num_feat1=num_feat1, low = low,  min=min, max=max,
                                                              num_feat2=num_feat2, con=con, dif = dif,
                                          binary=True, verbose=False)
    elif weight == 'tf_stop':
        x_train, y, x_test, l_dic= building_bow(data=data, labels=label, ntrain=ntrain,d1=kw1, d2=kw2,
                                                              num_feat1=num_feat1, low = low, min=min, max=max,
                                                              num_feat2=num_feat2, con=con, dif = dif,
                                          tf=True, stopwords=True, verbose=False)
    elif weight == 'tf':
        x_train, y, x_test,l_dic= building_bow(data=data, labels=label, ntrain=ntrain,d1=kw1, d2=kw2,
                                                              num_feat1=num_feat1, low = low,  min=min, max=max,
                                                              num_feat2=num_feat2, con=con, dif = dif,
                                          tf=True, verbose=False)
    elif weight == 'stopwords':
        x_train, y, x_test,l_dic= building_bow(data=data, labels=label, ntrain=ntrain,d1=kw1, d2=kw2,
                                                              num_feat1=num_feat1, low = low,  min=min, max=max,
                                                              num_feat2=num_feat2, con=con, dif = dif, 
                                          stopwords=True, verbose=False)
    elif weight == 'tf_idf':
        x_train, y, x_test, l_dic= building_bow(data=data, labels=label, ntrain=ntrain,d1=kw1, d2=kw2,
                                                              num_feat1=num_feat1, low = low,  min=min, max=max,
                                                              num_feat2=num_feat2, con=con, dif = dif,
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
        
        if dif == True: 
            dif_str = 'No difference'
        if dif == False:
            dif_str = 'Difference'
        f = open('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Baseline_yake/test_dep.txt','a')

        f.write('\n' + str(num_exp) + ',' + str(num_feat1)  + ',' + str(num_feat2) + ',' + str(min) +',' + str(max) 
                               +',' + str(weight) + ',' + dict_str+ ',' +  dif_str + ',' + str(classifier)+ 
                                ',' + str(seed_value) + ',' + str(f1) + ',' + str(a1) )
        f.close()
        if i == 4:

            f = open('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Baseline_yake/dep_var.txt','a')
            f.write('\n' + str(num_exp) + ',' + str(seeds) + ',' +  str(np.var(f_scores)) + ',' + str(np.std(f_scores))) 
            f.close()


