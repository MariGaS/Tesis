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
import nltk
from sklearn.metrics import f1_score
import ekphrasis
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from sklearn.naive_bayes import MultinomialNB
##  constructing BOW ##

tt = nltk.tokenize.TweetTokenizer()
def normalize(document): 

    document = [re.sub(r'https?:\/\/\S+', '', x) for x in document] #eliminate url
    document = [re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', x) for x in document] #eliminate url 
    document = [re.sub(r'{link}', '', x) for x in document] #eliminate link
    document = [re.sub(r"\[video\]", '', x) for x in document] #eliminate video 
    document = [re.sub(r'\s+', ' ' '', x).strip() for x in document]
    document = [x.replace("#","") for x in document]  #eliminate #
    document = [re.subn(r'[^\w\s,]',"", x)[0].strip() for x in document] #eliminate emoticons 

    return document

## BOW

# showing the example from https://github.com/cbaziotis/ekphrasis


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

def building_bow(data, labels, ntrain, min=1, max=1, num_feat=3000, binary=False, tf=False, tf_idf=False,
                 stopwords=False, tf_stop=False, verbose=True, analyzer_char=False):
    documents = data

    seed_val = 42
    np.random.seed(seed_val)
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
        vectorizer = CountVectorizer(ngram_range=(min, max), binary=True, preprocessor = ekphrasis_processor, analyzer=analyzer_type)
        #features_name = vectorizer.get_feature_names_out()
        #stop_words= []
    elif stopwords:
        vectorizer = TfidfVectorizer(ngram_range=(min, max), stop_words='english',
                                     analyzer=analyzer_type, sublinear_tf=True,  preprocessor = ekphrasis_processor)
        #stop_words= vectorizer.get_stop_words()
        #features_name = vectorizer.get_feature_names_out()
    elif tf:
        vectorizer = TfidfVectorizer(ngram_range=(min, max),
                                     analyzer=analyzer_type, sublinear_tf=True, use_idf=False,  preprocessor = ekphrasis_processor)
        #features_name = vectorizer.get_feature_names_out()
        #stop_words = []
    elif tf_stop:
        vectorizer = TfidfVectorizer(ngram_range=(min, max), stop_words='english',
                                     analyzer=analyzer_type, sublinear_tf=True, use_idf=False,  preprocessor = ekphrasis_processor)
        #features_name = vectorizer.get_feature_names_out()
        #stop_words = vectorizer.get_stop_words()

    elif tf_idf:
        vectorizer = TfidfVectorizer(ngram_range=(min, max), sublinear_tf=True, analyzer=analyzer_type,  preprocessor = ekphrasis_processor)
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
    else:
        x_train, y, x_test, chi, features_name = building_bow(data=data, labels=label, ntrain=ntrain, min=min, max=max,
                                                              num_feat=num_feat,
                                                              tf_idf=True,  verbose=False)
    if classifier == 'svm':
        parameters = {'C': [.05, .12, .25, .5, 1, 2, 4]}
        svr = svm.LinearSVC(class_weight='balanced')
        grid_anorexia = GridSearchCV(estimator=svr, param_grid=parameters, n_jobs=8, scoring='f1_macro', cv=5)       
        grid_anorexia.fit(x_train, y)
        y_pred = grid_anorexia.predict(x_test)
        a1 = grid_anorexia.best_params_

    if classifier == 'NB':
        model = MultinomialNB()
        model.fit(x_train, y)
        y_pred = model.predict(x_test)
        print(y_pred)
        a1 = 'None'
    f1 = f1_score(test_labels, y_pred)
    path_name_features = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Baseline/Dictionaries/Anorexia' + '/features_words_' + str(num_exp) + '.txt'

    f = open('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Baseline/f1_anorexia.txt','a')
    f.write('\n' + str(num_exp) + ',' + str(num_feat) + ',' + str(min) +',' + str(max) +',' + str(weight) + ',' + str(classifier)+ 
                                ',' + str(f1) + ',' + str(a1))
    f.close()

    with open(path_name_features, "w") as f:
        f.write("Experimento de anorexia número: " + str(num_exp) + '\n')
        for word in features_name:
            f.write(str(word) + '\n')
        f.close()
	



def run_experiment_depression(data, label, ntrain, test_labels, 
num_exp, min, max, num_feat, weight, classifier):

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
        print("Include the name of the weightinh")
        exit()

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

    path_name_features = 'Results/Depresion' + '/features_words_final_' + str(num_exp) + '.txt'


    with open(path_name_features, "w") as f:
        f.write("Experimento de depresión número: " + str(num_exp) + '\n')
        for word in features_name:
            f.write(str(word) + '\n')
        f.close()
	
    path_name_features = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Baseline/Dictionaries/Depression' + '/features_words_' + str(num_exp) + '.txt'

    f = open('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Baseline/f1_dep.txt','a')
    f.write('\n' + str(num_exp) + ',' + str(num_feat) + ',' + str(min) +',' + str(max) +',' + str(weight) + ',' + str(classifier)+ 
                                ',' + str(f1) + ',' + str(a1))
    f.close()

    with open(path_name_features, "w") as f:
        f.write("Experimento de anorexia número: " + str(num_exp) + '\n')
        for word in features_name:
            f.write(str(word) + '\n')
        f.close()
	
