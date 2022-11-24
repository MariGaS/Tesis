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

from sklearn.metrics import f1_score


##  constructing BOW ##


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


def building_bow(data, labels, ntrain, min=1, max=1, num_feat=3000, binary=False, tf=False, tf_idf=False,
                 norm=False, stopwords=False, tf_stop=False, verbose=True, analyzer_char=False):
    documents = data
    if norm:
        documents = normalize(documents)

    # split the data
    # x_train, x_val, y_train, y_val = train_test_split(documents, labels, test_size= split, random_state=42)
    x_train = documents[:ntrain]
    x_val = documents[ntrain:]
    y_train = labels[:ntrain]
    t_initial = time()

    analyzer_type = 'word'  # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    if analyzer_char:
        analyzer_type = 'char'
        
    stop_words =['hola']
    if binary:
        vectorizer = CountVectorizer(ngram_range=(min, max), binary=True)
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

    else:
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
    return X_train, y, X_test, ch2, feature_names, stop_words


def run_experiment_anorexia(test_labels, num_exp, param_list={}):
    parameters = {'C': [.05, .12, .25, .5, 1, 2, 4]}

    svr = svm.LinearSVC(class_weight='balanced')
    grid_anorexia = GridSearchCV(estimator=svr, param_grid=parameters, n_jobs=8, scoring='f1_macro', cv=5)

    data = param_list['Data']
    label = param_list['label']
    ntrain = param_list['ntrain']
    min = param_list['min']
    max = param_list['max']
    num_feat = param_list['num_feat']
    weight = param_list['weight']
    norm = param_list['norm']

    if weight == 'binary':
        x_train, y, x_test, chi, features_name, stop_words = building_bow(data=data, labels=label, ntrain=ntrain, min=min, max=max,
                                                              num_feat=num_feat,
                                                              binary=True, norm=norm, verbose=False)
    elif weight == 'tf_stop':
        x_train, y, x_test, chi, features_name, stop_words = building_bow(data=data, labels=label, ntrain=ntrain, min=min, max=max,
                                                              num_feat=num_feat,
                                                              tf=True, stopwords=True, norm=norm, verbose=False)
    elif weight == 'tf':
        x_train, y, x_test, chi, features_name, stop_words = building_bow(data=data, labels=label, ntrain=ntrain, min=min, max=max,
                                                              num_feat=num_feat,
                                                              tf=True, norm=norm, verbose=False)
    elif weight == 'stopwords':
        x_train, y, x_test, chi, features_name, stop_words = building_bow(data=data, labels=label, ntrain=ntrain, min=min, max=max,
                                                              num_feat=num_feat,
                                                              stopwords=True, norm=norm, verbose=False)
    else:
        x_train, y, x_test, chi, features_name, stop_words = building_bow(data=data, labels=label, ntrain=ntrain, min=min, max=max,
                                                              num_feat=num_feat,
                                                              tf_idf=True, norm=norm, verbose=False)

    grid_anorexia.fit(x_train, y)

    y_pred = grid_anorexia.predict(x_test)
    a1 = grid_anorexia.best_params_

    # print("Best paramter: ", a1)

    p, r, f, _ = precision_recall_fscore_support(test_labels, y_pred, average='macro', pos_label=1)
    result_name = 'result_anorexia_' + str(num_exp) + '.txt'
    path_name = 'Results/Anorexia' + '/' + result_name
    path_name_features = 'Results/Anorexia' + '/features_words_' + str(num_exp) + '.txt'
    path_name_stop = 'Results/Anorexia' + '/stopwords_' + str(num_exp) + '.txt'
    with open(path_name, "w") as f:
        f.write("Experimento de anorexia número: " + str(num_exp) + '\n')
        f.write("Confusion matrix: \n")
        f.write(str(confusion_matrix(test_labels, y_pred)))

        f.write('Metrics classification \n')
        f.write(str(metrics.classification_report(test_labels, y_pred)))

        f.write('Best parameter:\n')
        f.write(str(a1))
        f.write('\n')
        # f.write('Precission, recall and F1-score:')
        # f.write(p,r,f)
        f.close()

    with open(path_name_features, "w") as f:
        f.write("Experimento de anorexia número: " + str(num_exp) + '\n')
        for word in features_name:
            f.write(str(word) + '\n')
        f.close()
	
    if stop_words:
        with open(path_name_stop, "w") as f:
            f.write("Experimento de anorexia número: " + str(num_exp) + '\n')
            for word in stop_words:
                f.write(str(word) + '\n')
            f.close()

    return f1_score(test_labels, y_pred),a1,chi


def run_experiment_depression(test_labels, num_exp, param_list={}):
    parameters = {'C': [.05, .12, .25, .5, 1, 2, 4]}

    svr = svm.LinearSVC(class_weight='balanced')
    grid_depression = GridSearchCV(estimator=svr, param_grid=parameters, n_jobs=8, scoring='f1_macro', cv=5)

    data = param_list['Data']
    label = param_list['label']
    ntrain = param_list['ntrain']
    min = param_list['min']
    max = param_list['max']
    num_feat = param_list['num_feat']
    weight = param_list['weight']
    norm = param_list['norm']

    if weight == 'binary':
        x_train, y, x_test, chi, features_name, stop_words= building_bow(data=data, labels=label, ntrain=ntrain, min=min, max=max, num_feat=num_feat,
                                          binary=True, norm=norm, verbose=False)
    elif weight == 'tf_stop':
        x_train, y, x_test, chi, features_name, stop_words = building_bow(data=data, labels=label, ntrain=ntrain, min=min, max=max, num_feat=num_feat,
                                          tf=True, stopwords=True, norm=norm, verbose=False)
    elif weight == 'tf':
        x_train, y, x_test, chi, features_name, stop_words = building_bow(data=data, labels=label, ntrain=ntrain, min=min, max=max, num_feat=num_feat,
                                          tf=True, norm=norm, verbose=False)
    elif weight == 'stopwords':
        x_train, y, x_test, chi, features_name, stop_words = building_bow(data=data, labels=label, ntrain=ntrain, min=min, max=max, num_feat=num_feat,
                                          stopwords=True, norm=norm, verbose=False)
    elif weight == 'tf_idf':
        x_train, y, x_test, chi, features_name, stop_words = building_bow(data=data, labels=label, ntrain=ntrain, min=min, max=max, num_feat=num_feat,
                                          tf_idf=True, norm=norm, verbose=False)
    else:
        print("Include the name of the weightinh")
        exit()

    grid_depression.fit(x_train, y)

    y_pred = grid_depression.predict(x_test)
    a1 = grid_depression.best_params_

    # print("Best paramter: ", a1)

    p, r, f, _ = precision_recall_fscore_support(test_labels, y_pred, average='macro', pos_label=1)
    result_name = 'result_depression_' + str(num_exp) + '.txt'
    path_name = 'Results/Depresion' + '/' + result_name
    path_name_features = 'Results/Depresion' + '/features_words_final_' + str(num_exp) + '.txt'
    path_name_stop = 'Results/Depresion' + '/stopwords_' + str(num_exp) + '.txt'
    with open(path_name, "w") as f:
        f.write("Experimento de depresión número: " + str(num_exp) + '\n')
        f.write(("Confusion matrix: \n"))
        f.write(str(confusion_matrix(test_labels, y_pred)))
        f.write('Metrics classification \n')
        f.write(str(metrics.classification_report(test_labels, y_pred)))

        f.write('Best parameter:\n')
        f.write(str(a1))
        # f.write('\n')
        # f.write('Precission, recall and F1-score:')
        # f.write(p,r,f)
        f.close()

    with open(path_name_features, "w") as f:
        f.write("Experimento de depresión número: " + str(num_exp) + '\n')
        for word in features_name:
            f.write(str(word) + '\n')
        f.close()
	
    if stop_words:
        with open(path_name_stop, "w") as f:
            f.write("Experimento de depresión número: " + str(num_exp) + '\n')
            for word in stop_words:
                f.write(str(word) + '\n')
            f.close()

    return f1_score(test_labels, y_pred),a1,chi
