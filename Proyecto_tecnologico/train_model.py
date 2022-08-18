from text_functions import (get_text_chunk,
                            get_text_test_anorexia, get_text_labels,
                            get_text_test)
#from functions_for_vec import  normalize
import yake
from gensim.models import FastText
from gensim.models import phrases, word2vec
from nltk import TweetTokenizer
import os
import re
from os import listdir
from os.path import isfile, join
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from gensim.models import utils
from gensim.models import FastText
import gensim.downloader
from gensim.test.utils import get_tmpfile, datapath
#from gensim.models import fasttext
import fasttext
import fasttext.util
import pickle



tokenizer = TweetTokenizer()
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

def get_emotion_from_file(path_corpus): 
    words_list = []
    emotion_list = []
    is_in  = []
    with open(path_corpus, "r") as f: 

        header = 0
        for line in f :
            words = re.split(r'\t+', line)  #cada linea se divide por palabra 
            words_list.append(words[0])
            emotion_list.append(words[1])
            is_in.append(words[2][:-1])
 
          
            
    return words_list, emotion_list, is_in
path_emotions = 'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'
l1,l2,l3 = get_emotion_from_file(path_emotions)


def get_dic_emotions(l_words,l_emotions,l_is_in):
    dict_emotions = dict()
    cont = 1
    emotions = []
    for i in range(len(l_words)):
        if cont%10 != 0 and i%10 != 9:
            #print(i,cont,'i')
            if l_is_in[i] == '1': 
                emotions.append(l_emotions[i])
            
            cont +=1
            #print(i,cont,'f')
            #print(emotions)
        if cont%10 == 0 and i%10 == 9:
            if l_is_in[i] == '1':
                emotions.append(l_emotions[i])
            if len(emotions)>0:
                dict_emotions[l_words[i]] = emotions
            #print(i,cont,'l')
            #print(emotions)
            emotions = []
            cont = 1
            #print(l_words[i],i)
    return dict_emotions
    
dict_emotions = get_dic_emotions(l1,l2,l3)
def add_words(text, dict_emo):
    text = normalize(text)
    text_change = []
    for i in range(len(text)):
        string = ''
        corpus_palabras = []
        #obtienes todas las palabras del documento 
        corpus_palabras += tokenizer.tokenize(text[i])
        
        for word in corpus_palabras:
            #print(type(word))
            string = string + ' ' + word
            if word in dict_emo and word.isnumeric() == False: 
                string = string +' ' + word
                for j in range(len(dict_emo[word])):
                    string = string + ' ' +dict_emo[word][j]
        text_change.append(string)
    
    return text_change

train_neg_2017 = 'depression2022/training_data/2017_cases/neg'
train_pos_2017 = 'depression2022/training_data/2017_cases/pos'
train_neg_2018 = 'depression2022/training_data/2018_cases/neg'
train_pos_2018 = 'depression2022/training_data/2018_cases/pos'

#- MAKE LIST FROM 2017 AND 2018 -#
tr_neg_2017, tr_lab_2017 = get_text_labels(train_neg_2017, polarity='Negative')
tr_pos_2017, tr_lab_pos_2017 = get_text_labels(train_pos_2017, polarity='Pos')
tr_neg_2018, tr_lab_2018 = get_text_labels(train_neg_2018, polarity='Negative')
tr_pos_2018, tr_lab_pos_2018 = get_text_labels(train_pos_2018, polarity='Pos')

# ALL TRAINING DATA 2017
tr_txt_2017 = [*tr_neg_2017, *tr_pos_2017]
tr_y_2017 = [*tr_lab_2017, *tr_lab_pos_2017]
# ALL TRAINING DATA 2018
tr_txt_2018 = [*tr_neg_2018, *tr_pos_2018]
tr_y_2018 = [*tr_lab_2018, *tr_lab_pos_2018]


test_data = 'depression2022/test_data/datos'

test_url = []
test_labels = []

#with open('depression2022/test_data/risk_golden_truth.txt') as f:
#    lines = f.readlines()
#    for line in lines:

#        test_url.append(line[:-3])  # only the name of the subject

#        test_labels.append(int(line[-2:-1]))  # only the label
#    f.close()

#test_txt = get_text_test(test_data, test_url)
#train = tr_txt_2017 + tr_txt_2018 #all training data
#test = test_txt  #test data

#labels_dep = tr_y_2018 + tr_y_2017 #labeling

### ANOREXIA'S EXPERIMENTS ####
anxia_train = 'Anorexia_2018/Anorexia_Datasets_1/train'
anxia_test = 'Anorexia_2018/Anorexia_Datasets_1/test'

pos = 'positive_examples'
neg = 'negative_examples'

all_pos = []
all_neg = []
for i in range(1, 11):
    path_chunk_pos = anxia_train + '/' + pos + '/chunk' + str(i)
    path_chunk_neg = anxia_train + '/' + neg + '/chunk' + str(i)

    temp1 = get_text_chunk(path_chunk_pos)
    temp2 = get_text_chunk(path_chunk_neg)
    if i == 1:
        all_pos = temp1
        all_neg = temp2
    else:
        for j in range(len(temp1)):
            all_pos[j] += temp1[j]

        for j in range(len(temp2)):
            all_neg[j] += temp2[j]

tr_anorexia = [*all_pos, *all_neg]
tr_label = []


#test_url_anxia = []
#test_labels_anxia = []

#with open('Anorexia_2018/Anorexia_Datasets_1/test/test_golden_truth.txt') as f:
#    lines = f.readlines()
#    for line in lines:
#        test_url_anxia.append(line[:-3])  # only the name of the subject

#        test_labels_anxia.append(int(line[-2:-1]))  # only the label
#    f.close()


#test_anxia = []
#for i in range(1, 11):

#    temp1 = get_text_test_anorexia(anxia_test, test_url_anxia, i)

#    if i == 1:
#        test_anxia = temp1
        # print("Text extracted from chunk: ", i)
#    else:

#        for j in range(len(temp1)):
#            test_anxia[j] += temp1[j]
        # print("Text extracted from chunk: ", i)
        
#train_clean = normalize(add_words(tr_anorexia,dict_emotions))
#test_clean = normalize(add_words(test_anxia,dict_emotions))
#train_dep  = normalize(add_words(train,dict_emotions))
#test_dep   = normalize(add_words(test,dict_emotions))


#sentences = [*train_clean, *test_clean, *train_dep,*test_dep]
#all_text = []
#for i in range(len(sentences)):
#    all_text.append(sentences[i].split(" "))
print("Inicia entrenamiento")


def get_lines(text):
        #text = normalize(text)
    text_change = []
    string = ''
    for i in range(len(text)):
        string = string + ' ' + text[i]
    #text_change.append(string)
    return string
#all_pos_clean = normalize(all_pos)
#all_neg_clean = normalize(all_neg)
#t = get_lines(all_pos_clean)
#s = get_lines(all_neg_clean)

#def get_words_from_kw(kw):
#    list1 = []
#    for i in range(len(kw)):
#        list1.append(kw[i][0])
#    return list1
    
#kw_extractor = yake.KeywordExtractor()
#text = t
#language = "en"
#max_ngram_size = 1
#deduplication_threshold = 0.9
#numOfKeywords = 8000
#custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
#keywords1 = custom_kw_extractor.extract_keywords(text)

#with open('key1', 'wb') as f: 
    #Pickling
#    pickle.dump(keywords1, f)
#    f.close()
    
#kw_extractor = yake.KeywordExtractor()
#text = s
#language = "en"
#max_ngram_size = 1
#deduplication_threshold = 0.9
#numOfKeywords = 8000
#custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
#keywords2 = custom_kw_extractor.extract_keywords(text)


    
#with open('key2', 'wb') as f: 
    #Pickling
#    pickle.dump(keywords2, f)
#    f.close()

#kw_extractor = yake.KeywordExtractor()
#text = t
#language = "en"
#max_ngram_size = 2
#deduplication_threshold = 0.9
#numOfKeywords = 8000
#custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
#keywords3 = custom_kw_extractor.extract_keywords(text)

#with open('key3', 'wb') as f: 
    #Pickling
#    pickle.dump(keywords3, f)
#    f.close()

#kw_extractor = yake.KeywordExtractor()
#text = s
#language = "en"
#max_ngram_size = 2
#deduplication_threshold = 0.9
#numOfKeywords = 8000
#custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
#keywords4 = custom_kw_extractor.extract_keywords(text)

#with open('key4', 'wb') as f: 
    #Pickling
#    pickle.dump(keywords4, f)
#    f.close()

#model = FastText(vector_size=300, window=15, min_count=1)
#model.build_vocab(all_text)
#total_examples = model.corpus_count
#model.train(all_text, total_examples=total_examples, epochs=5)

#model.save('Model/emotions.model')

#fname = get_tmpfile("fasttext.model")
#model.save(fname)


all_pos_clean = normalize([*tr_pos_2017, *tr_pos_2018])
all_neg_clean = normalize([*tr_neg_2017, *tr_neg_2018])
t = get_lines(all_pos_clean)
s = get_lines(all_neg_clean)
kw_extractor = yake.KeywordExtractor()
text = t
language = "en"
max_ngram_size = 1
deduplication_threshold = 0.9
numOfKeywords = 15000
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
keywords5 = custom_kw_extractor.extract_keywords(text)

with open('key9', 'wb') as f: 
    #Pickling
    pickle.dump(keywords5, f)
    f.close()
    
kw_extractor = yake.KeywordExtractor()
text = s
language = "en"
max_ngram_size = 1
deduplication_threshold = 0.9
numOfKeywords = 15000
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
keywords6 = custom_kw_extractor.extract_keywords(text)


    
with open('key10', 'wb') as f: 
    #Pickling
    pickle.dump(keywords6, f)
    f.close()

#kw_extractor = yake.KeywordExtractor()
#text = t
#language = "en"
#max_ngram_size = 2
#deduplication_threshold = 0.9
#numOfKeywords = 8000
#custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
#keywords7 = custom_kw_extractor.extract_keywords(text)

#with open('key7', 'wb') as f: 
    #Pickling
#    pickle.dump(keywords7, f)
#    f.close()

#kw_extractor = yake.KeywordExtractor()
#text = s
#language = "en"
#max_ngram_size = 2
#deduplication_threshold = 0.9
#numOfKeywords = 8000
#custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
#keywords8 = custom_kw_extractor.extract_keywords(text)

#with open('key8', 'wb') as f: 
    #Pickling
#    pickle.dump(keywords8, f)
#    f.close()

print("Termina entrenamiento")
