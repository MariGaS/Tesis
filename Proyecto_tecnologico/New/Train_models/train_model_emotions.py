from text_functions import (get_text_chunk,
                            get_text_test_anorexia, get_text_labels,
                            get_text_test)
from gensim.models import FastText
from gensim.models import phrases, word2vec
from nltk import TweetTokenizer
import re
from os import listdir
from os.path import isfile, join
from gensim.models import utils
from gensim.models import FastText
import gensim.downloader
from gensim.test.utils import get_tmpfile, datapath
import fasttext
import fasttext.util



tokenizer = TweetTokenizer()


def normalize(document):
    #eliminate link 
    document = [re.sub(r'{link}', '', x) for x in document]
    # eliminate video
    document = [re.sub(r"\[video\]", '', x) for x in document]
    # eliminate url
    document = [re.sub(r'https?:\/\/\S+', '', x) for x in document]
    # eliminate url
    document = [re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', x) for x in document]
    # eliminate #
    document = [x.replace("#","") for x in document]

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
path_emotions = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'
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
            string = string + ' ' + word
            if word.lower() in dict_emo and word.isnumeric() == False:  
                for j in range(len(dict_emo[word.lower()])):
                    string = string + ' ' +dict_emo[word.lower()][j]
        text_change.append([string])
    
    return text_change

train_neg_2017 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/depression2022/training_data/2017_cases/neg'
train_pos_2017 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/depression2022/training_data/2017_cases/pos'
train_neg_2018 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/depression2022/training_data/2018_cases/neg'
train_pos_2018 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/depression2022/training_data/2018_cases/pos'


tr_neg_2017= get_text_labels(train_neg_2017)
tr_pos_2017= get_text_labels(train_pos_2017)
tr_neg_2018= get_text_labels(train_neg_2018)
tr_pos_2018= get_text_labels(train_pos_2018)

# ALL TRAINING DATA 2017
tr_txt_2017 = [*tr_neg_2017, *tr_pos_2017]
# ALL TRAINING DATA 2018
tr_txt_2018 = [*tr_neg_2018, *tr_pos_2018]



test_data = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/depression2022/test_data/datos'

test_url = []
test_labels = []

with open('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/depression2022/test_data/risk_golden_truth.txt') as f:
    lines = f.readlines()
    for line in lines:

        test_url.append(line[:-3])  # only the name of the subject

        test_labels.append(int(line[-2:-1]))  # only the label
    f.close()

test_txt = get_text_test(test_data, test_url)
train = tr_txt_2017 + tr_txt_2018 #all training data
test = test_txt  #test data

### ANOREXIA'S EXPERIMENTS ####
anxia_train = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Anorexia_2018/Anorexia_Datasets_1/train'
anxia_test = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Anorexia_2018/Anorexia_Datasets_1/test'

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


test_url_anxia = []
test_labels_anxia = []

with open('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Anorexia_2018/Anorexia_Datasets_1/test/test_golden_truth.txt') as f:
    lines = f.readlines()
    for line in lines:
        test_url_anxia.append(line[:-3])  # only the name of the subject

        test_labels_anxia.append(int(line[-2:-1]))  # only the label
    f.close()


test_anxia = []
for i in range(1, 11):

    temp1 = get_text_test_anorexia(anxia_test, test_url_anxia, i)

    if i == 1:
        test_anxia = temp1
        # print("Text extracted from chunk: ", i)
    else:

        for j in range(len(temp1)):
            test_anxia[j] += temp1[j]
        # print("Text extracted from chunk: ", i)
        #         
train_clean = normalize(tr_anorexia)
train_clean = add_words(train_clean,dict_emotions)

test_clean = normalize(test_anxia)
test_clean = add_words(test_clean, dict_emotions)

train_dep = normalize(train)
train_dep  = add_words(train_dep,dict_emotions)

test_dep = normalize(test)
test_dep   = add_words(test_dep,dict_emotions)


sentences = [*train_clean, *test_clean, *train_dep,*test_dep]
all_text = []
for i in range(len(sentences)):
    all_text.append(sentences[i].split(" "))
print("Inicia entrenamiento")


model = FastText(vector_size=300, window=15, min_count=1)
model.build_vocab(all_text)
total_examples = model.corpus_count
model.train(all_text, total_examples=total_examples, epochs=5)

model.save('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Models/emotions.model')

#fname = get_tmpfile("fasttext.model")
#model.save(fname)




print("Termina entrenamiento")
