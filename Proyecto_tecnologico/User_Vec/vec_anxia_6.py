import pandas as pd
import numpy as np
from nltk.tokenize import TweetTokenizer
import nltk
from nltk import TweetTokenizer
from text_functions import (get_text_chunk,
                            get_text_test_anorexia)
from functions_for_vec import run_exp_anorexia
from numpy import array, asarray, zeros
from time import time
import pandas as pd



### ANOREXIA'S EXPERIMENTS ####
anxia_train = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Anorexia_2018/Anorexia_Datasets_1/train'
anxia_test = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Anorexia_2018/Anorexia_Datasets_1/test'

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
for i in range(len(tr_anorexia)):
    if i < 20:
        tr_label.append(1)
    else:
        tr_label.append(0)

test_url_anxia = []
test_labels_anxia = []

with open('/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Anorexia_2018/Anorexia_Datasets_1/test/test_golden_truth.txt') as f:
    lines = f.readlines()
    for line in lines:
        test_url_anxia.append(line[:-3])  # only the name of the subject

        test_labels_anxia.append(int(line[-2:-1]))  # only the label
    f.close()

# print(tr_anorexia[0][:10])
# print(test_labels_anxia[:3])

#  ---- TEST-EXTRACTION  --------
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

# --------EXPERIMENTS ----------------#
norm_data1 ={'type' : 'normalize'}
norm_data2 ={'type' : 'standard'}
norm_vec1 ={'type' : 'avg'}
norm_vec2 = {'type' : 'norm'}

f_score = []
best    =[]
norm_dataset = []
norm_vect = []
vocabulary = []
print('experiment 17')
f17,a17 = run_exp_anorexia(41,tr_anorexia,test_anxia,tr_label,test_labels_anxia,'dict51',option =3,norm_data=True, norm_vec=True,
                         sub_param=norm_data2,  param_norm= norm_vec1)
f_score.append(f17)
best.append(a17)
norm_dataset.append(norm_data2['type'])
norm_vect.append(norm_vec1['type'])
vocabulary.append('dict51')

print('experiment 18')
f18,a18 = run_exp_anorexia(42,tr_anorexia,test_anxia,tr_label,test_labels_anxia,'dict51',option =3,norm_data=True, norm_vec=True,
                         sub_param=norm_data2,  param_norm= norm_vec2)
f_score.append(f18)
best.append(a18)
norm_dataset.append(norm_data2['type'])
norm_vect.append(norm_vec2['type'])
vocabulary.append('dict51')

print('experiment 19')
f19,a19 = run_exp_anorexia(43,tr_anorexia,test_anxia,tr_label,test_labels_anxia,'dict51_ex',option =3,norm_data=True,norm_vec=False,
                         sub_param=norm_data1)
f_score.append(f19)
best.append(a19)
norm_dataset.append(norm_data1['type'])
norm_vect.append('none')
vocabulary.append('dict51_ex')

print('experiment 20')
f20,a20 = run_exp_anorexia(44,tr_anorexia,test_anxia,tr_label,test_labels_anxia,'dict51_ex',option =3,norm_data=True, norm_vec=False,
                         sub_param=norm_data2)
f_score.append(f20)
best.append(a20)
norm_dataset.append(norm_data2['type'])
norm_vect.append('none')
vocabulary.append('dict51_ex')

print('experiment 21')
f21,a21 = run_exp_anorexia(45,tr_anorexia,test_anxia,tr_label,test_labels_anxia,'dict51_ex',option =3,norm_data=True, norm_vec=True,
                         sub_param=norm_data1,  param_norm= norm_vec1)
f_score.append(f21)
best.append(a21)
norm_dataset.append(norm_data1['type'])
norm_vect.append(norm_vec1['type'])
vocabulary.append('dict51_ex')

print('experiment 22')
f22,a22= run_exp_anorexia(46,tr_anorexia,test_anxia,tr_label,test_labels_anxia,'dict51_ex',option =3,norm_data=True, norm_vec=True,
                         sub_param=norm_data1,  param_norm= norm_vec2)
f_score.append(f22)
best.append(a22)
norm_dataset.append(norm_data1['type'])
norm_vect.append(norm_vec2['type'])
vocabulary.append('dict51_ex')

print('experiment 23')
f23,a23 = run_exp_anorexia(47,tr_anorexia,test_anxia,tr_label,test_labels_anxia,'dict51_ex',option =3,norm_data=True, norm_vec=True,
                         sub_param=norm_data2,  param_norm= norm_vec1)
f_score.append(f23)
best.append(a23)
norm_dataset.append(norm_data2['type'])
norm_vect.append(norm_vec1['type'])
vocabulary.append('dict51_ex')

print('experiment 24')
f24,a24 = run_exp_anorexia(48,tr_anorexia,test_anxia,tr_label,test_labels_anxia,'dict51_ex',option =3,norm_data=True, norm_vec=True,
                         sub_param=norm_data2,  param_norm= norm_vec2)
f_score.append(f24)
best.append(a24)
norm_dataset.append(norm_data1['type'])
norm_vect.append(norm_vec2['type'])
vocabulary.append('dict51_ex')

print('End experiments')
l = [str(x) for x in range(1,9)]
data = { 'best_parameter': best, 'f1': f_score, 'is_norm_dataset': norm_dataset, 'is_norm_vec': norm_vect, 'dict': vocabulary}
df = pd.DataFrame(data, index= l)


df.to_csv('/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Anorexia_vec/result_anorexia_vec_6.csv',
          sep='\t')
