
from text_functions import (get_text_labels,
                            get_text_test)
from vector_key import run_exp_dep_sim
import pandas as pd
import multiprocessing

train_neg_2017 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/depression2022/training_data/2017_cases/neg'
train_pos_2017 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/depression2022/training_data/2017_cases/pos'
train_neg_2018 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/depression2022/training_data/2018_cases/neg'
train_pos_2018 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/depression2022/training_data/2018_cases/pos'

#- MAKE LIST FROM 2017 AND 2018 -#
tr_neg_2017, tr_lab_2017 = get_text_labels(train_neg_2017, polarity='Negative')
tr_pos_2017, tr_lab_pos_2017 = get_text_labels(train_pos_2017, polarity='Pos')
tr_neg_2018, tr_lab_2018 = get_text_labels(train_neg_2018, polarity='Negative')
tr_pos_2018, tr_lab_pos_2018 = get_text_labels(train_pos_2018, polarity='Pos')

tr_y_pos  = [*tr_lab_pos_2017, *tr_lab_pos_2018]
tr_y_neg   = [*tr_lab_2017, *tr_lab_2018]
train_y = tr_y_pos +tr_y_neg

test_data = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/depression2022/test_data/datos'

test_url = []
test_labels = []

with open('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/depression2022/test_data/risk_golden_truth.txt') as f:
    lines = f.readlines()
    for line in lines:
        if line[0:11] != 'subject3958':
            test_url.append(line[:-3])  # only the name of the subject

            test_labels.append(int(line[-2:-1]))  # only the label
    f.close()

test_txt = get_text_test(test_data, test_url)
# negative and positive training data
train_neg = [*tr_neg_2017, *tr_neg_2018]
train_pos = [*tr_pos_2017, *tr_pos_2018]
# test data
test = test_txt


f_score = []
tol = []
n_feat = []
l5 =[]
svm = []
com =[]
vec = []
best =[]
print('Begins experiments')

arg1 = [x for x in range(41,61)]
arg2 = [train_pos, train_pos, train_pos, train_pos, train_pos, train_pos, train_pos, train_pos, train_pos, train_pos,train_pos, train_pos, train_pos, train_pos, train_pos, train_pos, train_pos, train_pos, train_pos, train_pos]
arg3 = [train_neg, train_neg, train_neg, train_neg, train_neg, train_neg, train_neg, train_neg, train_neg, train_neg,train_neg, train_neg, train_neg, train_neg, train_neg, train_neg, train_neg, train_neg, train_neg, train_neg]
arg4 = [test, test, test, test, test, test, test, test, test, test,test, test, test, test, test, test, test, test, test, test]
arg5 = [test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels]
arg6 = [train_y,  train_y, train_y, train_y, train_y, train_y, train_y, train_y, train_y, train_y,train_y,  train_y, train_y, train_y, train_y, train_y, train_y, train_y, train_y, train_y]
arg7 = [0.0003,0.0003,0.0003,0.0001,0.0001,0.0001,0.00005,0.00005,0.00005,0.00001,0.00001,0.00001,0.00003, 0.00003,0.00003,0.007,0.007,0.007,0.0000009,0.000005 ] #score1 
arg8 = [0.0003,0.0007,0.0001,0.0001,0.0005,0.00005,0.00005,.00001,0.00008,0.00001,0.00006,0.000005,0.00003,0.00008,0.00001,0.007,0.003,0.009,0.0000004,0.000005] #score2
arg9 = [0.99]*20 #tolerancia
arg10 = [1]*20 #chose
arg11 = [False]*20 #fuzzy
arg12 = [False,False,False,False,False,False,False,False,False,False,True,True,True,True,True,True,True,True,True,True] #remove stop
arg13 = [True]*20 #train_data 
arg14 = [False]*20#compress
arg15 = [True]*20 #concatenate
arg16 = [True]*20 #dif
arg17 = [True] *20 #w
with multiprocessing.Pool(processes=20) as pool:
    results = pool.starmap(run_exp_dep_sim, zip(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg10,arg9,arg17,arg16,arg11,arg12,arg13,arg14,arg15))
print(results)
        
f1,a,len1, len2  = zip(*results)

print('End experiments')


data = { 'score1': arg7, 'score2': arg8, 'tol': arg9, 'fuzzy':arg11,'remove_stop' : arg12,
            'svm':arg13,'compress':arg14, 'n_dif': arg16,'best': a,'f1': f1, 'words_pos': len1, 'words_neg':len2,
            'concatenate': arg15, 'word_embeddings': arg10}
df = pd.DataFrame(data, index= arg1)        
print('End experiments')

df.to_csv('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Results/Depresion_w_key/Result_w_dep3.csv', sep='\t')

                


