
from text_functions import (get_text_labels,
                            get_text_test)
from functions_for_vec import run_exp_dep
from time import time
import pandas as pd


train_neg_2017 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/depression2022/training_data/2017_cases/neg'
train_pos_2017 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/depression2022/training_data/2017_cases/pos'
train_neg_2018 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/depression2022/training_data/2018_cases/neg'
train_pos_2018 = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/depression2022/training_data/2018_cases/pos'


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


test_data = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/depression2022/test_data/datos'

test_url = []
test_labels = []



with open('/home/est_posgrado_maria.garcia/Proyecto_tecnologico/depression2022/test_data/risk_golden_truth.txt') as f:
    lines = f.readlines()
    for line in lines:
        if line[0:11] != 'subject3958': 
            test_url.append(line[:-3])  # only the name of the subject

            test_labels.append(int(line[-2:-1]))  # only the label
    f.close()

test_txt = get_text_test(test_data, test_url)


train = tr_txt_2017 + tr_txt_2018 #all training data
test = test_txt  #test data

labels_dep = tr_y_2017 + tr_y_2018 #labeling



norm_data1 ={'type' : 'normalize'}
norm_data2 ={'type' : 'standard'}
norm_vec1 ={'type' : 'avg'}
norm_vec2 = {'type' : 'norm'}

f_score = []
best    =[]
norm_dataset = []
norm_vect = []
vocabulary = []
print('Begins experiments')
f17,a17 = run_exp_dep(17,train,test,labels_dep,test_labels,'dict51',option=2,norm_data=True, norm_vec=True,
                         sub_param=norm_data2,  param_norm= norm_vec1)
f_score.append(f17)
best.append(a17)
norm_dataset.append(norm_data2['type'])
norm_vect.append(norm_vec1['type'])
vocabulary.append('dict51')

f18,a18 = run_exp_dep(18,train,test,labels_dep,test_labels,'dict51',option=2,norm_data=True, norm_vec=True,
                         sub_param=norm_data2,  param_norm= norm_vec2)
f_score.append(f18)
best.append(a18)
norm_dataset.append(norm_data2['type'])
norm_vect.append(norm_vec2['type'])
vocabulary.append('dict51')
f19,a19 = run_exp_dep(19,train,test,labels_dep,test_labels,'dict51_ex',option=2,norm_data=True,norm_vec=False,
                         sub_param=norm_data1)
f_score.append(f19)
best.append(a19)
norm_dataset.append(norm_data1['type'])
norm_vect.append('none')
vocabulary.append('dict51_ex')
f20,a20 = run_exp_dep(20,train,test,labels_dep,test_labels,'dict51_ex',option=2,norm_data=True, norm_vec=False,
                         sub_param=norm_data2)
f_score.append(f20)
best.append(a20)
norm_dataset.append(norm_data2['type'])
norm_vect.append('none')
vocabulary.append('dict51_ex')
f21,a21 = run_exp_dep(21,train,test,labels_dep,test_labels,'dict51_ex',option=2,norm_data=True, norm_vec=True,
                         sub_param=norm_data1,  param_norm= norm_vec1)
f_score.append(f21)
best.append(a21)
norm_dataset.append(norm_data1['type'])
norm_vect.append(norm_vec1['type'])
vocabulary.append('dict51_ex')
f22,a22= run_exp_dep(22,train,test,labels_dep,test_labels,'dict51_ex',option = 2,norm_data=True, norm_vec=True,
                         sub_param=norm_data1,  param_norm= norm_vec2)
f_score.append(f22)
best.append(a22)
norm_dataset.append(norm_data1['type'])
norm_vect.append(norm_vec2['type'])
vocabulary.append('dict51_ex')
f23,a23 = run_exp_dep(23,train,test,labels_dep,test_labels,'dict51_ex',option = 2,norm_data=True, norm_vec=True,
                         sub_param=norm_data2,  param_norm= norm_vec1)
f_score.append(f23)
best.append(a23)
norm_dataset.append(norm_data2['type'])
norm_vect.append(norm_vec1['type'])
vocabulary.append('dict51_ex')
f24,a24 = run_exp_dep(24,train,test,labels_dep,test_labels,'dict51_ex',option = 2,norm_data=True, norm_vec=True,
                         sub_param=norm_data2,  param_norm= norm_vec2)
f_score.append(f24)
best.append(a24)
norm_dataset.append(norm_data1['type'])
norm_vect.append(norm_vec2['type'])
vocabulary.append('dict51_ex')

print('End experiments')
l = [str(x) for x in range(1,9)]
data = { 'best_parameter': best, 'f1': f_score, 'is_norm_dataset': norm_dataset, 'is_norm_vec': norm_vect,
         'dict': vocabulary}
df = pd.DataFrame(data, index= l)


df.to_csv('/home/est_posgrado_maria.garcia/Proyecto_tecnologico//Results/Depresion_vec/result_dep_vec_3.csv',
          sep='\t')
