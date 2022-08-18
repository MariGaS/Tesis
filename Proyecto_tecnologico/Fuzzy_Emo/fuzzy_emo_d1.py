
from text_functions import (get_text_labels,
                            get_text_test)
from vector_fuzzy import run_exp_depresion
import pandas as pd
from itertools import product


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


print('Begin experiment')


f_score = []
best    =[]
norm_dataset = []
vocabulary = []
len_vocab = []
sim = []
print('Begins experiments')

remove = []
l1, l2 = [True, False], ['standard', 'normalize']
output = list(product(l1, l2))

for i in range(16):
    if i%4 == 0:
        #valor de remove_stop
        k = output[i//4][0]
        # valor de subparam
        s = output[i//4][1]
        norm = {'type' : s}
        f, b, l_d = run_exp_depresion(i+1, train, test, labels_dep, test_labels, chose =4,tau=0.98, remove_stop=k,
                                  name_dict='dict5',norm_data=True, only_bow = False,sub_param=norm)
        f_score.append(f)
        best.append(b)
        norm_dataset.append(s)
        remove.append(k)
        vocabulary.append('dict5')
        len_vocab.append(l_d)
        sim.append(0.98)
    if i%4==1:
        #valor de remove_stop
        k = output[i//4][0]
        # valor de subparam
        s = output[i//4][1]
        norm = {'type' : s}
        f, b, l_d = run_exp_depresion(i+1, train, test, labels_dep, test_labels, chose =4,tau=0.98, remove_stop=k,
                                  name_dict='dict6',norm_data=True, only_bow = False,sub_param=norm)
        f_score.append(f)
        remove.append(k)
        best.append(b)
        norm_dataset.append(s)
        vocabulary.append('dict6')
        len_vocab.append(l_d)
        sim.append(0.98)
    if i%4==2:
        #valor de remove_stop
        k = output[i//4][0]
        # valor de subparam
        s = output[i//4][1]
        norm = {'type' : s}
        f, b, l_d = run_exp_depresion(i+1, train, test, labels_dep, test_labels,chose =4, tau=0.98, remove_stop=k,
                                  name_dict='dict7',norm_data=True, only_bow = False,sub_param=norm)
        f_score.append(f)
        best.append(b)
        norm_dataset.append(s)
        vocabulary.append('dict7')
        len_vocab.append(l_d)
        sim.append(0.98)
        remove.append(k)
    if i%4==3:
        #valor de remove_stop
        k = output[i//4][0]
        # valor de subparam
        s = output[i//4][1]
        norm = {'type' : s}
        f, b, l_d = run_exp_depresion(i+1, train, test, labels_dep, test_labels,chose =4, tau=0.98, remove_stop=k,
                                  name_dict='dict8',norm_data=True, only_bow = False,sub_param=norm)
        f_score.append(f)
        best.append(b)
        norm_dataset.append(s)
        vocabulary.append('dict8')
        len_vocab.append(l_d)
        sim.append(0.98)
        remove.append(k)
        
for i in range(16):
    if i%4 == 0:
        #valor de remove_stop
        k = output[i//4][0]
        # valor de subparam
        s = output[i//4][1]
        norm = {'type' : s}
        f, b, l_d = run_exp_depresion(i+16, train, test, labels_dep, test_labels,chose =4, tau=0.95, remove_stop=k,
                                  name_dict='dict5',norm_data=True, only_bow = False,sub_param=norm)
        f_score.append(f)
        best.append(b)
        norm_dataset.append(s)
        vocabulary.append('dict5')
        len_vocab.append(l_d)
        sim.append(0.95)
        remove.append(k)
    if i%4==1:
        #valor de remove_stop
        k = output[i//4][0]
        # valor de subparam
        s = output[i//4][1]
        norm = {'type' : s}
        f, b, l_d = run_exp_depresion(i+16, train, test, labels_dep, test_labels, chose =4, tau=0.95, remove_stop=k,
                                  name_dict='dict6',norm_data=True,only_bow = False, sub_param=norm)
        f_score.append(f)
        best.append(b)
        norm_dataset.append(s)
        vocabulary.append('dict6')
        len_vocab.append(l_d)
        sim.append(0.95)
        remove.append(k)
    if i%4==2:
        #valor de remove_stop
        k = output[i//4][0]
        # valor de subparam
        s = output[i//4][1]
        norm = {'type' : s}
        f, b, l_d = run_exp_depresion(i+16, train, test, labels_dep, test_labels, chose =4,tau=0.95, remove_stop=k,
                                  name_dict='dict7',norm_data=True,only_bow = False, sub_param=norm)
        f_score.append(f)
        best.append(b)
        norm_dataset.append(s)
        vocabulary.append('dict7')
        len_vocab.append(l_d)
        sim.append(0.95)
        remove.append(k)
    if i%4==3:
        #valor de remove_stop
        k = output[i//4][0]
        # valor de subparam
        s = output[i//4][1]
        norm = {'type' : s}
        f, b, l_d = run_exp_depresion(i+16, train, test, labels_dep, test_labels, chose =4,tau=0.95, remove_stop=k,
                                  name_dict='dict8',norm_data=True, only_bow = False,sub_param=norm)
        f_score.append(f)
        best.append(b)
        norm_dataset.append(s)
        vocabulary.append('dict8')
        len_vocab.append(l_d)
        sim.append(0.95)
        remove.append(k)

print('End experiments')
l = [str(x) for x in range(1,33)]
data = { 'best_parameter': best, 'f1': f_score, 'is_norm_dataset': norm_dataset, 'remove': remove,  'dict': vocabulary, 'sim': sim}
df = pd.DataFrame(data, index= l)


df.to_csv('/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Depresion_fuzzy_emo/Result_fuzzy_emo_dep1.csv',sep='\t')


