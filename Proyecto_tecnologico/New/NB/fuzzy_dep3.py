
from text_functions import (get_text_test,
                            get_text_labels)
from vec_function import run_exp_dep_sim


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
num_test = len(test_labels)
num_train = len(train_y)

arg1 = [100,
700,
700,
200,
100,
150,
150,
170,
200,
170] #score1 
arg2 = [200,
200,
200,
200,
100,
200,
200,
200,
200,
200] #score2
arg3 = [0.9,
0.9,
0.9,
0.9,
0.9,
0.9,
0.9,
0.9,
0.9,
0.9]
arg6 = [False,
False,
True,
True,
False,
False,
True,
False,
True,
True] #fuzzy 
arg7 = [False]*10
arg8 = [4,
4,
4,
4,
2,
4,
4,
2,
2,
2]
print('Begins experiments')
# En este no importa si hay en común
for i in range(10):
    f = run_exp_dep_sim(i+80,test_labels, train_y,num_test,num_train,arg1[i],arg2[i],tau=arg3[i],
                                chose =2,add ='',groups=[], dif = False, fuzzy= arg6[i],remove_stop=arg7[i], 
                                compress=False, dic =arg8[i], tf = False,clustering= 'None', w_clustering= False)