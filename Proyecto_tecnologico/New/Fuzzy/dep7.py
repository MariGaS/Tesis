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


arg1 = [2000,2000,1500,1700,2000,2000,1500,1700,1700,1700,1500,1500,1500,1500,1500,1500,1500,1500,1500,1500,1500,1500] #score1 
arg2 = [2000,2000,2000,1500,2000,2000,2000,1500,1500,1500,1500,1000,1500,1000,1500,1500,1500,1500,2000,1000,1000,2000] #score2
arg3 = [0.80, 0.85, 0.90, 0.92, 0.88, 0.83 ]*22 #tolerancia 
arg4 = [True,False,False,False,True,False,True,True,False,True,True,True,True,False,False,False,True,False,False,True,False,True] #fuzzy
arg5 = [4,4,2,4,2,2,2,2,2,4,2,4,4,4,4,2,2,2,4,2,2,4] #fuzzy


print('Begins experiments')
# En este no importa si hay en común
for i in range(22):
    f,x= run_exp_dep_sim(i+1151, test_labels, train_y,num_test, num_train,arg1[i],arg2[0],tau=arg3[0],
                            chose =2,dif = False, fuzzy= arg4[i],remove_stop=False,compress=False, dic =arg5[i], tf = False)
    f,x= run_exp_dep_sim(i+1173, test_labels, train_y,num_test, num_train,arg1[i],arg2[0],tau=arg3[1],
                            chose =2,dif = False, fuzzy= arg4[i],remove_stop=False,compress=False, dic =arg5[i], tf = False)
    f,x= run_exp_dep_sim(i+1195, test_labels, train_y,num_test, num_train,arg1[i],arg2[0],tau=arg3[2],
                            chose =2,dif = False, fuzzy= arg4[i],remove_stop=False,compress=False, dic =arg5[i], tf = False)
    f,x= run_exp_dep_sim(i+1217, test_labels, train_y,num_test, num_train,arg1[i],arg2[0],tau=arg3[3],
                            chose =2,dif = False, fuzzy= arg4[i],remove_stop=False,compress=False, dic =arg5[i], tf = False)
    f,x= run_exp_dep_sim(i+1239, test_labels, train_y,num_test, num_train,arg1[i],arg2[0],tau=arg3[4],
                            chose =2,dif = False, fuzzy= arg4[i],remove_stop=False,compress=False, dic =arg5[i], tf = False)
    f,x= run_exp_dep_sim(i+1261, test_labels, train_y,num_test, num_train,arg1[i],arg2[0],tau=arg3[5],
                            chose =2,dif = False, fuzzy= arg4[i],remove_stop=False,compress=False, dic =arg5[i], tf = False)

print('End experiments')