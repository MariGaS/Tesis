
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


arg1 = [1000,1500,1500,1700,1000,500, 500, 1500,2000,100,1500,1000, 1000,1500,1500,1700,1000,500, 500, 1500,2000,100,1500,1000] #score1 
arg2 = [1000,1000,1500,1500,1300,500, 1000,1700,2000,500,2000,1500, 1000,1000,1500,1500,1300,500, 1000,1700,2000,500,2000,1500] #score2
arg3 = [0.99]*24 #tolerancia 

arg5 = [True, True, True, True, True, True,True, True, True, True, True, True, False, False, False, False, False, False,False, False, False, False, False, False] #fuzzy


print('Begins experiments')
# En este no importa si hay en com??n
for i in range(24):
    f,x,y,z= run_exp_dep_sim(i+1, test_labels, train_y,num_test, num_train,arg1[i],arg2[i],tau=arg3[i],groups=20,
                            chose =2,dif = False, fuzzy= arg5[i],remove_stop=False,compress=False, dic = 1)

    f,x,y,z= run_exp_dep_sim(i+25, test_labels, train_y,num_test, num_train,arg1[i],arg2[i],tau=arg3[i],groups=20,
                            chose =3,dif = False, fuzzy= arg5[i],remove_stop=False,compress=False, dic = 1)

    f,x,y,z= run_exp_dep_sim(i+49, test_labels, train_y,num_test, num_train,arg1[i],arg2[i],tau=arg3[i],groups=20,
                            chose =2,dif = False, fuzzy= arg5[i],remove_stop=False,compress=True, dic = 1)
    f,x,y,z= run_exp_dep_sim(i+73, test_labels, train_y,num_test, num_train,arg1[i],arg2[i],tau=arg3[i],groups=20,
                            chose =3,dif = False, fuzzy= arg5[i],remove_stop=False,compress=True, dic = 1)


    f,x,y,z= run_exp_dep_sim(i+97, test_labels, train_y,num_test, num_train,arg1[i],arg2[i],tau=arg3[i],groups=20,
                            chose =2,dif = False, fuzzy= arg5[i],remove_stop=True,compress=False, dic = 1)

    f,x,y,z= run_exp_dep_sim(i+121, test_labels, train_y,num_test, num_train,arg1[i],arg2[i],tau=arg3[i],groups=20,
                            chose =3,dif = False, fuzzy= arg5[i],remove_stop=True,compress=False, dic = 1)

    f,x,y,z= run_exp_dep_sim(i+145, test_labels, train_y,num_test, num_train,arg1[i],arg2[i],tau=arg3[i],groups=20,
                            chose =2,dif = False, fuzzy= arg5[i],remove_stop=True,compress=True, dic = 1)
    f,x,y,z= run_exp_dep_sim(i+169, test_labels, train_y,num_test, num_train,arg1[i],arg2[i],tau=arg3[i],groups=20,
                            chose =3,dif = False, fuzzy= arg5[i],remove_stop=True,compress=True, dic = 1)
print('End experiments')