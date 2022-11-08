from text_functions import (get_text_test,
                            get_text_labels)
from vector_fuzzy import run_exp_dep_sim


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

arg3 = [0.99]*24 #tolerancia 
arg5 = [ True, True, False, False]
arg6 = [True, False, True, False]
# En este no importa si hay en común
for i in range(4):
    f = run_exp_dep_sim(i+4*49,test_labels, train_y,num_test,num_train,name_dic='dict5',tau=arg3[i],
                            chose =2, fuzzy= arg5[i],remove_stop=arg6[i], only_bow=False, tf = True,classificator='NB')

    f = run_exp_dep_sim(i+4*50,test_labels, train_y,num_test,num_train,name_dic='dict6',tau=arg3[i],
                            chose =2, fuzzy= arg5[i],remove_stop=arg6[i],  only_bow=False, tf = True,classificator='NB')

    f = run_exp_dep_sim(i+4*51,test_labels, train_y,num_test,num_train,name_dic='dict7',tau=arg3[i],
                            chose =2, fuzzy= arg5[i],remove_stop=arg6[i], only_bow=False, tf = True,classificator='NB')

    f = run_exp_dep_sim(i+4*52,test_labels, train_y,num_test,num_train,name_dic='dict8',tau=arg3[i],
                            chose =2, fuzzy= arg5[i],remove_stop=arg6[i],  only_bow=False, tf = True,classificator='NB')

    f = run_exp_dep_sim(i+4*53,test_labels, train_y,num_test,num_train,name_dic='dict5',tau=arg3[i],
                            chose =2, fuzzy= arg5[i],remove_stop=arg6[i],  only_bow=False, tf = False,classificator='NB')

    f = run_exp_dep_sim(i+4*54,test_labels, train_y,num_test,num_train,name_dic='dict6',tau=arg3[i],
                            chose =2, fuzzy= arg5[i],remove_stop=arg6[i],  only_bow=False, tf = False,classificator='NB')

    f = run_exp_dep_sim(i+4*55,test_labels,train_y,num_test,num_train,name_dic='dict7',tau=arg3[i],
                            chose =2, fuzzy= arg5[i],remove_stop=arg6[i],  only_bow=False, tf = False,classificator='NB')

    f = run_exp_dep_sim(i+4*56,test_labels, train_y,num_test,num_train,name_dic='dict8',tau=arg3[i],
                            chose =2, fuzzy= arg5[i],remove_stop=arg6[i],  only_bow=False, tf = False,classificator='NB')
    f = run_exp_dep_sim(i+4*57,test_labels, train_y,num_test,num_train,name_dic='dict5',tau=arg3[i],
                            chose =3, fuzzy= arg5[i],remove_stop=arg6[i],  only_bow=False, tf = True,classificator='NB')

    f = run_exp_dep_sim(i+4*58,test_labels, train_y,num_test,num_train,name_dic='dict6',tau=arg3[i],
                            chose =3, fuzzy= arg5[i],remove_stop=arg6[i],  only_bow=False, tf = True,classificator='NB')

    f = run_exp_dep_sim(i+4*59,test_labels, train_y,num_test,num_train,name_dic='dict7',tau=arg3[i],
                            chose =3, fuzzy= arg5[i],remove_stop=arg6[i],  only_bow=False, tf = True,classificator='NB')

    f = run_exp_dep_sim(i+4*60,test_labels, train_y,num_test,num_train,name_dic='dict8',tau=arg3[i],
                            chose =3, fuzzy= arg5[i],remove_stop=arg6[i], only_bow=False,  tf = True,classificator='NB')

    f = run_exp_dep_sim(i+4*61,test_labels, train_y,num_test,num_train,name_dic='dict5',tau=arg3[i],
                            chose =3, fuzzy= arg5[i],remove_stop=arg6[i],  only_bow=False, tf = False,classificator='NB')

    f = run_exp_dep_sim(i+4*62,test_labels, train_y,num_test,num_train,name_dic='dict6',tau=arg3[i],
                            chose =3, fuzzy= arg5[i],remove_stop=arg6[i],  only_bow=False, tf = False,classificator='NB')

    f = run_exp_dep_sim(i+4*63,test_labels,train_y,num_test,num_train,name_dic='dict7',tau=arg3[i],
                            chose =3, fuzzy= arg5[i],remove_stop=arg6[i],  only_bow=False, tf = False,classificator='NB')

    f = run_exp_dep_sim(i+4*64,test_labels, train_y,num_test,num_train,name_dic='dict8',tau=arg3[i],
                            chose =3, fuzzy= arg5[i],remove_stop=arg6[i],  only_bow=False, tf = False,classificator='NB')