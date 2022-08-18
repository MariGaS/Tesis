
from text_functions import (get_text_labels,
                            get_text_test)
from vector_sim import run_exp_dep_sim
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

tr_y_pos  = [*tr_lab_pos_2017, *tr_lab_pos_2018]
tr_y_neg   = [*tr_lab_2017, *tr_lab_2018]
train_y = tr_y_pos +tr_y_neg

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
print('Begins experiments')
num_feats = [5000,7000, 8000, 8500,9000, 10000,11000,12000,13000, 15000]

for i in range(20):
    if i < 10:
        f = run_exp_dep_sim(i+121, train_pos, train_neg,test,test_labels, train_y,num_feats[i],num_feats[i],tau=0.99,
                                 chose =3,fuzzy = True, remove_stop=False,train_data = False, compress=False)
        f_score.append(f)
        tol.append(0.99)
        n_feat.append(num_feats[i])
        l5.append('False')
        svm.append('False')
        com.append('False')
        vec.append('Fuzzy')

    else:
        f= run_exp_dep_sim(i+121, train_pos, train_neg,test,test_labels,train_y, num_feats[i-10],num_feats[i-10], tau=0.99,
                                chose =3,fuzzy = True, remove_stop=True,train_data = False, compress=False)
        f_score.append(f)
        tol.append(0.99)
        n_feat.append(num_feats[i-10])
        l5.append('True')
        svm.append('False')
        com.append('False')
        vec.append('Fuzzy')

for i in range(20):
    if i < 10:
        f = run_exp_dep_sim(i+141, train_pos, train_neg,test,test_labels,train_y, num_feats[i],num_feats[i], tau=0.99,
                                 chose =3,fuzzy = False, remove_stop=False,train_data = False, compress=False)
        f_score.append(f)
        tol.append(0.99)
        n_feat.append(num_feats[i])
        l5.append('False')
        svm.append('False')
        com.append('False')
        vec.append('No-fuzzy')

    else:
        f= run_exp_dep_sim(i+141, train_pos, train_neg,test,test_labels,train_y, num_feats[i-10],num_feats[i-10], tau=0.99,
                                chose =3,fuzzy = False, remove_stop=True,train_data = False, compress=False)
        f_score.append(f)
        tol.append(0.99)
        n_feat.append(num_feats[i-14])
        l5.append('True')
        svm.append('False')
        com.append('False')
        vec.append('No-fuzzy')
        
print('End experiments')


        
print('End experiments')

l = [str(x) for x in range(1,41)]
data = { 'tol': tol, 'num_feats': n_feat,'remove_stop' : l5,'fuzzy':vec,'svm':svm,'compress':com, 'f1': f_score}
df = pd.DataFrame(data, index= l)


df.to_csv('/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Depresion_sim/All_result_sim_dep4.csv', sep='\t')

l1 = []
l2 = []
l3 = []
l4 = []
l6 = []
l7 = []
l8 = []
l9 = []
for i in range(40):
    if f_score[i] > 0.40:
        l1.append(f_score[i])
        l2.append(n_feat[i])
        l3.append(tol[i])
        l4.append(str(i+1))
        l6.append(l5[i])
        l7.append(com[i])
        l8.append(vec[i])
        l9.append(svm[i])

data = { 'tol': l3, 'num_feats': l2, 'remove_stop': l6, 'fuzzy':l8,'svm':l9,'compress':l7,'f1': l1}
df = pd.DataFrame(data, index=l4)

df.to_csv('/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Depresion_sim/Best_Result_sim_dep4.csv', sep='\t')
