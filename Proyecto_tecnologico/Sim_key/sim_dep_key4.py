
from text_functions import (get_text_labels,
                            get_text_test)
from vector_key import run_exp_dep_sim
import pandas as pd
import multiprocessing

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
num_feats = [1000,1500,2000,2500,3000,4000,4500,5000,5500, 6000]


#for i in range(20):
#    if i < 10:
#        f = run_exp_dep_sim(i+121, train_pos, train_neg,test,test_labels, train_y,num_feats[i],num_feats[i],tau=0.99,
#                                 chose =3,fuzzy = True, remove_stop=False,train_data = False, compress=False)
#        f_score.append(f)
#        tol.append(0.99)
#        n_feat.append(num_feats[i])
#        l5.append('False')
#        svm.append('False')
#        com.append('False')
#        vec.append('Fuzzy')

#    else:
#        f= run_exp_dep_sim(i+121, train_pos, train_neg,test,test_labels,train_y, num_feats[i-10],num_feats[i-10], tau=0.99,
#                                chose =3,fuzzy = True, remove_stop=True,train_data = False, compress=False)
#        f_score.append(f)
#        tol.append(0.99)
#        n_feat.append(num_feats[i-10])
#        l5.append('True')
#        svm.append('False')
#        com.append('False')
#        vec.append('Fuzzy')

#for i in range(20):
#    if i < 10:
#        f = run_exp_dep_sim(i+141, train_pos, train_neg,test,test_labels,train_y, num_feats[i],num_feats[i], tau=0.99,
#                                 chose =3,fuzzy = False, remove_stop=False,train_data = False, compress=False)
#        f_score.append(f)
#        tol.append(0.99)
#        n_feat.append(num_feats[i])
#        l5.append('False')
#        svm.append('False')
#        com.append('False')
#        vec.append('No-fuzzy')

#    else:
#        f= run_exp_dep_sim(i+141, train_pos, train_neg,test,test_labels,train_y, num_feats[i-10],num_feats[i-10], tau=0.99,
#                                chose =3,fuzzy = False, remove_stop=True,train_data = False, compress=False)
#        f_score.append(f)
#        tol.append(0.99)
#        n_feat.append(num_feats[i-10])
#        l5.append('True')
#        svm.append('False')
#        com.append('False')
#        vec.append('No-fuzzy')


arg1 = [x for x in range(121,141)]
arg2 = [train_pos, train_pos, train_pos, train_pos, train_pos, train_pos, train_pos, train_pos, train_pos, train_pos,train_pos, train_pos, train_pos, train_pos, train_pos, train_pos, train_pos, train_pos, train_pos, train_pos]
arg3 = [train_neg, train_neg, train_neg, train_neg, train_neg, train_neg, train_neg, train_neg, train_neg, train_neg,train_neg, train_neg, train_neg, train_neg, train_neg, train_neg, train_neg, train_neg, train_neg, train_neg]
arg4 = [test, test, test, test, test, test, test, test, test, test,test, test, test, test, test, test, test, test, test, test]
arg5 = [test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels]
arg6 = [train_y,  train_y, train_y, train_y, train_y, train_y, train_y, train_y, train_y, train_y,train_y,  train_y, train_y, train_y, train_y, train_y, train_y, train_y, train_y, train_y]
arg7 = [1000,1500,2000,2500,3000,4000,4500,5000,5500, 6000,1000,1500,2000,2500,3000,4000,4500,5000,5500, 6000]
arg8 = [1000,1500,2000,2500,3000,4000,4500,5000,5500, 6000,1000,1500,2000,2500,3000,4000,4500,5000,5500, 6000]
arg9 = [0.99]*20 #tolerancia
arg10 = [3]*20 #chose
arg11 = [True]*20 #fuzzy
arg12 = [False,False,False,False,False,False,False,False,False,False,True,True,True,True,True,True,True,True,True,True] #remove stop
arg13 = [False]*20 #train_data 
arg14 = [False]*20#compress


with multiprocessing.Pool(processes=20) as pool:
    results = pool.starmap(run_exp_dep_sim, zip(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg10,arg9,arg11,arg12,arg13,arg14))
print(results)
        
print('End experiments')

#l = [str(x) for x in range(1,11)]
data = { 'num_feats': arg7, 'tol': arg9, 'fuzzy':arg11,'remove_stop' : arg12,'svm':arg13,'compress':arg14, 'f1': results}
df = pd.DataFrame(data, index= arg1)        
print('End experiments')


df.to_csv('/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Depresion_key/All_result_key_dep4.csv', sep='\t')


