
from text_functions import (get_text_labels,
                            get_text_test)
from vector_sim import run_exp_dep_sim
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
best =[]
print('Begins experiments')
num_feats = [5000,7000, 8000, 8500,9000, 10000,11000,12000,13000, 15000]
      

arg1 = [x for x in range(161,181)]
arg2 = [train_pos, train_pos, train_pos, train_pos, train_pos, train_pos, train_pos, train_pos, train_pos, train_pos,train_pos, train_pos, train_pos, train_pos, train_pos, train_pos, train_pos, train_pos, train_pos, train_pos]
arg3 = [train_neg, train_neg, train_neg, train_neg, train_neg, train_neg, train_neg, train_neg, train_neg, train_neg,train_neg, train_neg, train_neg, train_neg, train_neg, train_neg, train_neg, train_neg, train_neg, train_neg]
arg4 = [test, test, test, test, test, test, test, test, test, test,test, test, test, test, test, test, test, test, test, test]
arg5 = [test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels, test_labels]
arg6 = [train_y,  train_y, train_y, train_y, train_y, train_y, train_y, train_y, train_y, train_y,train_y,  train_y, train_y, train_y, train_y, train_y, train_y, train_y, train_y, train_y]
arg7 = [5000,7000, 8000, 8500,9000, 10000,11000,12000,13000, 15000,5000,7000, 8000, 8500,9000, 10000,11000,12000,13000, 15000]
arg8 = [5000,7000, 8000, 8500,9000, 10000,11000,12000,13000, 15000,5000,7000, 8000, 8500,9000, 10000,11000,12000,13000, 15000]
arg9 = [0.99]*20 #tolerancia
arg10 = [3]*20 #chose
arg11 = [True]*20 #fuzzy
arg12 = [False,False,False,False,False,False,False,False,False,False,True,True,True,True,True,True,True,True,True,True] #remove stop
arg13 = [True]*20 #train_data 
arg14 = [False]*20#compress


with multiprocessing.Pool(processes=20) as pool:
    results = pool.starmap(run_exp_dep_sim, zip(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg10,arg9,arg11,arg12,arg13,arg14))
print(results)
        
f1,a  = zip(*results)

print('End experiments')

#l = [str(x) for x in range(1,11)]
data = { 'num_feats': arg7, 'tol': arg9, 'fuzzy':arg11,'remove_stop' : arg12,'svm':arg13,'compress':arg14, 'f1': f1, 'best': a}
df = pd.DataFrame(data, index= arg1)
print('End experiments')


df.to_csv('/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Depresion_sim/All_result_sim_dep5_1.csv', sep='\t')
        



