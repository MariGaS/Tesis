from BOW_functions import (run_experiment_anorexia,
                           run_experiment_depression)
from text_functions import (get_text_labels,
                            get_text_test, 
                            get_text_chunk,
                            get_text_test_anorexia)
from time import time
import pandas as pd


train_neg_2017 = 'depression2022/training_data/2017_cases/neg'
train_pos_2017 = 'depression2022/training_data/2017_cases/pos'
train_neg_2018 = 'depression2022/training_data/2018_cases/neg'
train_pos_2018 = 'depression2022/training_data/2018_cases/pos'

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


test_data = 'depression2022/test_data/datos'

test_url = []
test_labels = []

with open('depression2022/test_data/risk_golden_truth.txt') as f:
    lines = f.readlines()
    for line in lines:

        test_url.append(line[:-3])  # only the name of the subject

        test_labels.append(int(line[-2:-1]))  # only the label
    f.close()

test_txt = get_text_test(test_data, test_url)


## EXPERIMENTS  DEPRESSION##
train = tr_txt_2017 + tr_txt_2018
test = test_txt

labels_dep = tr_y_2017 + tr_y_2018

data_dep = train + test
ntrain = len(train)

t0 = time()


f_score = []
min_l   = []
max_l   = []
num_feat_l = []
weight_l= []
best    =[]
chi     =[]

pa1 = {'Data': data_dep,
               'label': labels_dep,
               'ntrain': ntrain,
               'min': 1,
               'max': 1,
               'num_feat': 1000,
               'weight': 'stopwords',
               'norm': True
               }
print("Primer experimento")
f1,a1,chi1 = run_experiment_depression(
    test_labels=test_labels, num_exp=1, param_list=pa1)
    
f_score.append(f1)
min_l.append(pa1['min'])
max_l.append(pa1['max'])
num_feat_l.append(pa1['num_feat'])
weight_l.append(pa1['weight'])
best.append(a1)
chi.append(chi1)

pa2 = {'Data': data_dep,
               'label': labels_dep,
               'ntrain': ntrain,
               'min': 2,
               'max': 2,
               'num_feat': 3500,
               'weight': 'stopwords',
               'norm': True
               }

print("Segundo experimento")
f2,a2,chi2 = run_experiment_depression(
    test_labels=test_labels, num_exp=2, param_list=pa2)
    
f_score.append(f2)
min_l.append(pa2['min'])
max_l.append(pa2['max'])
num_feat_l.append(pa2['num_feat'])
weight_l.append(pa2['weight'])
best.append(a2)
chi.append(chi2)

pa3 = {'Data': data_dep,
               'label': labels_dep,
               'ntrain': ntrain,
               'min': 2,
               'max': 2,
               'num_feat': 1500,
               'weight': 'tf_idf',
               'norm': True
               }
print("Tercer experimento")
f3,a3,chi3 = run_experiment_depression(
    test_labels=test_labels, num_exp=3, param_list=pa3)
    
f_score.append(f3)
min_l.append(pa3['min'])
max_l.append(pa3['max'])
num_feat_l.append(pa3['num_feat'])
weight_l.append(pa3['weight'])
best.append(a3)
chi.append(chi3)

pa4 = {'Data': data_dep,
               'label': labels_dep,
               'ntrain': ntrain,
               'min': 1,
               'max': 2,
               'num_feat': 1500,
               'weight': 'tf_idf',
               'norm': True
               }
print("Cuarto experimento")
f4,a4,chi4 = run_experiment_depression(
    test_labels=test_labels, num_exp=4, param_list=pa4)

f_score.append(f4)
min_l.append(pa4['min'])
max_l.append(pa4['max'])
num_feat_l.append(pa4['num_feat'])
weight_l.append(pa4['weight'])
best.append(a4)
chi.append(chi4)

pa5 = {'Data': data_dep,
               'label': labels_dep,
               'ntrain': ntrain,
               'min': 1,
               'max': 3,
               'num_feat': 2500,
               'weight': 'tf_idf',
               'norm': True
               }

print("quinto experimento")
f5,a5,chi5 = run_experiment_depression(
    test_labels=test_labels, num_exp=5, param_list=pa5)

f_score.append(f5)
min_l.append(pa5['min'])
max_l.append(pa5['max'])
num_feat_l.append(pa5['num_feat'])
weight_l.append(pa5['weight'])
best.append(a5)
chi.append(chi5)

pa6 = {'Data': data_dep,
               'label': labels_dep,
               'ntrain': ntrain,
               'min': 1,
               'max': 2,
               'num_feat': 3000,
               'weight': 'stopwords',
               'norm': True
               }
print("sexto experimento")
f6,a6,chi6 = run_experiment_depression(
    test_labels=test_labels, num_exp=6, param_list=pa6)
    
f_score.append(f6)
min_l.append(pa6['min'])
max_l.append(pa6['max'])
num_feat_l.append(pa6['num_feat'])
weight_l.append(pa6['weight'])
best.append(a6)
chi.append(chi6)

pa7 = {'Data': data_dep,
               'label': labels_dep,
               'ntrain': ntrain,
               'min': 1,
               'max': 1,
               'num_feat': 3000,
               'weight': 'tf',
               'norm': True
               }
print("septimo experimento")
f7,a7,chi7 = run_experiment_depression(
    test_labels=test_labels, num_exp=7, param_list=pa7)
    
f_score.append(f7)
min_l.append(pa7['min'])
max_l.append(pa7['max'])
num_feat_l.append(pa7['num_feat'])
weight_l.append(pa7['weight'])
best.append(a7)
chi.append(chi7)

pa8 = {'Data': data_dep,
               'label': labels_dep,
               'ntrain': ntrain,
               'min': 2,
               'max': 2,
               'num_feat': 2500,
               'weight': 'tf',
               'norm': True
               }
print("octavo experimento")
f8,a8,chi8 = run_experiment_depression(
    test_labels=test_labels, num_exp=8, param_list=pa8)

f_score.append(f8)
min_l.append(pa8['min'])
max_l.append(pa8['max'])
num_feat_l.append(pa8['num_feat'])
weight_l.append(pa8['weight'])
best.append(a8)
chi.append(chi8)

pa9 = {'Data': data_dep,
               'label': labels_dep,
               'ntrain': ntrain,
               'min': 1,
               'max': 2,
               'num_feat': 1800,
               'weight': 'tf',
               'norm': True
               }

print("noveno experimento")
f9, a9, chi9 = run_experiment_depression(
    test_labels=test_labels, num_exp=9, param_list=pa9)

f_score.append(f9)
min_l.append(pa9['min'])
max_l.append(pa9['max'])
num_feat_l.append(pa9['num_feat'])
weight_l.append(pa9['weight'])
best.append(a9)
chi.append(chi9)

pa10 = {'Data': data_dep,
                'label': labels_dep,
                'ntrain': ntrain,
                'min': 1,
                'max': 3,
                'num_feat': 3500,
                'weight': 'tf',
                'norm': True
                }

print("decimo experimento")
f10, a10, chi10 = run_experiment_depression(
    test_labels=test_labels, num_exp=10, param_list=pa10)

f_score.append(f10)
min_l.append(pa10['min'])
max_l.append(pa10['max'])
num_feat_l.append(pa10['num_feat'])
weight_l.append(pa10['weight'])
best.append(a10)
chi.append(chi10)


pa11 = {'Data': data_dep,
                'label': labels_dep,
                'ntrain': ntrain,
                'min': 2,
                'max': 3,
                'num_feat': 3500,
                'weight': 'tf',
                'norm': True
                }

print("11avo experimento")
f11, a11, chi11 = run_experiment_depression(
    test_labels=test_labels, num_exp=11, param_list=pa11)
    
f_score.append(f11)
min_l.append(pa11['min'])
max_l.append(pa11['max'])
num_feat_l.append(pa11['num_feat'])
weight_l.append(pa11['weight'])
best.append(a11)
chi.append(chi11)



pa12 = {'Data': data_dep,
                'label': labels_dep,
                'ntrain': ntrain,
                'min': 2,
                'max': 2,
                'num_feat': 2000,
                'weight': 'tf_stop',
                'norm': True
                }

print("12avo experimento")
f12, a12, chi12 = run_experiment_depression(
    test_labels=test_labels, num_exp=12, param_list=pa12)
    
f_score.append(f12)
min_l.append(pa12['min'])
max_l.append(pa12['max'])
num_feat_l.append(pa12['num_feat'])
weight_l.append(pa12['weight'])
best.append(a12)
chi.append(chi12)

pa13 = {'Data': data_dep,
                'label': labels_dep,
                'ntrain': ntrain,
                'min': 1,
                'max': 2,
                'num_feat': 1500,
                'weight': 'tf_stop',
                'norm': True
                }

print("13avo experimento")
f13, a13, chi13 = run_experiment_depression(
    test_labels=test_labels, num_exp=13, param_list=pa13)

f_score.append(f13)
min_l.append(pa13['min'])
max_l.append(pa13['max'])
num_feat_l.append(pa13['num_feat'])
weight_l.append(pa13['weight'])
best.append(a13)
chi.append(chi13)

pa14 = {'Data': data_dep,
                'label': labels_dep,
                'ntrain': ntrain,
                'min': 1,
                'max': 3,
                'num_feat': 2500,
                'weight': 'tf_stop',
                'norm': True
                }

print("14avo experimento")
f14, a14, chi14 = run_experiment_depression(
    test_labels=test_labels, num_exp=14, param_list=pa14)
    
f_score.append(f14)
min_l.append(pa14['min'])
max_l.append(pa14['max'])
num_feat_l.append(pa14['num_feat'])
weight_l.append(pa14['weight'])
best.append(a14)
chi.append(chi14)

pa15 = {'Data': data_dep,
                'label': labels_dep,
                'ntrain': ntrain,
                'min': 1,
                'max': 1,
                'num_feat': 1700,
                'weight': 'tf_stop',
                'norm': True
                }

print("15avo experimento")
f15, a15, chi15 = run_experiment_depression(
    test_labels=test_labels, num_exp=15, param_list=pa15)

f_score.append(f15)
min_l.append(pa15['min'])
max_l.append(pa15['max'])
num_feat_l.append(pa15['num_feat'])
weight_l.append(pa15['weight'])
best.append(a15)
chi.append(chi15)

l = [str(x) for x in range(1,16)] 
data = { 'min': min_l, 'max': max_l, 'num_feat' : num_feat_l, 'weighting': weight_l, 'best_parameter': best, 'f1': f_score}
df = pd.DataFrame(data, index= l)

#with open('Results/Depresion/all_Result_depre.txt', 'a') as f:
#    dfAsString = df1.to_string(header=True, index=True)
#    f.write(dfAsString)
#    f.close()
df.to_csv('Results/Depresion/all_Result_depre.csv',
          sep='\t')

print("Finalizado depresion")

l1 =[]
l2 =[]
l3 =[]
l4 =[]
l5 = []
l6 = []
l7 =[]
l = []
for i in range(15):
    if f_score[i]>0.60: 
        l1.append(min_l[i])
        l2.append(max_l[i])
        l3.append(num_feat_l[i])
        l4.append(weight_l[i])
        l5.append(best[i])
        l6.append(chi[i])
        l7.append(f_score[i])
        l.append(str(i+1))

data = { 'min': l1, 'max': l2, 'num_feat' : l3, 'weighting': l4, 'best_parameter': l5,  'f1': l7}

df = pd.DataFrame(data, index= l)

# with open('Results/Depresion/all_Result_depre.txt', 'a') as f:
#    dfAsString = df1.to_string(header=True, index=True)
#    f.write(dfAsString)
#    f.close()
df.to_csv('Results/Depresion/Best_Result_depresion.csv',
          sep='\t')
        




