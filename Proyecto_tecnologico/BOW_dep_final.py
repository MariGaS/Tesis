from BOW_functions import (run_experiment_depression)
from text_functions import (get_text_labels,
                            get_text_test)
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
               'min': 1,
               'max': 1,
               'num_feat': 500,
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
               'min': 1,
               'max': 1,
               'num_feat': 100,
               'weight': 'stopwords',
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
    test_labels=test_labels, num_exp=4, param_list=pa15)

f_score.append(f15)
min_l.append(pa15['min'])
max_l.append(pa15['max'])
num_feat_l.append(pa15['num_feat'])
weight_l.append(pa15['weight'])
best.append(a15)
chi.append(chi15)

l = [str(x) for x in range(1,5)] 
data = { 'min': min_l, 'max': max_l, 'num_feat' : num_feat_l, 'weighting': weight_l, 'best_parameter': best, 'f1': f_score}
df = pd.DataFrame(data, index= l)

#with open('Results/Depresion/all_Result_depre.txt', 'a') as f:
#    dfAsString = df1.to_string(header=True, index=True)
#    f.write(dfAsString)
#    f.close()
df.to_csv('Results/Depresion/Result_depre_final.csv',
          sep='\t')

print("Finalizado depresion")






