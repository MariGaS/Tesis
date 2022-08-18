from BOW_functions import (run_experiment_anorexia,
                           run_experiment_depression)
from text_functions import (get_text_labels,
                            get_text_test,
                            get_text_chunk,
                            get_text_test_anorexia)
from time import time
import pandas as pd



### ANOREXIA'S EXPERIMENTS ####
anxia_train = 'Anorexia_2018/Anorexia_Datasets_1/train'
anxia_test = 'Anorexia_2018/Anorexia_Datasets_1/test'

pos = 'positive_examples'
neg = 'negative_examples'

all_pos = []
all_neg = []
for i in range(1, 11):
    path_chunk_pos = anxia_train + '/' + pos + '/chunk' + str(i)
    path_chunk_neg = anxia_train + '/' + neg + '/chunk' + str(i)

    temp1 = get_text_chunk(path_chunk_pos)
    temp2 = get_text_chunk(path_chunk_neg)
    if i == 1:
        all_pos = temp1
        all_neg = temp2
    else:
        for j in range(len(temp1)):
            all_pos[j] += temp1[j]

        for j in range(len(temp2)):
            all_neg[j] += temp2[j]

tr_anorexia = [*all_pos, *all_neg]
tr_label = []
for i in range(len(tr_anorexia)):
    if i < 20:
        tr_label.append(1)
    else:
        tr_label.append(0)

test_url_anxia = []
test_labels_anxia = []

with open('Anorexia_2018/Anorexia_Datasets_1/test/test_golden_truth.txt') as f:
    lines = f.readlines()
    for line in lines:
        test_url_anxia.append(line[:-3])  # only the name of the subject

        test_labels_anxia.append(int(line[-2:-1]))  # only the label
    f.close()

# print(tr_anorexia[0][:10])
# print(test_labels_anxia[:3])

### TEST-EXTRACTION###
test_anxia = []
for i in range(1, 11):

    temp1 = get_text_test_anorexia(anxia_test, test_url_anxia, i)

    if i == 1:
        test_anxia = temp1
        # print("Text extracted from chunk: ", i)
    else:

        for j in range(len(temp1)):
            test_anxia[j] += temp1[j]
        # print("Text extracted from chunk: ", i)

# print(test_anixia[0][:6])

train_anxia = tr_anorexia
test_anxia = test_anxia

labels_anxia = tr_label

data_anxia = train_anxia + test_anxia
ntrain = len(train_anxia)

t0 = time()

f_score = []
min_l   = []
max_l   = []
num_feat_l = []
weight_l= []
best    =[]
chi     =[]

pa1 = {'Data': data_anxia,
       'label': labels_anxia,
       'ntrain': ntrain,
       'min': 1,
       'max': 1,
       'num_feat': 4000,
       'weight': 'tf_stop',
       'norm': True
       }

fa1,a1,chi1 = run_experiment_anorexia(
    test_labels=test_labels_anxia, num_exp=1, param_list=pa1)
f_score.append(fa1)
min_l.append(pa1['min'])
max_l.append(pa1['max'])
num_feat_l.append(pa1['num_feat'])
weight_l.append(pa1['weight'])
best.append(a1)
chi.append(chi1)

pa2 = {'Data': data_anxia,
       'label': labels_anxia,
       'ntrain': ntrain,
       'min': 2,
       'max': 2,
       'num_feat': 3000,
       'weight': 'tf_stop',
       'norm': True
       }

fa2,a2,chi2 = run_experiment_anorexia(
    test_labels=test_labels_anxia, num_exp=2, param_list=pa2)
    
f_score.append(fa2)
min_l.append(pa2['min'])
max_l.append(pa2['max'])
num_feat_l.append(pa2['num_feat'])
weight_l.append(pa2['weight'])
best.append(a2)
chi.append(chi2)

pa3 = {'Data': data_anxia,
       'label': labels_anxia,
       'ntrain': ntrain,
       'min': 1,
       'max': 2,
       'num_feat': 500,
       'weight': 'tf_stop',
       'norm': True
       }

fa3,a3,chi3 = run_experiment_anorexia(
    test_labels=test_labels_anxia, num_exp=3, param_list=pa3)

f_score.append(fa3)
min_l.append(pa3['min'])
max_l.append(pa3['max'])
num_feat_l.append(pa3['num_feat'])
weight_l.append(pa3['weight'])
best.append(a3)
chi.append(chi3)

pa4 = {'Data': data_anxia,
       'label': labels_anxia,
       'ntrain': ntrain,
       'min': 1,
       'max': 3,
       'num_feat': 2000,
       'weight': 'tf_stop',
       'norm': True
       }

fa4,a4,chi4 = run_experiment_anorexia(
    test_labels=test_labels_anxia, num_exp=4, param_list=pa4)
f_score.append(fa4)
min_l.append(pa4['min'])
max_l.append(pa4['max'])
num_feat_l.append(pa4['num_feat'])
weight_l.append(pa4['weight'])
best.append(a4)
chi.append(chi4)

pa5 = {'Data': data_anxia,
       'label': labels_anxia,
       'ntrain': ntrain,
       'min': 1,
       'max': 3,
       'num_feat': 1500,
       'weight': 'tf_idf',
       'norm': True
       }

fa5,a5,chi5 = run_experiment_anorexia(
    test_labels=test_labels_anxia, num_exp=5, param_list=pa5)

f_score.append(fa5)
min_l.append(pa5['min'])
max_l.append(pa5['max'])
num_feat_l.append(pa5['num_feat'])
weight_l.append(pa5['weight'])
best.append(a5)
chi.append(chi5)

pa6 = {'Data': data_anxia,
       'label': labels_anxia,
       'ntrain': ntrain,
       'min': 1,
       'max': 2,
       'num_feat': 1500,
       'weight': 'stopwords',
       'norm': True
       }

fa6,a6,chi6 = run_experiment_anorexia(
    test_labels=test_labels_anxia, num_exp=6, param_list=pa6)
    
f_score.append(fa6)
min_l.append(pa6['min'])
max_l.append(pa6['max'])
num_feat_l.append(pa6['num_feat'])
weight_l.append(pa6['weight'])
best.append(a6)
chi.append(chi6)

pa7 = {'Data': data_anxia,
       'label': labels_anxia,
       'ntrain': ntrain,
       'min': 1,
       'max': 2,
       'num_feat': 1500,
       'weight': 'tf',
       'norm': True
       }

fa7,a7,chi7 = run_experiment_anorexia(
    test_labels=test_labels_anxia, num_exp=7, param_list=pa7)

f_score.append(fa7)
min_l.append(pa7['min'])
max_l.append(pa7['max'])
num_feat_l.append(pa7['num_feat'])
weight_l.append(pa7['weight'])
best.append(a7)
chi.append(chi7)

pa8 = {'Data': data_anxia,
       'label': labels_anxia,
       'ntrain': ntrain,
       'min': 1,
       'max': 3,
       'num_feat': 1500,
       'weight': 'tf',
       'norm': True
       }

fa8,a8,chi8 = run_experiment_anorexia(
    test_labels=test_labels_anxia, num_exp=8, param_list=pa8)

f_score.append(fa8)
min_l.append(pa8['min'])
max_l.append(pa8['max'])
num_feat_l.append(pa8['num_feat'])
weight_l.append(pa8['weight'])
best.append(a8)
chi.append(chi8)

pa9 = {'Data': data_anxia,
       'label': labels_anxia,
       'ntrain': ntrain,
       'min': 1,
       'max': 1,
       'num_feat': 2000,
       'weight': 'tf_idf',
       'norm': True
       }

fa9,a9,chi9 = run_experiment_anorexia(
    test_labels=test_labels_anxia, num_exp=9, param_list=pa9)

f_score.append(fa9)
min_l.append(pa9['min'])
max_l.append(pa9['max'])
num_feat_l.append(pa9['num_feat'])
weight_l.append(pa9['weight'])
best.append(a9)
chi.append(chi9)

pa10 = {'Data': data_anxia,
       'label': labels_anxia,
       'ntrain': ntrain,
       'min': 1,
       'max': 1,
       'num_feat': 1400,
       'weight': 'stopwords',
       'norm': True
       }

fa10,a10,chi10 = run_experiment_anorexia(
    test_labels=test_labels_anxia, num_exp=10, param_list=pa10)

f_score.append(fa10)
min_l.append(pa10['min'])
max_l.append(pa10['max'])
num_feat_l.append(pa10['num_feat'])
weight_l.append(pa10['weight'])
best.append(a10)
chi.append(chi10)

pa11 = {'Data': data_anxia,
       'label': labels_anxia,
       'ntrain': ntrain,
       'min': 2,
       'max': 2,
       'num_feat': 1200,
       'weight': 'tf_idf',
       'norm': True
       }

fa11,a11,chi11 = run_experiment_anorexia(
    test_labels=test_labels_anxia, num_exp=11, param_list=pa11)
f_score.append(fa11)
min_l.append(pa11['min'])
max_l.append(pa11['max'])
num_feat_l.append(pa11['num_feat'])
weight_l.append(pa11['weight'])
best.append(a11)
chi.append(chi11)

pa12 = {'Data': data_anxia,
       'label': labels_anxia,
       'ntrain': ntrain,
       'min': 2,
       'max': 2,
       'num_feat': 1500,
       'weight': 'binary',
       'norm': True
       }

fa12,a12,chi12 = run_experiment_anorexia(
    test_labels=test_labels_anxia, num_exp=12, param_list=pa12)

f_score.append(fa12)
min_l.append(pa12['min'])
max_l.append(pa12['max'])
num_feat_l.append(pa12['num_feat'])
weight_l.append(pa12['weight'])
best.append(a12)
chi.append(chi12)

pa13 = {'Data': data_anxia,
       'label': labels_anxia,
       'ntrain': ntrain,
       'min': 1,
       'max': 2,
       'num_feat': 1200,
       'weight': 'binary',
       'norm': True
       }

fa13,a13,chi13 = run_experiment_anorexia(
    test_labels=test_labels_anxia, num_exp=13, param_list=pa13)
    
f_score.append(fa13)
min_l.append(pa13['min'])
max_l.append(pa13['max'])
num_feat_l.append(pa13['num_feat'])
weight_l.append(pa13['weight'])
best.append(a13)
chi.append(chi13)

pa14 = {'Data': data_anxia,
       'label': labels_anxia,
       'ntrain': ntrain,
       'min': 1,
       'max': 3,
       'num_feat': 1500,
       'weight': 'binary',
       'norm': True
       }

fa14,a14,chi14 = run_experiment_anorexia(
    test_labels=test_labels_anxia, num_exp=14, param_list=pa14)

f_score.append(fa14)
min_l.append(pa14['min'])
max_l.append(pa14['max'])
num_feat_l.append(pa14['num_feat'])
weight_l.append(pa14['weight'])
best.append(a14)
chi.append(chi14)

pa15 = {'Data': data_anxia,
       'label': labels_anxia,
       'ntrain': ntrain,
       'min': 2,
       'max': 3,
       'num_feat': 1500,
       'weight': 'binary',
       'norm': True
       }

fa15,a15,chi15 = run_experiment_anorexia(
    test_labels=test_labels_anxia, num_exp=15, param_list=pa15)

f_score.append(fa15)
min_l.append(pa15['min'])
max_l.append(pa15['max'])
num_feat_l.append(pa15['num_feat'])
weight_l.append(pa15['weight'])
best.append(a15)
chi.append(chi15)

pa16 = {'Data': data_anxia,
       'label': labels_anxia,
       'ntrain': ntrain,
       'min': 2,
       'max': 3,
       'num_feat': 1500,
       'weight': 'tf_idf',
       'norm': True
       }

fa16,a16,chi16 = run_experiment_anorexia(
    test_labels=test_labels_anxia, num_exp=16, param_list=pa16)

f_score.append(fa16)
min_l.append(pa16['min'])
max_l.append(pa16['max'])
num_feat_l.append(pa16['num_feat'])
weight_l.append(pa16['weight'])
best.append(a16)
chi.append(chi16)

pa17 = {'Data': data_anxia,
       'label': labels_anxia,
       'ntrain': ntrain,
       'min': 2,
       'max': 3,
       'num_feat': 1500,
       'weight': 'stopwprds',
       'norm': True
       }

fa17,a17,chi17 = run_experiment_anorexia(
    test_labels=test_labels_anxia, num_exp=17, param_list=pa17)

f_score.append(fa17)
min_l.append(pa17['min'])
max_l.append(pa17['max'])
num_feat_l.append(pa17['num_feat'])
weight_l.append(pa17['weight'])
best.append(a17)
chi.append(chi17)

l = [str(x) for x in range(1,18)] 
data = { 'min': min_l, 'max': max_l, 'num_feat' : num_feat_l, 'weighting': weight_l, 'best_parameter': best, 'f1': f_score}

#data = {'min': [pa1['min'], pa2['min'], pa3['min'], pa4['min'], pa5['min'], pa6['min'], pa7['min'], pa8['min']],
#        'max': [pa1['max'], pa2['max'], pa3['max'], pa4['max'], pa5['max'], pa6['max'], pa7['max'], pa8['max']],
#        'num_feat': [pa1['num_feat'], pa2['num_feat'], pa3['num_feat'], pa4['num_feat'], pa5['num_feat'],
#                     pa6['num_feat'], pa7['num_feat'], pa8['num_feat']],
#        'weighting': [pa1['weight'], pa2['weight'], pa3['weight'], pa4['weight'], pa5['weight'], pa6['weight'],
#                      pa7['weight'], pa8['weight']], 'f1': [fa1, fa2, fa3, fa4, fa5, fa6, fa7, fa8], 'chi': [chi1,chi2,chi3,chi4,chi5,chi6,chi7,chi8], 'best_parameter':[a1,a2,a3,a4,a5,a6,a7,a8]}

df = pd.DataFrame(data, index= l)

# with open('Results/Depresion/all_Result_depre.txt', 'a') as f:
#    dfAsString = df1.to_string(header=True, index=True)
#    f.write(dfAsString)
#    f.close()
df.to_csv('Results/Anorexia/all_Result_anorexia.csv',
          sep='\t')
l1 =[]
l2 =[]
l3 =[]
l4 =[]
l5 = []
l6 = []
l7 =[]
l = []
for i in range(17):
    if f_score[i]>0.75: 
        l1.append(min_l[i])
        l2.append(max_l[i])
        l3.append(num_feat_l[i])
        l4.append(weight_l[i])
        l5.append(best[i])
        l6.append(chi[i])
        l7.append(f_score[i])
        l.append(str(i))

data = { 'min': l1, 'max': l2, 'num_feat' : l3, 'weighting': l4, 'best_parameter': l5,  'f1': l7}

df = pd.DataFrame(data, index= l)

# with open('Results/Depresion/all_Result_depre.txt', 'a') as f:
#    dfAsString = df1.to_string(header=True, index=True)
#    f.write(dfAsString)
#    f.close()
df.to_csv('Results/Anorexia/Best_Result_anorexia.csv',
          sep='\t')
        
