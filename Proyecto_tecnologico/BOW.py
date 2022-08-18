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

labels_dep = tr_y_2018 + tr_y_2017

data_dep = train + test
ntrain = len(train)

t0 = time()

param_list1 = {'Data': data_dep,
               'label': labels_dep,
               'ntrain': ntrain,
               'min': 1,
               'max': 1,
               'num_feat': 100,
               'weight': 'stopwords',
               'norm': True
               }
print("Primer experimento")
f1 = run_experiment_depression(
    test_labels=test_labels, num_exp=1, param_list=param_list1)

param_list2 = {'Data': data_dep,
               'label': labels_dep,
               'ntrain': ntrain,
               'min': 2,
               'max': 2,
               'num_feat': 3000,
               'weight': 'stopwords',
               'norm': True
               }

print("Segundo experimento")
f2 = run_experiment_depression(
    test_labels=test_labels, num_exp=2, param_list=param_list2)

param_list3 = {'Data': data_dep,
               'label': labels_dep,
               'ntrain': ntrain,
               'min': 2,
               'max': 2,
               'num_feat': 500,
               'weight': 'tf_idf',
               'norm': True
               }
print("Tercer experimento")
f3 = run_experiment_depression(
    test_labels=test_labels, num_exp=3, param_list=param_list3)

param_list4 = {'Data': data_dep,
               'label': labels_dep,
               'ntrain': ntrain,
               'min': 1,
               'max': 2,
               'num_feat': 500,
               'weight': 'tf_idf',
               'norm': True
               }
print("Cuarto experimento")
f4 = run_experiment_depression(
    test_labels=test_labels, num_exp=4, param_list=param_list4)


param_list5 = {'Data': data_dep,
               'label': labels_dep,
               'ntrain': ntrain,
               'min': 1,
               'max': 3,
               'num_feat': 1500,
               'weight': 'tf_idf',
               'norm': True
               }

print("quinto experimento")
f5 = run_experiment_depression(
    test_labels=test_labels, num_exp=5, param_list=param_list5)

param_list6 = {'Data': data_dep,
               'label': labels_dep,
               'ntrain': ntrain,
               'min': 1,
               'max': 2,
               'num_feat': 2500,
               'weight': 'stopwords',
               'norm': True
               }
print("sexto experimento")
f6 = run_experiment_depression(
    test_labels=test_labels, num_exp=6, param_list=param_list6)


param_list7 = {'Data': data_dep,
               'label': labels_dep,
               'ntrain': ntrain,
               'min': 1,
               'max': 1,
               'num_feat': 2500,
               'weight': 'tf',
               'norm': True
               }
print("septimo experimento")
f7 = run_experiment_depression(
    test_labels=test_labels, num_exp=7, param_list=param_list7)

param_list8 = {'Data': data_dep,
               'label': labels_dep,
               'ntrain': ntrain,
               'min': 2,
               'max': 2,
               'num_feat': 2500,
               'weight': 'tf',
               'norm': True
               }
print("octavo experimento")
f8 = run_experiment_depression(
    test_labels=test_labels, num_exp=8, param_list=param_list8)

param_list9 = {'Data': data_dep,
               'label': labels_dep,
               'ntrain': ntrain,
               'min': 1,
               'max': 2,
               'num_feat': 2500,
               'weight': 'tf',
               'norm': True
               }

print("noveno experimento")
f9 = run_experiment_depression(
    test_labels=test_labels, num_exp=9, param_list=param_list9)

param_list10 = {'Data': data_dep,
                'label': labels_dep,
                'ntrain': ntrain,
                'min': 1,
                'max': 3,
                'num_feat': 2500,
                'weight': 'tf',
                'norm': True
                }

print("decimo experimento")
f10 = run_experiment_depression(
    test_labels=test_labels, num_exp=10, param_list=param_list10)

param_list11 = {'Data': data_dep,
                'label': labels_dep,
                'ntrain': ntrain,
                'min': 2,
                'max': 3,
                'num_feat': 2500,
                'weight': 'tf',
                'norm': True
                }

print("11avo experimento")
f11 = run_experiment_depression(
    test_labels=test_labels, num_exp=11, param_list=param_list11)

param_list15 = {'Data': data_dep,
                'label': labels_dep,
                'ntrain': ntrain,
                'min': 1,
                'max': 1,
                'num_feat': 2500,
                'weight': 'tf_stop',
                'norm': True
                }

print("15avo experimento")
f15 = run_experiment_depression(
    test_labels=test_labels, num_exp=15, param_list=param_list15)

param_list12 = {'Data': data_dep,
                'label': labels_dep,
                'ntrain': ntrain,
                'min': 2,
                'max': 2,
                'num_feat': 2500,
                'weight': 'tf_stop',
                'norm': True
                }

print("12avo experimento")
f12 = run_experiment_depression(
    test_labels=test_labels, num_exp=12, param_list=param_list12)

param_list13 = {'Data': data_dep,
                'label': labels_dep,
                'ntrain': ntrain,
                'min': 1,
                'max': 2,
                'num_feat': 2500,
                'weight': 'tf_stop',
                'norm': True
                }

print("13avo experimento")
f13 = run_experiment_depression(
    test_labels=test_labels, num_exp=13, param_list=param_list13)

param_list14 = {'Data': data_dep,
                'label': labels_dep,
                'ntrain': ntrain,
                'min': 1,
                'max': 3,
                'num_feat': 2500,
                'weight': 'tf_stop',
                'norm': True
                }

print("14avo experimento")
f14 = run_experiment_depression(
    test_labels=test_labels, num_exp=14, param_list=param_list14)



data={'min':[param_list1['min'], param_list2['min'], param_list3['min'], param_list4['min'], param_list5['min'], param_list6['min'],param_list7['min'],param_list8['min'], param_list9['min'], param_list10['min'], param_list11['min'],param_list12['min'],param_list13['min'],param_list14['min'],param_list15['min']],'max':[param_list1['max'], param_list2['max'], param_list3['max'],param_list4['max'], param_list5['max'],param_list6['max'],param_list7['max'], param_list8['max'],param_list9['max'],param_list10['max'],param_list11['max'],param_list12['max'],param_list13['max'],param_list4['max'],param_list15['max']], 'num_feat':[param_list1['num_feat'],param_list2['num_feat'], param_list3['num_feat'], param_list4['num_feat'], param_list5['num_feat'],param_list6['num_feat'],param_list7['num_feat'],param_list8['num_feat'],param_list9['num_feat'],
param_list10['num_feat'],param_list11['num_feat'],param_list12['num_feat'],param_list13['num_feat'],param_list14['num_feat'],param_list15['num_feat']],
      'weighting': [param_list1['weight'],param_list2['weight'],param_list3['weight'],param_list4['weight'],param_list5['weight'],param_list6['weight'],param_list7['weight'],param_list8['weight'],
      param_list9['weight'],param_list10['weight'],param_list11['weight'],param_list12['weight'],param_list13['weight'],param_list14['weight'],param_list15['weight']], 'f1':[f1, f2, f3, f4, f5, f6, f7, f8,f9,f10,f11,f12,f13,f14,f15]}

df=pd.DataFrame(data,index=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'])


#with open('Results/Depresion/all_Result_depre.txt', 'a') as f:
#    dfAsString = df1.to_string(header=True, index=True)
#    f.write(dfAsString)
#    f.close()
df.to_csv('Results/Depresion/all_Result_depre.csv',
          sep='\t')

print("Finalizado depresion")
### ANOREXIA'S EXPERIMENTS ####
anxia_train = 'Anorexia_2018/Anorexia_Datasets_1/train'
anxia_test  = 'Anorexia_2018/Anorexia_Datasets_1/test'

pos = 'positive_examples'
neg = 'negative_examples'

all_pos = []
all_neg = []
for i in range(1,11):
    path_chunk_pos = anxia_train +'/' + pos + '/chunk'+str(i)
    path_chunk_neg = anxia_train +'/' + neg + '/chunk'+str(i)

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
        

test_url_anxia   = []
test_labels_anxia = []

with open('Anorexia_2018/Anorexia_Datasets_1/test/test_golden_truth.txt') as f:
    lines = f.readlines()
    for line in lines: 

        test_url_anxia.append(line[:-3]) #only the name of the subject 

        test_labels_anxia.append(int(line[-2:-1])) #only the label
    f.close()

#print(tr_anorexia[0][:10])
#print(test_labels_anxia[:3])

### TEST-EXTRACTION###
test_anxia = []
for i in range(1,11):

    temp1 = get_text_test_anorexia(anxia_test, test_url_anxia, i)

    if i == 1: 
        test_anxia = temp1
        #print("Text extracted from chunk: ", i)
    else: 

        for j in range(len(temp1)):
            test_anxia[j] += temp1[j]
        #print("Text extracted from chunk: ", i)


#print(test_anixia[0][:6])

train_anxia = tr_anorexia
test_anxia  = test_anxia

labels_anxia = tr_label

data_anxia = train_anxia + test_anxia
ntrain = len(train_anxia)

t0 = time()



pa1 = {'Data': tr_anxia,
       'label': labels_anxia,
       'ntrain': ntrain,
       'min': 1,
       'max': 1,
       'num_feat': 4000,
       'weight': 'tf_stop',
       'norm': True
       }

fa1 = run_experiment_anorexia(
    test_labels=test_labels_anxia, num_exp=1, param_list=pa1)


pa2 = {'Data': tr_anxia,
       'label': labels_anxia,
       'ntrain': ntrain,
       'min': 2,
       'max': 2,
       'num_feat': 3000,
       'weight': 'tf_stop',
       'norm': True
       }

fa2 = run_experiment_anorexia(
    test_labels=test_labels_anxia, num_exp=2, param_list=pa2)

pa3 = {'Data': tr_anxia,
       'label': labels_anxia,
       'ntrain': ntrain,
       'min': 1,
       'max': 2,
       'num_feat': 500,
       'weight': 'tf_stop',
       'norm': True
       }

fa3 = run_experiment_anorexia(
    test_labels=test_labels_anxia, num_exp=3, param_list=pa3)

pa4 = {'Data': tr_anxia,
       'label': labels_anxia,
       'ntrain': ntrain,
       'min': 1,
       'max': 3,
       'num_feat': 2000,
       'weight': 'tf_stop',
       'norm': True
       }

fa4 = run_experiment_anorexia(
    test_labels=test_labels_anxia, num_exp=4, param_list=pa4)


pa5 = {'Data': tr_anxia,
       'label': labels_anxia,
       'ntrain': ntrain,
       'min': 1,
       'max': 3,
       'num_feat': 1500,
       'weight': 'tf_idf',
       'norm': True
       }

fa5 = run_experiment_anorexia(
    test_labels=test_labels_anxia, num_exp=5, param_list=pa5)

pa6 = {'Data': tr_anxia,
       'label': labels_anxia,
       'ntrain': ntrain,
       'min': 1,
       'max': 2,
       'num_feat': 1500,
       'weight': 'stopwords',
       'norm': True
       }

fa6 = run_experiment_anorexia(
    test_labels=test_labels_anxia, num_exp=6, param_list=pa6)

pa7 = {'Data': tr_anxia,
       'label': labels_anxia,
       'ntrain': ntrain,
       'min': 1,
       'max': 2,
       'num_feat': 1500,
       'weight': 'tf',
       'norm': True
       }

fa7 = run_experiment_anorexia(
    test_labels=test_labels_anxia, num_exp=7, param_list=pa7)

pa8 = {'Data': tr_anxia,
       'label': labels_anxia,
       'ntrain': ntrain,
       'min': 1,
       'max': 3,
       'num_feat': 1500,
       'weight': 'tf',
       'norm': True
       }

fa8 = run_experiment_anorexia(
    test_labels=test_labels_anxia, num_exp=8, param_list=pa8)


data={'min':[pa1['min'], pa2['min'], pa3['min'], pa4['min'], pa5['min'], pa6['min'],pa7['min'],pa8['min']],'max':[pa1['max'], pa2['max'], pa3['max'],pa4['max'], pa5['max'],pa6['max'],pa7['max'], pa8['max']], 'num_feat':[pa1['num_feat'],pa2['num_feat'], pa3['num_feat'], pa4['num_feat'], pa5['num_feat'],pa6['num_feat'],pa7['num_feat'],pa8['num_feat']],
      'weighting': [pa1['weight'],pa2['weight'],pa3['weight'],pa4['weight'],pa5['weight'],pa6['weight'],pa7['weight'],pa8['weight']], 'f1':[fa1, fa2, fa3, fa4, fa5, fa6, fa7, fa8]}

df=pd.DataFrame(data,index=['1','2','3','4','5','6','7','8'])


#with open('Results/Depresion/all_Result_depre.txt', 'a') as f:
#    dfAsString = df1.to_string(header=True, index=True)
#    f.write(dfAsString)
#    f.close()
df.to_csv('Results/Anorexia/all_Result_anorexia.csv',
          sep='\t')


