from vec_functions import (run_experiment_depression, my_preprocessor)
from text_functions import (get_text_labels,
                            get_text_test)



train_neg_2017 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/depression2022/training_data/2017_cases/neg'
train_pos_2017 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/depression2022/training_data/2017_cases/pos'
train_neg_2018 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/depression2022/training_data/2018_cases/neg'
train_pos_2018 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/depression2022/training_data/2018_cases/pos'


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


test_data = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/depression2022/test_data/datos'

test_url = []
test_labels = []

with open('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/depression2022/test_data/risk_golden_truth.txt') as f:
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

data_dep = [my_preprocessor(x) for x in data_dep ]

arg1 = [300,500,500,100, 700, 200, 1000, 80, 300] #numfeature 1
arg2 = [300,100,200,100, 200, 200, 1000, 30, 200] #numfeature 2
arg3 = [1]*10   #dic
arg4 = [False]*10    #dif

for i in range(10):

    run_experiment_depression(data_dep, labels_dep, ntrain, test_labels, i+10*25, 
                            2,2,arg1[i],arg2[i],arg3[i],arg4[i],'binary', 'svm')
    run_experiment_depression(data_dep, labels_dep, ntrain, test_labels, i+10*26, 
                            2,2,arg1[i],arg2[i],arg3[i],arg4[i],'tf_stop', 'svm')
    run_experiment_depression(data_dep, labels_dep, ntrain, test_labels, i+10*27, 
                            2,2,arg1[i],arg2[i],arg3[i],arg4[i],'tf', 'svm')
    run_experiment_depression(data_dep, labels_dep, ntrain, test_labels, i+10*28, 
                            2,2,arg1[i],arg2[i],arg3[i],arg4[i],'stopwords', 'svm')
    run_experiment_depression(data_dep, labels_dep, ntrain, test_labels, i+10*29, 
                            2,2,arg1[i],arg2[i],arg3[i],arg4[i],'tf_idf', 'svm')