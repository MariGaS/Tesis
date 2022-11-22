from vec_functions import (run_experiment_depression, my_preprocessor)
from text_functions import (get_text_labels,
                            get_text_test)
from time import time
import pandas as pd



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

num_features = [100, 200, 500, 700, 1000, 1500, 1700, 2000, 2500, 3000, 3200, 3500, 4000]
t0 = time()
for i in range(12):
    run_experiment_depression(data_dep, labels_dep, ntrain, test_labels, i+12*15, 1, 1, num_features[i], 'tf_stop', 'NB')
    run_experiment_depression(data_dep, labels_dep, ntrain, test_labels, i+12*16, 1, 1, num_features[i], 'tf', 'NB')
    run_experiment_depression(data_dep, labels_dep, ntrain, test_labels, i+12*17, 1, 1, num_features[i], 'stopwords', 'NB')
    run_experiment_depression(data_dep, labels_dep, ntrain, test_labels, i+12*18, 1, 1, num_features[i], 'tf_idf', 'NB')
    run_experiment_depression(data_dep, labels_dep, ntrain, test_labels, i+12*19, 1, 1, num_features[i], 'binary', 'NB')

    run_experiment_depression(data_dep, labels_dep, ntrain, test_labels, i+12*20, 2, 2, num_features[i], 'tf_stop', 'NB')
    run_experiment_depression(data_dep, labels_dep, ntrain, test_labels, i+12*21, 2, 2, num_features[i], 'tf', 'NB')
    run_experiment_depression(data_dep, labels_dep, ntrain, test_labels, i+12*22, 2, 2, num_features[i], 'stopwords', 'NB')

