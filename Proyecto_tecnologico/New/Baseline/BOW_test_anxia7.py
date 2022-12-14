from vec_functions import (run_experiment_anorexia, my_preprocessor)
from text_functions import (get_text_chunk,
                            get_text_test_anorexia)
from BOW_functions import (ekphrasis_processor)
from time import time
### ANOREXIA'S EXPERIMENTS ####
anxia_train = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Anorexia_2018/Anorexia_Datasets_1/train'
anxia_test = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Anorexia_2018/Anorexia_Datasets_1/test'

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

with open('/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/Anorexia_2018/Anorexia_Datasets_1/test/test_golden_truth.txt') as f:
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
data_anxia = [my_preprocessor(x) for x in data_anxia] 
t0 = time()

num_features = [100, 200, 500, 700, 1000, 1500, 1700, 2000, 2500, 3000, 3200, 3500, 4000]

for i in range(12):
    run_experiment_anorexia(data_anxia, labels_anxia, ntrain, test_labels_anxia, i+12*99, 1, 1, num_features[i], 'tf_stop', 'NB')
    run_experiment_anorexia(data_anxia, labels_anxia, ntrain, test_labels_anxia, i+12*100, 1, 1, num_features[i], 'tf', 'NB')
    run_experiment_anorexia(data_anxia, labels_anxia, ntrain, test_labels_anxia, i+12*101, 1, 1, num_features[i], 'stopwords', 'NB')
    run_experiment_anorexia(data_anxia, labels_anxia, ntrain, test_labels_anxia, i+12*102, 1, 1, num_features[i], 'tf_idf', 'NB')
    run_experiment_anorexia(data_anxia, labels_anxia, ntrain, test_labels_anxia, i+12*103, 1, 1, num_features[i], 'binary', 'NB')

    run_experiment_anorexia(data_anxia, labels_anxia, ntrain, test_labels_anxia, i+12*104, 2, 2, num_features[i], 'tf_stop', 'NB')
    run_experiment_anorexia(data_anxia, labels_anxia, ntrain, test_labels_anxia, i+12*105, 2, 2, num_features[i], 'tf', 'NB')
    run_experiment_anorexia(data_anxia, labels_anxia, ntrain, test_labels_anxia, i+12*106, 2, 2, num_features[i], 'stopwords', 'NB')
    run_experiment_anorexia(data_anxia, labels_anxia, ntrain, test_labels_anxia, i+12*107, 2, 2, num_features[i], 'tf_idf', 'NB')
    run_experiment_anorexia(data_anxia, labels_anxia, ntrain, test_labels_anxia, i+12*108, 2, 2, num_features[i], 'binary', 'NB')

    run_experiment_anorexia(data_anxia, labels_anxia, ntrain, test_labels_anxia, i+12*109, 3, 3, num_features[i], 'tf_stop', 'NB')
    run_experiment_anorexia(data_anxia, labels_anxia, ntrain, test_labels_anxia, i + 12*110, 3, 3, num_features[i], 'tf', 'NB')
    run_experiment_anorexia(data_anxia, labels_anxia, ntrain, test_labels_anxia, i + 12*111, 3, 3, num_features[i], 'stopwords', 'NB')
    run_experiment_anorexia(data_anxia, labels_anxia, ntrain, test_labels_anxia, i + 12* 112, 3, 3, num_features[i], 'tf_idf', 'NB')
    run_experiment_anorexia(data_anxia, labels_anxia, ntrain, test_labels_anxia, i + 12*113, 3, 3, num_features[i], 'binary', 'NB')
