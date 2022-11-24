from vec_functions import (run_experiment_anorexia, my_preprocessor)
from vec_functions import (ekphrasis_processor)
from text_functions import (get_text_chunk,
                            get_text_test_anorexia)

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
data_anxia = [my_preprocessor(x) for x in data_anxia ]

arg1 = [100,500,100, 500, 1000, 800, 200, 1000, 1500, 300] #numfeature 1
arg2 = [100,200,200, 200, 1000, 800, 100, 1200, 1200, 700] #numfeature 2
arg3 = [2]*10    #dic
arg4 = [False,True,True,False, True, False, True, False, False, False]    #dif

for i in range(10):
    run_experiment_anorexia(data_anxia, labels_anxia, ntrain, test_labels_anxia, i+10*70, 
                            2,2,arg1[i],arg2[i],arg3[i],arg4[i],'binary', 'NB','normal_preprocess')
    run_experiment_anorexia(data_anxia, labels_anxia, ntrain, test_labels_anxia, i+10*71, 
                            2,2,arg1[i],arg2[i],arg3[i],arg4[i],'tf_stop', 'NB', 'normal_preprocess')
    run_experiment_anorexia(data_anxia, labels_anxia, ntrain, test_labels_anxia, i+10*72, 
                            2,2,arg1[i],arg2[i],arg3[i],arg4[i],'tf', 'NB', 'normal_preprocess')
    run_experiment_anorexia(data_anxia, labels_anxia, ntrain, test_labels_anxia, i+10*73, 
                            2,2,arg1[i],arg2[i],arg3[i],arg4[i],'stopwords', 'NB', 'normal_preprocess')
    run_experiment_anorexia(data_anxia, labels_anxia, ntrain, test_labels_anxia, i+10*74, 
                            2,2,arg1[i],arg2[i],arg3[i],arg4[i],'tf_idf', 'NB', 'normal_preprocess')
    run_experiment_anorexia(data_anxia, labels_anxia, ntrain, test_labels_anxia, i+10*75, 
                            2,2,arg1[i],arg2[i],arg3[i],arg4[i],'binary', 'svm', 'normal_preprocess')
    run_experiment_anorexia(data_anxia, labels_anxia, ntrain, test_labels_anxia, i+10*76, 
                            2,2,arg1[i],arg2[i],arg3[i],arg4[i],'tf_stop', 'svm', 'normal_preprocess')
    run_experiment_anorexia(data_anxia, labels_anxia, ntrain, test_labels_anxia, i+10*77, 
                            2,2,arg1[i],arg2[i],arg3[i],arg4[i],'tf', 'svm', 'normal_preprocess')
    run_experiment_anorexia(data_anxia, labels_anxia, ntrain, test_labels_anxia, i+10*78, 
                            2,2,arg1[i],arg2[i],arg3[i],arg4[i],'stopwords', 'svm', 'normal_preprocess')
    run_experiment_anorexia(data_anxia, labels_anxia, ntrain, test_labels_anxia, i+10*79, 
                            2,2,arg1[i],arg2[i],arg3[i],arg4[i],'tf_idf', 'svm','normal_preprocess')