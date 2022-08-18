from text_functions import (get_text_chunk,
                            get_text_test_anorexia)
from vector_fuzzy import run_exp_anorexia
import pandas as pd
from itertools import product


### ANOREXIA'S EXPERIMENTS ####
anxia_train = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Anorexia_2018/Anorexia_Datasets_1/train'
anxia_test = '/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Anorexia_2018/Anorexia_Datasets_1/test'

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

with open('/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Anorexia_2018/Anorexia_Datasets_1/test/test_golden_truth.txt') as f:
    lines = f.readlines()
    for line in lines:
        test_url_anxia.append(line[:-3])  # only the name of the subject

        test_labels_anxia.append(int(line[-2:-1]))  # only the label
    f.close()

# print(tr_anorexia[0][:10])
# print(test_labels_anxia[:3])

#  ---- TEST-EXTRACTION  --------
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

# --------EXPERIMENTS ----------------#
#norm_data1 ={'type' : 'normalize'}
#norm_data2 ={'type' : 'standard'}

f_score = []
best    =[]
norm_dataset = []
len_vocab = []
vocabulary = []
remove = []
sim = []
print('Begins experiments')

l1, l2 = [True, False], ['standard', 'normalize']
output = list(product(l1, l2))

for i in range(16):
    if i%4 == 0:
        #valor de remove_stop
        k = output[i//4][0]
        # valor de subparam
        s = output[i//4][1]
        norm = {'type' : s}
        f, b, l_d = run_exp_anorexia(i+113, tr_anorexia, test_anxia, tr_label, test_labels_anxia, chose =4,tau=0.95, remove_stop=k,
                                  name_dict='dict1',norm_data=True, sub_param=norm)
        f_score.append(f)
        best.append(b)
        norm_dataset.append(s)
        vocabulary.append('dict1')
        len_vocab.append(l_d)
        remove.append(str(k))
        sim.append(0.95)
    if i%4==1:
        #valor de remove_stop
        k = output[i//4][0]
        # valor de subparam
        s = output[i//4][1]
        norm = {'type' : s}
        f, b, l_d = run_exp_anorexia(i+113, tr_anorexia, test_anxia, tr_label, test_labels_anxia, chose = 4, tau=0.95, remove_stop=k,
                                  name_dict='dict2',norm_data=True, sub_param=norm)
        f_score.append(f)
        best.append(b)
        norm_dataset.append(s)
        vocabulary.append('dict2')
        len_vocab.append(l_d)
        remove.append(str(k))
        sim.append(0.95)
    if i%4==2:
        #valor de remove_stop
        k = output[i//4][0]
        # valor de subparam
        s = output[i//4][1]
        norm = {'type' : s}
        f, b, l_d = run_exp_anorexia(i+113, tr_anorexia, test_anxia, tr_label, test_labels_anxia, chose =4, tau=0.95, remove_stop=k,
                                  name_dict='dict3',norm_data=True, sub_param=norm)
        f_score.append(f)
        best.append(b)
        norm_dataset.append(s)
        vocabulary.append('dict3')
        len_vocab.append(l_d)
        remove.append(str(k))
        sim.append(0.95)
    if i%4==3:
        #valor de remove_stop
        k = output[i//4][0]
        # valor de subparam
        s = output[i//4][1]
        norm = {'type' : s}
        f, b, l_d = run_exp_anorexia(i+113, tr_anorexia, test_anxia, tr_label, test_labels_anxia, chose = 4,tau=0.95, remove_stop=k,
                                  name_dict='dict4',norm_data=True, sub_param=norm)
        f_score.append(f)
        best.append(b)
        norm_dataset.append(s)
        vocabulary.append('dict4')
        len_vocab.append(l_d)
        remove.append(str(k))
        sim.append(0.95)

        
        
print('End experiments')
print(len(f_score), len(best), len(norm_dataset), len(remove), len(vocabulary), len(sim))
l = [str(x) for x in range(1,17)]
data = { 'best_parameter': best, 'f1': f_score, 'is_norm_dataset': norm_dataset, 'remove_stop': remove, 'dict': vocabulary, 'sim': sim}
df = pd.DataFrame(data, index= l)


df.to_csv('/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Anorexia_fuzzy/result_fuzzy_anxia8.csv',sep='\t')

l1 = []
l2 = []
l3 = []
l4 = []
l5 = []
l6 = []
l7 =[]
for i in range(16):
    if f_score[i] > 0.78:
        l1.append(best[i])
        l2.append(norm_dataset[i])
        l3.append(vocabulary[i])
        l4.append(f_score[i])
        l5.append(str(i+1))
        l6.append(remove[i])
        l7.append(sim[i])
data = {'best_parameter': l1, 'f1': l4, 'is_norm_dataset': l2,'remove_stop':l6, 'dict': l3, 'sim': l7}
df = pd.DataFrame(data, index=l5)

# with open('Results/Depresion/all_Result_depre.txt', 'a') as f:
#    dfAsString = df1.to_string(header=True, index=True)
#    f.write(dfAsString)
#    f.close()
df.to_csv('/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Anorexia_fuzzy/Best_Result_fuzzy_anxia8.csv',sep='\t')
