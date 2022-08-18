from text_functions import (get_text_chunk,
                            get_text_test_anorexia)
from vector_key import run_exp_anxia_sim

import pandas as pd

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


f_score = []
tol = []
n_feat = []
l5 =[]
svm = []
com =[]
vec=[]
best=[]
print('Begins experiments')
num_feats = [500, 1000, 2000, 2200, 2500, 3000, 4500, 5000, 5100]


for i in range(18):
    if i < 9:
        f,a = run_exp_anxia_sim(i+145, all_pos, all_neg,test_anxia,test_labels_anxia, tr_label,num_feats[i],num_feats[i],tau=0.99,
                                          chose =3,fuzzy= True,remove_stop=False,train_data = True, compress=False)
        f_score.append(f)
        tol.append(0.99)
        n_feat.append(num_feats[i])
        l5.append('False')
        svm.append('True')
        com.append('False')
        vec.append('Fuzzy')
        best.append(a)

    else:
        f,a= run_exp_anxia_sim(i+145, all_pos, all_neg,test_anxia,test_labels_anxia, tr_label,num_feats[i-9],num_feats[i-9], tau=0.99,
                                          chose =3,fuzzy= True,remove_stop=True, train_data = True, compress = False)
        f_score.append(f)
        tol.append(0.99)
        n_feat.append(num_feats[i-9])
        l5.append('True')
        svm.append('True')
        com.append('False')
        vec.append('Fuzzy')
        best.append(a)

for i in range(18):
    if i < 9:
        f,a = run_exp_anxia_sim(i+163, all_pos, all_neg,test_anxia,test_labels_anxia, tr_label,num_feats[i],num_feats[i],tau=0.99,
                                          chose =3,fuzzy= False,remove_stop=False,train_data = True, compress=False)
        f_score.append(f)
        tol.append(0.99)
        n_feat.append(num_feats[i])
        l5.append('False')
        svm.append('True')
        com.append('False')
        vec.append('No-Fuzzy')
        best.append(a)

    else:
        f,a= run_exp_anxia_sim(i+163, all_pos, all_neg,test_anxia,test_labels_anxia,tr_label, num_feats[i-9],num_feats[i-9], tau=0.99,
                                          chose =3,fuzzy= False,remove_stop=True, train_data = True, compress = False)
        f_score.append(f)
        tol.append(0.99)
        n_feat.append(num_feats[i-9])
        l5.append('True')
        svm.append('True')
        com.append('False')
        vec.append('No-fuzzy')
        best.append(a)
        
print('End experiments')

l = [str(x) for x in range(1,37)]
data = { 'tol': tol, 'num_feats': n_feat, 'remove_stop':l5,'compress':com, 'svm': svm,'fuzzy':vec, 'best':best,'f1': f_score}
df = pd.DataFrame(data, index= l)


df.to_csv('/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Anorexia_key/All_result_key_anxia5.csv',sep='\t')

l1 = []
l2 = []
l3 = []
l4 = []
l6 =[]
l7 = []
l8 = []
l9 = []
for i in range(36):
    if f_score[i] > 0.78:
        l1.append(f_score[i])
        l2.append(n_feat[i])
        l3.append(tol[i])
        l4.append(str(i+1))
        l6.append(l5[i])
        l7.append(com[i])
        l8.append(svm[i])
        l9.append(vec[i])
        l10.append(best[i])
data = { 'tol': l3, 'num_feats': l2, 'remove_stop': l6,'compress':l7, 'svm':l8,'fuzzy':l9,'best':l10,'f1': l1}
df = pd.DataFrame(data, index=l4)

df.to_csv('/home/est_posgrado_maria.garcia/Proyecto_tecnologico/Results/Anorexia_key/Best_Result_key_anxia5.csv',sep='\t')
