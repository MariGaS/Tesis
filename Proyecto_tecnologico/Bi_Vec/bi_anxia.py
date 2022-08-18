from text_functions import (get_text_chunk,
                            get_text_test_anorexia)
from bigrams_vector import run_exp_anxia_bi

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
n_bigram = []
remove = []
print('Begins experiments')

feat_bigram = [[500,1000], [1000,300], [1000,500], [1000,400], [1000,1500], [1000,2000],[1000,3000],[2000,400],
               [2000,1000], [2000,2000], [2200,400], [2200,700],[2200,1000], [2200,1500], [2200,2000], [2200,3000],
               [3000,100], [3000,1000], [3000,1500], [3000,2000], [3000,2500], [3000,3500], [3000,4000], [3000,5000],
               [4000,100], [4000,1000], [4000,1500], [4000,2000], [4000,2500], [4000,3500], [4000,4000], [4000,5000],
               [100,1000], [500,1000], [500,2000], [100,3000], [500,3000], [500,1500], [1000,2000], [1000,3000]]

for i in range(80):
    if i < 40:
        f = run_exp_anxia_bi(i+1, all_pos, all_neg,test_anxia,test_labels_anxia, feat_bigram[i][0], feat_bigram[i][1],
                             tau=0.99,remove_stop=True)
        f_score.append(f)
        tol.append(0.99)
        remove.append('True')
        n_feat.append(feat_bigram[i][0])
        n_bigram.append(feat_bigram[i][1])

    else :
        f = run_exp_anxia_bi(i+1, all_pos, all_neg,test_anxia,test_labels_anxia, feat_bigram[i-40][0], feat_bigram[32-i][1],
                             tau=0.99,remove_stop=False)
        f_score.append(f)
        tol.append(0.99)
        remove.append('False')
        n_feat.append(feat_bigram[i-40][0])
        n_bigram.append(feat_bigram[i-40][1])

print('End experiments')

l = [str(x) for x in range(1,81)]
data = { 'num_feats': n_feat,'num_bigrams': n_bigram,'remove_stop':remove,'tol': tol,  'f1': f_score}
df = pd.DataFrame(data, index= l)

df.to_csv('Results/Anorexia_bi/All_result_bi_anxia.csv',sep='\t')

l1 = []
l2 = []
l3 = []
l4 = []
l5 = []
l6 = []
for i in range(80):
    if f_score[i] > 0.79:
        l1.append(f_score[i])
        l2.append(n_feat[i])
        l3.append(tol[i])
        l5.append(n_bigram[i])
        l6.append(remove[i])
        l4.append(str(i+1))

data = { 'num_feats': l2,'num_bigrams': l5,'remove_stop':l6,'tol': l3,  'f1': l1}
df = pd.DataFrame(data, index=l4)

df.to_csv('Results/Anorexia_bi/Best_Result_bi_anxia.csv',sep='\t')
