from text_functions import (get_text_labels,
                            get_text_test)
from bigrams_vector import run_exp_dep_bi
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

test_data = 'depression2022/test_data/datos'

test_url = []
test_labels = []

with open('depression2022/test_data/risk_golden_truth.txt') as f:
    lines = f.readlines()
    for line in lines:
        if line[0:11] != 'subject3958':
            test_url.append(line[:-3])  # only the name of the subject

            test_labels.append(int(line[-2:-1]))  # only the label
    f.close()

test_txt = get_text_test(test_data, test_url)
# negative and positive training data
train_neg = [*tr_neg_2017, *tr_neg_2018]
train_pos = [*tr_pos_2017, *tr_pos_2018]
# test data
test = test_txt


f_score = []
tol = []
n_feat = []
n_bigram = []
remove = []
print('Begins experiments')

feat_bigram = [[1000,400], [1000,1500], [1000,2000],[1000,3000],[2000,400],
               [2000,1000], [2000,2000], [2200,400], [2200,1000], [2200,2000], [2200,3000],
               [3000,1000], [3000,1500], [3000,2000], [3000,3500], [3000,4000], [3000,5000],
               [4000,1000], [4000,1500], [4000,2000], [4000,3500], [4000,4000], [4000,5000],
               [500,1000], [500,2000], [500,3000],  [1000,2000], [1000,3000]]

for i in range(56):
    if i < 28:
        f = run_exp_dep_bi(i+1,train_pos,train_neg,test,test_labels, feat_bigram[i][0], feat_bigram[i][1],
                             tau=0.99,remove_stop=True)
        f_score.append(f)
        tol.append(0.99)
        remove.append('True')
        n_feat.append(feat_bigram[i][0])
        n_bigram.append(feat_bigram[i][1])

    else :
        f = run_exp_dep_bi(i+1, train_pos, train_neg,test,test_labels, feat_bigram[i-28][0], feat_bigram[i-28][1],
                             tau=0.99,remove_stop=False)
        f_score.append(f)
        tol.append(0.99)
        remove.append('False')
        n_feat.append(feat_bigram[i-28][0])
        n_bigram.append(feat_bigram[i-28][1])

print('End experiments')

l = [str(x) for x in range(1,57)]
data = { 'num_feats': n_feat,'num_bigrams': n_bigram,'remove_stop':remove,'tol': tol,  'f1': f_score}
df = pd.DataFrame(data, index= l)

df.to_csv('Results/Anorexia_bi/All_result_bi_anxia.csv',sep='\t')

l1 = []
l2 = []
l3 = []
l4 = []
l5 = []
l6 = []
for i in range(56):
    if f_score[i] > 0.52:
        l1.append(f_score[i])
        l2.append(n_feat[i])
        l3.append(tol[i])
        l5.append(n_bigram[i])
        l6.append(remove[i])
        l4.append(str(i+1))

data = { 'num_feats': l2,'num_bigrams': l5,'remove_stop':l6,'tol': l3,  'f1': l1}
df = pd.DataFrame(data, index=l4)

df.to_csv('Results/Anorexia_bi/Best_Result_bi_anxia.csv',sep='\t')