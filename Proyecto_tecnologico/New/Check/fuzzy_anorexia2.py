from vec_function import run_exp_anxia_sim


tr_label = []
for i in range(152):
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

num_test = len(test_labels_anxia)
num_train= 152

arg1 = [100,100,100,100,100,100,100,100,100,100,500,500,200,100,200,100,200,100,200,100] #score1 
arg2 = [100,100,100,100,100,100,200,200,100,100,200,200,100,200,100,200,100,200,100,200] #score2
arg3 = [3,
3,
3,
3,
3,
3,
1,
1,
3,
3,
3,
3,
3,
3,
3,
3,
3,
3,
3,
3]  #word embedding 
arg4 = [0.99,0.99,0.95,0.95,0.9,0.9,0.99,0.99,0.9,0.9,0.9,0.9,0.99,0.99,0.99,0.99,0.95,0.95,0.95,0.95]#tolerancia 
arg5 = [] # groups
arg6 = 'None' #add
arg7 = 'None' #clustering 
arg8 = [False,
False,
False,
False,
False,
False,
True,
True,
False,
False,
False,
False,
False,
False,
False,
False,
False,
False,
False,
False]# different
arg9 = [True,
False,
True,
False,
True,
False,
True,
False,
True,
False,
True,
False,
True,
True,
False,
False,
True,
True,
False,
False] #fuzzy 
arg10 = [True,
True,
True,
True,
True,
True,
False,
False,
False,
False,
True,
True,
True,
False,
True,
False,
True,
False,
True,
False]#remove 
arg11 = [False]*20 #compress 
arg12 =[1,
1,
1,
1,
1,
1,
2,
2,
2,
2,
2,
2,
5,
5,
5,
5,
5,
5,
5,
5]#dic 
arg13 = [False]*20#tf is false  
arg14 = [False]*20 #wclustering

for i in range(20):
    f = run_exp_anxia_sim(i+20,test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=arg4[i],
                            chose =arg3[i],add =arg6,groups=arg5, dif = arg8[i], fuzzy= arg9[i],remove_stop=arg10[i], 
                            compress=arg11[i], dic =arg12[i], tf = arg13[i], clustering= arg7, w_clustering= arg14[i])