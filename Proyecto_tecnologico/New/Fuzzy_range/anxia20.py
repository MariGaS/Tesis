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


arg1 = [100,100,100,100,100,100,100,100,100,100,500,500,200,100,200,100,200,100,200,100,300,300,32,32] #score1 
arg2 = [100,100,100,100,100,100,200,200,100,100,200,200,100,200,100,200,100,200,100,200,700,700,82,82] #score2
arg3 = [0.99,0.99,0.95,0.95,0.9,0.9,0.99,0.99,0.9,0.9,0.9,0.9,0.99,0.99,0.99,0.99,0.95,0.95,0.95,0.95,0.95,0.95, 0.99,0.95]
arg4 = [3,3,3,3,3,3,1,1,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
arg5 = [False,False,False,False,False,False,True,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False] #fuzzy
arg6 = [True,False,True,False,True,False,True,False,True,False,True,False,True,True,False,False,True,True,False,False,True,False, True, False]
arg7 = [True,True,True,True,True,True,False,False,False,False,True,True,True,False,True,False,True,False,True,False,False,False, True,True]
arg8 = [1,1,1,1,1,1,2,2,2,2,2,2,5,5,5,5,5,5,5,5,1,1,1,1]
print('Begins experiments')


# En este no importa si hay en común
for i in range(24):
    f = run_exp_anxia_sim(i+(24*298),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =arg4[i],add ='positive',groups=[5,10,20], dif = arg5[i], fuzzy= arg6[i],remove_stop=arg7[i], 
                            compress=False, dic =arg8[i], tf = False,clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*299),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =arg4[i],add ='positive',groups=[5,10], dif = arg5[i], fuzzy= arg6[i],remove_stop=arg7[i], 
                            compress=False, dic =arg8[i], tf = False,clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*300),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =arg4[i],add ='positive',groups=[10,20], dif = arg5[i], fuzzy= arg6[i],remove_stop=arg7[i], 
                            compress=False, dic =arg8[i], tf = False,clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*301),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =arg4[i],add ='positive',groups=[5,20], dif = arg5[i], fuzzy= arg6[i],remove_stop=arg7[i], 
                            compress=False, dic =arg8[i], tf = False,clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*302),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =arg4[i],add ='positive',groups=[5], dif = arg5[i], fuzzy= arg6[i],remove_stop=arg7[i], 
                            compress=False, dic =arg8[i], tf = False,clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*303),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =arg4[i],add ='positive',groups=[10], dif = arg5[i], fuzzy= arg6[i],remove_stop=arg7[i], 
                            compress=False, dic =arg8[i], tf = False,clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*304),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =arg4[i],add ='positive',groups=[20], dif = arg5[i], fuzzy= arg6[i],remove_stop=arg7[i], 
                            compress=False, dic =arg8[i], tf = False,clustering= 'None', w_clustering= True)

    f = run_exp_anxia_sim(i+(24*305),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =arg4[i],add ='positive',groups=[5,10,20], dif = arg5[i], fuzzy= arg6[i],remove_stop=arg7[i], 
                            compress=False, dic =arg8[i], tf = True,clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*306),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =arg4[i],add ='positive',groups=[5,10], dif = arg5[i], fuzzy= arg6[i],remove_stop=arg7[i], 
                            compress=False, dic =arg8[i], tf = True,clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*307),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =arg4[i],add ='positive',groups=[10,20], dif = arg5[i], fuzzy= arg6[i],remove_stop=arg7[i], 
                            compress=False, dic =arg8[i], tf = True,clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*308),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =arg4[i],add ='positive',groups=[5,20], dif = arg5[i], fuzzy= arg6[i],remove_stop=arg7[i], 
                            compress=False, dic =arg8[i], tf = True,clustering= 'None', w_clustering= True)

    f = run_exp_anxia_sim(i+(24*309),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =arg4[i],add ='positive',groups=[5], dif = arg5[i], fuzzy= arg6[i],remove_stop=arg7[i], 
                            compress=False, dic =arg8[i], tf = True,clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*310),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =arg4[i],add ='positive',groups=[10], dif = arg5[i], fuzzy= arg6[i],remove_stop=arg7[i], 
                            compress=False, dic =arg8[i], tf = True,clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*311),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=arg3[i],
                            chose =arg4[i],add ='positive',groups=[20], dif = arg5[i], fuzzy= arg6[i],remove_stop=arg7[i], 
                            compress=False, dic =arg8[i], tf = True,clustering= 'None', w_clustering= True)
