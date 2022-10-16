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


arg1 = [1000,1000, 1000,1500,1500,2000,100,200,100,500,500,300,1000,1000, 1000,1500,1500,2000,100,200,100,500,500,300 ] #score1 
arg2 = [1500,1200,800,1000,1200,2000,100,100,200,200,2000,700,1500,1200,800,1000,1200,2000,100,100,200,200,2000,700] #score2

print('Begins experiments')


# En este no importa si hay en común
for i in range(24):
    f = run_exp_anxia_sim(i+(24*194),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='both',groups=[20], dif = True, fuzzy= True,remove_stop=True, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*195),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='both',groups=[20], dif = False, fuzzy= True,remove_stop=True, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*196),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='both',groups=[20], dif = True, fuzzy= False,remove_stop=True, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*197),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='both',groups=[20], dif = False, fuzzy= False,remove_stop=True, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)

    f = run_exp_anxia_sim(i+(24*198),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='both',groups=[20], dif = True, fuzzy= True,remove_stop=False, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*199),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='both',groups=[20], dif = False, fuzzy= True,remove_stop=False, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*200),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='both',groups=[20], dif = True, fuzzy= False,remove_stop=False, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*201),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='both',groups=[20], dif = False, fuzzy= False,remove_stop=False, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)


    f = run_exp_anxia_sim(i+(24*202),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='positive',groups=[20], dif = True, fuzzy= True,remove_stop=True, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*203),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='positive',groups=[20], dif = False, fuzzy= True,remove_stop=True, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*204),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='positive',groups=[20], dif = True, fuzzy= False,remove_stop=True, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*205),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='positive',groups=[20], dif = False, fuzzy= False,remove_stop=True, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)

    f = run_exp_anxia_sim(i+(24*206),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='positive',groups=[20], dif = True, fuzzy= True,remove_stop=False, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*207),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='positive',groups=[20], dif = False, fuzzy= True,remove_stop=False, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*208),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='positive',groups=[20], dif = True, fuzzy= False,remove_stop=False, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*209),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='positive',groups=[20], dif = False, fuzzy= False,remove_stop=False, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)


    f = run_exp_anxia_sim(i+(24*210),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='negative',groups=[20], dif = True, fuzzy= True,remove_stop=True, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*211),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='negative',groups=[20], dif = False, fuzzy= True,remove_stop=True, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*212),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='negative',groups=[20], dif = True, fuzzy= False,remove_stop=True, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*213),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='negative',groups=[20], dif = False, fuzzy= False,remove_stop=True, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)

    f = run_exp_anxia_sim(i+(24*214),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='negative',groups=[20], dif = True, fuzzy= True,remove_stop=False, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*215),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='negative',groups=[20], dif = False, fuzzy= True,remove_stop=False, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*216),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='negative',groups=[20], dif = True, fuzzy= False,remove_stop=False, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*217),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='negative',groups=[20], dif = False, fuzzy= False,remove_stop=False, 
                            compress=False, dic =1, tf = False, clustering= 'None', w_clustering= True)