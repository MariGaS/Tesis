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
    f = run_exp_anxia_sim(i+(24*242),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='both',groups=[5], dif = True, fuzzy= True,remove_stop=True, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*243),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='both',groups=[5], dif = False, fuzzy= True,remove_stop=True, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*244),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='both',groups=[5], dif = True, fuzzy= False,remove_stop=True, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*245),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='both',groups=[5], dif = False, fuzzy= False,remove_stop=True, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)

    f = run_exp_anxia_sim(i+(24*246),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='both',groups=[5], dif = True, fuzzy= True,remove_stop=False, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*247),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='both',groups=[5], dif = False, fuzzy= True,remove_stop=False, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*248),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='both',groups=[5], dif = True, fuzzy= False,remove_stop=False, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*249),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='both',groups=[5], dif = False, fuzzy= False,remove_stop=False, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)


    f = run_exp_anxia_sim(i+(24*250),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='positive',groups=[5], dif = True, fuzzy= True,remove_stop=True, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*251),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='positive',groups=[5], dif = False, fuzzy= True,remove_stop=True, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*252),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='positive',groups=[5], dif = True, fuzzy= False,remove_stop=True, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*253),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='positive',groups=[5], dif = False, fuzzy= False,remove_stop=True, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)

    f = run_exp_anxia_sim(i+(24*254),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='positive',groups=[5], dif = True, fuzzy= True,remove_stop=False, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*255),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='positive',groups=[5], dif = False, fuzzy= True,remove_stop=False, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*256),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='positive',groups=[5], dif = True, fuzzy= False,remove_stop=False, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*257),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='positive',groups=[5], dif = False, fuzzy= False,remove_stop=False, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)


    f = run_exp_anxia_sim(i+(24*258),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='negative',groups=[5], dif = True, fuzzy= True,remove_stop=True, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*259),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='negative',groups=[5], dif = False, fuzzy= True,remove_stop=True, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*260),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='negative',groups=[5], dif = True, fuzzy= False,remove_stop=True, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*261),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='negative',groups=[5], dif = False, fuzzy= False,remove_stop=True, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)

    f = run_exp_anxia_sim(i+(24*262),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='negative',groups=[5], dif = True, fuzzy= True,remove_stop=False, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*263),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='negative',groups=[5], dif = False, fuzzy= True,remove_stop=False, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*264),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='negative',groups=[5], dif = True, fuzzy= False,remove_stop=False, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)
    f = run_exp_anxia_sim(i+(24*265),test_labels_anxia, tr_label,num_test,num_train,arg1[i],arg2[i],tau=0.95,
                            chose =1,add ='negative',groups=[5], dif = False, fuzzy= False,remove_stop=False, 
                            compress=False, dic =2, tf = False, clustering= 'None', w_clustering= True)