from dic import get_dictionary, get_list_key

##                                         USER VERSION OF THE DICTIONARIES OF THE USERS                        #
#---------------------------ANOREXIA-------------------#
#version 1
user_pos_anxia1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/anxia_pos_ver3'
user_neg_anxia1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/anxia_neg_ver3'
#version 2
user_pos_anxia2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/anxia_pos_ver4'
user_neg_anxia2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/anxia_neg_ver4'

user_pos_anxia3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/anxia_pos_ver5'
user_neg_anxia3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/anxia_neg_ver5'
#version 2
user_pos_anxia4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/anxia_pos_ver6'
user_neg_anxia4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/anxia_neg_ver6'

#---------------------------DEPRESSION-----------------#
#version 1
user_pos_dep1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_pos_ver3'
user_neg_dep1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_neg_ver3'
#version 2
user_pos_dep2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_pos_ver4'
user_neg_dep2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_neg_ver4'

user_pos_dep3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_pos_ver5'
user_neg_dep3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_neg_ver5'
#version 2
user_pos_dep4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_pos_ver6'
user_neg_dep4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/User_dictionary/dep_neg_ver6'



pos_anxia1 = get_list_key(user_pos_anxia1)
neg_anxia1 = get_list_key(user_neg_anxia1)


pos_anxia2 = get_list_key(user_pos_anxia2)
neg_anxia2 = get_list_key(user_neg_anxia2)

arg1 = [1000,1000, 1000,1500,1500,2000,100,200,100,500,500,300,1000,1000, 1000,1500,1500,2000,100,200,100,500,500,300 ] #score1 
arg2 = [1500,1200,800,1000,1200,2000,100,100,200,200,2000,700,1500,1200,800,1000,1200,2000,100,100,200,200,2000,700] #score2

for i in range(len(arg1)):
    name_path1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_user/Difference/Lowercase/pos_anxia_' +str(arg1[i]) +'_'+ str(arg2[i]) + '.txt'
    name_path2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_user/Difference/Lowercase/neg_anxia_' +str(arg1[i]) +'_'+ str(arg2[i])+ '.txt'

    get_dictionary(name_path1, name_path2, pos_anxia2, neg_anxia2, arg1[i], arg2[i], no_distintic= False, con = False)

    name_path3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_user/Difference/Uppercase/pos_anxia_' +str(arg1[i]) +'_'+ str(arg2[i]) + '.txt'
    name_path4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_user/Difference/Uppercase/neg_anxia_' +str(arg1[i]) +'_'+ str(arg2[i]) + '.txt'

    get_dictionary(name_path3, name_path4, pos_anxia1, neg_anxia1, arg1[i], arg2[i], no_distintic= False, con = False)



    name_path5 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_user/No_difference/Uppercase/pos_anxia_' +str(arg1[i]) +'_'+ str(arg2[i])+ '.txt'
    name_path6 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_user/No_difference/Uppercase/neg_anxia_' +str(arg1[i]) +'_'+ str(arg2[i]) + '.txt'

    get_dictionary(name_path5, name_path6, pos_anxia1, neg_anxia1, arg1[i], arg2[i], no_distintic= True, con = False)

    name_path7 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_user/No_difference/Lowercase/pos_anxia_' +str(arg1[i]) +'_'+ str(arg2[i]) + '.txt'
    name_path8 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_user/No_difference/Lowercase/neg_anxia_' +str(arg1[i]) +'_'+ str(arg2[i]) + '.txt'

    get_dictionary(name_path7, name_path8, pos_anxia2, neg_anxia2, arg1[i], arg2[i], no_distintic= True, con = False)

#DICTIONARIES DEPRESSION 
pos_dep1 = get_list_key(user_pos_dep1)
neg_dep1 = get_list_key(user_neg_dep1)

pos_dep2 = get_list_key(user_pos_dep2)
neg_dep2 = get_list_key(user_neg_dep2)


for i in range(len(arg1)):
    name_path1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_user/No_difference/Lowercase/pos_dep_' +str(arg1[i]) +'_'+ str(arg2[i]) + '.txt'
    name_path2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_user/No_difference/Lowercase/neg_dep_' +str(arg1[i]) +'_'+ str(arg2[i]) + '.txt'

    get_dictionary(name_path1, name_path2, pos_dep2, neg_dep2, arg1[i], arg2[i], no_distintic= True, con = False)

    name_path3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_user/No_difference/Uppercase/pos_dep_' +str(arg1[i]) +'_'+ str(arg2[i])+ '.txt'
    name_path4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_user/No_difference/Uppercase/neg_dep_' +str(arg1[i]) +'_'+ str(arg2[i]) + '.txt'

    get_dictionary(name_path3, name_path4, pos_dep1, neg_dep1, arg1[i], arg2[i], no_distintic= True, con = False)

    name_path5 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_user/Difference/Lowercase/pos_dep_' +str(arg1[i]) +'_'+ str(arg2[i]) + '.txt'
    name_path6 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_user/Difference/Lowercase/neg_dep_' +str(arg1[i]) +'_'+ str(arg2[i]) + '.txt'

    get_dictionary(name_path5, name_path6, pos_dep2, neg_dep2, arg1[i], arg2[i], no_distintic= False, con = False)

    name_path7 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_user/Difference/Uppercase/pos_dep_' +str(arg1[i]) +'_'+ str(arg2[i]) + '.txt'
    name_path8 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_user/Difference/Uppercase/neg_dep_' + str(arg1[i]) +'_'+ str(arg2[i])+ '.txt'

    get_dictionary(name_path7, name_path8, pos_dep1, neg_dep1, arg1[i], arg2[i], no_distintic= False, con = False)




