from dic import get_dictionary, get_list_key

##                                                   CONCATANTE ALL THE TEXT 
#version 1
con_pos_anxia1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_pos_ver3'
con_neg_anxia1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_neg_ver3'
#version 2
con_pos_anxia2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_pos_ver4'
con_neg_anxia2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_neg_ver4'

con_pos_anxia3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_pos_ver5'
con_neg_anxia3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_neg_ver5'
#version 2
con_pos_anxia4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_pos_ver6'
con_neg_anxia4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/anxia_neg_ver6'

#---------------------------DEPRESSION-----------------#
#version 1
con_pos_dep1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_pos_ver3'
con_neg_dep1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_neg_ver3'
#version 2
con_pos_dep2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_pos_ver4'
con_neg_dep2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_neg_ver4'

con_pos_dep3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_pos_ver5'
con_neg_dep3= '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_neg_ver5'
#version 2
con_pos_dep4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_pos_ver6'
con_neg_dep4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Con_dictionary/dep_neg_ver6'


pos_anxia1 = get_list_key(con_pos_anxia1)
neg_anxia1 = get_list_key(con_neg_anxia1)


pos_anxia2 = get_list_key(con_pos_anxia2)
neg_anxia2 = get_list_key(con_neg_anxia2)

arg1 = [1000,1000, 1000,1500,1500,2000,100,200,100,500,500,300] #score1 
arg2 = [1500,1200,800,1000,1200,2000,100,100,200,200,2000,700] #score2
'''
for i in range(len(arg1)):
    name_path1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_con/Difference/Lowercase/pos_anxia_' +str(arg1[i]) +'_'+ str(arg2[i]) + '.txt'
    name_path2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_con/Difference/Lowercase/neg_anxia_' + str(arg1[i]) +'_'+ str(arg2[i]) + '.txt'

    get_dictionary(name_path1, name_path2, pos_anxia2, neg_anxia2, arg1[i], arg2[i], no_distintic= False, con = True)

    name_path3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_con/Difference/Uppercase/pos_anxia_' +str(arg1[i]) +'_'+ str(arg2[i]) + '.txt'
    name_path4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_con/Difference/Uppercase/neg_anxia_' +str(arg1[i]) +'_'+ str(arg2[i]) + '.txt'

    get_dictionary(name_path3, name_path4, pos_anxia1, neg_anxia1, arg1[i], arg2[i], no_distintic= False, con = True)



    name_path5 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_con/No_difference/Uppercase/pos_anxia_' +str(arg1[i]) +'_'+ str(arg2[i]) + '.txt'
    name_path6 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_con/No_difference/Uppercase/neg_anxia_' +str(arg1[i]) +'_'+ str(arg2[i]) + '.txt'

    get_dictionary(name_path5, name_path6, pos_anxia1, neg_anxia1, arg1[i], arg2[i], no_distintic= True, con = True)

    name_path7 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_con/No_difference/Lowercase/pos_anxia_' +str(arg1[i]) +'_'+ str(arg2[i]) + '.txt'
    name_path8 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_con/No_difference/Lowercase/neg_anxia_' + str(arg1[i]) +'_'+ str(arg2[i]) + '.txt'

    get_dictionary(name_path7, name_path8, pos_anxia2, neg_anxia2, arg1[i], arg2[i], no_distintic= True, con = True)
    print(i)
#DICTIONARIES DEPRESSION 
'''
pos_dep1 = get_list_key(con_pos_dep1)
neg_dep1 = get_list_key(con_neg_dep1)

pos_dep2 = get_list_key(con_pos_dep2)
neg_dep2 = get_list_key(con_neg_dep2)


for i in range(len(arg1)):
    name_path1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_con/No_difference/Lowercase/pos_dep_' +str(arg1[i]) +'_'+ str(arg2[i]) + '.txt'
    name_path2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_con/No_difference/Lowercase/neg_dep_' +str(arg1[i]) +'_'+ str(arg2[i])+ '.txt'

    get_dictionary(name_path1, name_path2, pos_dep2, neg_dep2, arg1[i], arg2[i], no_distintic= True, con = True)
    print(i)
    name_path3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_con/No_difference/Uppercase/pos_dep_' +str(arg1[i]) +'_'+ str(arg2[i]) + '.txt'
    name_path4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_con/No_difference/Uppercase/neg_dep_' +str(arg1[i]) +'_'+ str(arg2[i]) + '.txt'

    get_dictionary(name_path3, name_path4, pos_dep1, neg_dep1, arg1[i], arg2[i], no_distintic= True, con = True)
    print(i)
    name_path5 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_con/Difference/Lowercase/pos_dep_' +str(arg1[i]) +'_'+ str(arg2[i]) + '.txt'
    name_path6 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_con/Difference/Lowercase/neg_dep_' + str(arg1[i]) +'_'+ str(arg2[i]) + '.txt'
    if (arg1[i] != 2000 or arg2[i] != 2000):
        get_dictionary(name_path5, name_path6, pos_dep2, neg_dep2, arg1[i], arg2[i], no_distintic= False, con = True)
        print(i)
        name_path7 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_con/Difference/Uppercase/pos_dep_' +str(arg1[i]) +'_'+ str(arg2[i]) + '.txt'
        name_path8 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_con/Difference/Uppercase/neg_dep_' +str(arg1[i]) +'_'+ str(arg2[i]) + '.txt'

        get_dictionary(name_path7, name_path8, pos_dep1, neg_dep1, arg1[i], arg2[i], no_distintic= False, con = True)
        print(i)



