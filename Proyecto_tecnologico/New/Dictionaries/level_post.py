from dic import get_dictionary, get_list_key
#POST LEVEL DICTIONARIES 
#---------------------------ANOREXIA-------------------#
#version 1
post_pos_anxia1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Post_dictionary/post_level/anxia_pos_ver1key30'
post_neg_anxia1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Post_dictionary/post_level/anxia_neg_ver1key30'
#version 2
post_pos_anxia2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Post_dictionary/post_level/anxia_pos_ver2key30'
post_neg_anxia2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Post_dictionary/post_level/anxia_neg_ver2key30'

#-------------------------DEPRESSION--------------------#
#version 1 
post_pos_dep1  =  '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Post_dictionary/post_level/dep_pos_ver1key30'
post_neg_dep1  =  '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Post_dictionary/post_level/dep_neg_ver1key30'
#version 2
post_pos_dep2  =  '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Post_dictionary/post_level/dep_pos_ver2key30'
post_neg_dep2  =  '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Post_dictionary/post_level/dep_neg_ver1key30'


#MAKE DICTIONARIES FOR ANOREXIA#

pos_anxia1 = get_list_key(post_pos_anxia1)
neg_anxia1 = get_list_key(post_neg_anxia1)


pos_anxia2 = get_list_key(post_pos_anxia2)
neg_anxia2 = get_list_key(post_neg_anxia2)

arg1 = [1000,1000, 1000,1500,1500,2000,100,200,100,500,500,300,1000,1000, 1000,1500,1500,2000,100,200,100,500,500,300 ] #score1 
arg2 = [1500,1200,800,1000,1200,2000,100,100,200,200,2000,700,1500,1200,800,1000,1200,2000,100,100,200,200,2000,700] #score2

for i in range(len(arg1)):
    name_path1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_post/Difference/Lowercase/pos_anxia_' + str(arg1[i]) +'_'+ str(arg2[i])+ '.txt'
    name_path2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_post/Difference/Lowercase/neg_anxia_' + str(arg1[i]) +'_'+ str(arg2[i]) + '.txt'

    get_dictionary(name_path1, name_path2, pos_anxia2, neg_anxia2, arg1[i], arg2[i], no_distintic= False, con = False)

    name_path3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_post/Difference/Uppercase/pos_anxia_' +str(arg1[i]) +'_'+ str(arg2[i])+ '.txt'
    name_path4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_post/Difference/Uppercase/neg_anxia_' + str(arg1[i]) +'_'+ str(arg2[i]) + '.txt'

    get_dictionary(name_path3, name_path4, pos_anxia1, neg_anxia1, arg1[i], arg2[i], no_distintic= False, con = False)



    name_path5 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_post/No_difference/Uppercase/pos_anxia_' +str(arg1[i]) +'_'+ str(arg2[i]) + '.txt'
    name_path6 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_post/No_difference/Uppercase/neg_anxia_' +str(arg1[i]) +'_'+ str(arg2[i])+ '.txt'

    get_dictionary(name_path5, name_path6, pos_anxia1, neg_anxia1, arg1[i], arg2[i], no_distintic= True, con = False)

    name_path7 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_post/No_difference/Lowercase/pos_anxia_' +str(arg1[i]) +'_'+ str(arg2[i]) + '.txt'
    name_path8 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_post/No_difference/Lowercase/neg_anxia_' +str(arg1[i]) +'_'+ str(arg2[i]) + '.txt'

    get_dictionary(name_path7, name_path8, pos_anxia2, neg_anxia2, arg1[i], arg2[i], no_distintic= True, con = False)

#DICTIONARIES DEPRESSION 
pos_dep1 = get_list_key(post_pos_dep1)
neg_dep1 = get_list_key(post_neg_dep1)

pos_dep2 = get_list_key(post_pos_dep2)
neg_dep2 = get_list_key(post_neg_dep2)


for i in range(len(arg1)):
    name_path1 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_post/No_difference/Lowercase/pos_dep_' +str(arg1[i]) +'_'+ str(arg2[i]) + '.txt'
    name_path2 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_post/No_difference/Lowercase/neg_dep_' +str(arg1[i]) +'_'+ str(arg2[i])+ '.txt'

    get_dictionary(name_path1, name_path2, pos_dep2, neg_dep2, arg1[i], arg2[i], no_distintic= True, con = False)

    name_path3 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_post/No_difference/Uppercase/pos_dep_' +str(arg1[i]) +'_'+ str(arg2[i]) + '.txt'
    name_path4 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_post/No_difference/Uppercase/neg_dep_' +str(arg1[i]) +'_'+ str(arg2[i]) + '.txt'

    get_dictionary(name_path3, name_path4, pos_dep1, neg_dep1, arg1[i], arg2[i], no_distintic= True, con = False)

    name_path5 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_post/Difference/Lowercase/pos_dep_' + str(arg1[i]) +'_'+ str(arg2[i])+ '.txt'
    name_path6 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_post/Difference/Lowercase/neg_dep_' + str(arg1[i]) +'_'+ str(arg2[i])+'.txt'

    get_dictionary(name_path5, name_path6, pos_dep2, neg_dep2, arg1[i], arg2[i], no_distintic= False, con = False)

    name_path7 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_post/Difference/Uppercase/pos_dep_' +str(arg1[i]) +'_'+ str(arg2[i]) + '.txt'
    name_path8 = '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/New/Dictionaries/Level_post/Difference/Uppercase/neg_dep_' +str(arg1[i]) +'_'+ str(arg2[i]) + '.txt'

    get_dictionary(name_path7, name_path8, pos_dep1, neg_dep1, arg1[i], arg2[i], no_distintic= False, con = False)






