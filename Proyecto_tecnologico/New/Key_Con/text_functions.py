import xml.etree.ElementTree as ET
from os import listdir
from os.path import isfile, join
# #####--- FUNCTIONS TO READ TRAINING DATA ----########## #


## Functions to read  post from the users ##
def get_urls(path): 
    # Function that return the name of the subjects in the folder 

    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    onlyfiles.sort()

    return onlyfiles

def get_text_post_train(user_path):
    #Function that gets the post from a user 

    #The tree of the user 
    tree = ET.parse(user_path)
    # Obtain the root of the three
    root = tree.getroot()
    # List with the historial of the user post 
    hist_post = [] # list with entries from the user 

    #iterate recursively over all the sub-tree 
    #source : https://docs.python.org/3/library/xml.etree.elementtree.html

    #Iterate first by the root childs named 'WRITTING'
    for post in root.iter('WRITING'): 
        #The iterate by the child with name 'TITLE', this correspond to the title of each one of the posts 
        for t in post.iter('TITLE'):
            entry = t.text
            # Check that the entry is not empty 
            if entry != ' ' and entry != '  ' and entry != '' and entry!= '\n' and entry!= '   ': 
                #This is only to only obtain the text without blank space at the begining and final of the entry 
                if entry[-1] == ' ':
                    hist_post.append(entry[1:-1])
                elif entry[-1] == ' ' and entry[-2] == '  ': 
                    hist_post.append(entry[1:-2])
                elif entry[-1] == ' ' and entry[-2] == '  ' and entry[-3] == '   ': 
                    hist_post.append(entry[1:-3])
                else:
                    hist_post.append(entry[1:])
        #Iterate by the child with name 'TEXT' this entry correspond to the post's text 
        for t in post.iter('TEXT'):
            entry = t.text
            
            if entry != ' ' and entry != '  ' and entry != '' and entry != '   ':
                if entry[-1] == ' ' and entry[-2] != ' ' :
                    hist_post.append(entry[1:-1])
                elif entry[-1] == ' ' and entry[-2] == ' ' and entry[-3] != ' ': 
                    hist_post.append(entry[1:-2])
                elif entry[-1] == ' ' and entry[-2] == ' ' and entry[-3] == ' ': 
                    hist_post.append(entry[1:-3])
                else:
                    hist_post.append(entry[1:])
    all_post = ""  # concatenate all post
    for i in range(len(hist_post)):
        all_post = all_post + hist_post[i] + " "

    return all_post


def get_text_labels(path, polarity='Negative'):

    all_documents = [] # list with all the documents
    all_label = [] # label

    user_path = get_urls(path)
    for i in range(len(user_path)):
        subject = user_path[i]  #for example test_subjet1005.xml
        path_subject = path + '/' + subject
        document = get_text_post_train(path_subject) #get document with all the history of a user

        all_documents += [document]
        if polarity == 'Negative':
            all_label += [0]
        else:
            all_label += [1]

    return all_documents, all_label



def get_test_post(user_path):
    tree = ET.parse(user_path)
    root = tree.getroot()
    hist_post = []  # list with entries from the user

    # iterate recursively over all the sub-tree
    # source : https://docs.python.org/3/library/xml.etree.elementtree.html
    num_post = len(root)

    for i in range(1, num_post):
        entry = root[i][0].text
        if type(entry) == str: 
            if entry != ' ' and entry != '  ' and entry != '' and entry!= '\n' and entry!= '   ': 
                #This is only to only obtain the text without blank space at the begining and final of the entry 
                if entry[-1] == ' ':
                    hist_post.append(entry[1:-1])
                elif entry[-1] == ' ' and entry[-2] == '  ': 
                    hist_post.append(entry[1:-2])
                elif entry[-1] == ' ' and entry[-2] == '  ' and entry[-3] == '   ': 
                    hist_post.append(entry[1:-3])
                else:
                    hist_post.append(entry[1:])
                
        if len(root[i][2]) == 0:
            entry = root[i][2].text
            if type(entry) == str:
                if entry != ' ' and entry != '  ' and entry != '' and entry!= '\n' and entry!= '   ': 
                    #This is only to only obtain the text without blank space at the begining and final of the entry 
                    if entry[-1] == ' ':
                        hist_post.append(entry[1:-1])
                    elif entry[-1] == ' ' and entry[-2] == '  ': 
                        hist_post.append(entry[1:-2])
                    elif entry[-1] == ' ' and entry[-2] == '  ' and entry[-3] == '   ': 
                        hist_post.append(entry[1:-3])
                    else:
                        hist_post.append(entry[1:])

        else:
            num_sub_post = len(root[i][2])
            for j in range(1, num_sub_post - 1):
                if user_path == '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/depression2022/test_data/datos/subject1566.xml' and j == 1:
                    entry = root[i][2][1].text
                    if type(entry) == str:
                        if entry != ' ' and entry != '  ' and entry != '' and entry!= '\n' and entry!= '   ': 
                        #This is only to only obtain the text without blank space at the begining and final of the entry 
                            if entry[-1] == ' ':
                                hist_post.append(entry[1:-1])
                            elif entry[-1] == ' ' and entry[-2] == '  ': 
                                hist_post.append(entry[1:-2])
                            elif entry[-1] == ' ' and entry[-2] == '  ' and entry[-3] == '   ': 
                                hist_post.append(entry[1:-3])
                            else:
                                hist_post.append(entry[1:])
                
                elif user_path == '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/depression2022/test_data/datos/subject1566.xml' and j > 3 and len(
                    root[i][2][j][2]) != 0:
                        for k in range(1, len(root[i][2][j][2])):
                            entry = root[i][2][j][2][k][2].text
                            if type(entry) == str:
                                if entry != ' ' and entry != '  ' and entry != '' and entry!= '\n' and entry!= '   ': 
                                #This is only to only obtain the text without blank space at the begining and final of the entry 
                                    if entry[-1] == ' ':
                                        hist_post.append(entry[1:-1])
                                    elif entry[-1] == ' ' and entry[-2] == '  ': 
                                        hist_post.append(entry[1:-2])
                                    elif entry[-1] == ' ' and entry[-2] == '  ' and entry[-3] == '   ': 
                                        hist_post.append(entry[1:-3])
                                    else:
                                        hist_post.append(entry[1:])

                elif user_path != '/home/est_posgrado_maria.garcia/Tesis/Proyecto_tecnologico/depression2022/test_data/datos/subject1566.xml':
                    entry = root[i][2][j][2].text

                    if type(entry) == str:
                        if entry != ' ' and entry != '  ' and entry != '' and entry!= '\n' and entry!= '   ': 
                        #This is only to only obtain the text without blank space at the begining and final of the entry 
                            if entry[-1] == ' ':
                                hist_post.append(entry[1:-1])
                            elif entry[-1] == ' ' and entry[-2] == '  ': 
                                hist_post.append(entry[1:-2])
                            elif entry[-1] == ' ' and entry[-2] == '  ' and entry[-3] == '   ': 
                                hist_post.append(entry[1:-3])
                            else:
                                hist_post.append(entry[1:])
    ##--------concatenate post -------------#
    all_post = ""  # concatenate all post
    for i in range(len(hist_post)):
        all_post = all_post + hist_post[i] + " "

    return all_post

def get_text_test(path, list_subject):
    all_documents = []  # list with all the documents

    for i in range(len(list_subject)):
        subject = list_subject[i]  # for example test_subjet1005.xml
        path_subject = path + '/' + subject + '.xml'

        document = get_test_post(path_subject)  # get document with all the history of a user

        all_documents += [document]

    return all_documents


def get_text_chunk(path): 
    all_documents = [] #list with all the documents 

    user_path = get_urls(path)
    for i in range(len(user_path)): 
        subject = user_path[i] #for example test_subjet1005.xml 
        path_subject = path + '/' + subject 
        document = get_text_post_train(path_subject) #get document with all the history of a user 

        all_documents += [document] 
    return all_documents
    

def get_text_test_anorexia(path, list_subject, chunk): 
    all_documents = [] #list with all the documents 

    for i in range(len(list_subject)): 
        subject = list_subject[i] #for example test_subjet1005.xml 
        path_subject = path + '/chunk'+ str(chunk) + '/' + subject + '_' + str(chunk) + '.xml'
        
        document = get_text_post_train(path_subject) #get document with all the history of a user 
        
        all_documents += [document] 
    #print("text extracted for all users")
    return all_documents
    

