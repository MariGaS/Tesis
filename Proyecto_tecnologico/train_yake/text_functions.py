import xml.etree.ElementTree as ET
import os
import re
from os import listdir
from os.path import isfile, join
# #####--- FUNCTIONS TO READ TRAINING DATA ----########## #


def get_urls(path):

    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    onlyfiles.sort()

    return onlyfiles


def get_text_post_train(user_path):
    tree = ET.parse(user_path)
    root = tree.getroot()
    hist_post = []  # list with entries from the user

    # iterate recursively over all the sub-tree
    # source : https://docs.python.org/3/library/xml.etree.elementtree.html
    for post in root.iter('WRITING'): 

        for t in post.iter('TITLE'):
            entry = t.text
            
            if entry != ' ' and entry != '  ' and entry != '' and entry!= '\n' and entry!= '   ': 
                hist_post.append(entry[1:])
        for t in post.iter('TEXT'):
            entry = t.text
            
            if entry != ' ' and entry != '  ' and entry != '' and entry != '   ':
                hist_post.append(entry[1:])

    # #--------concatenate post -------------#
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
        title = root[i][0].text
        if type(title) == str: 
            if title == ' ' or title == '  ':
                continue 
            else: 
                hist_post.append(title[0:-1]) 
                
        if len(root[i][2]) == 0:
            entry = root[i][2].text

            if type(entry) == str:
                entry = entry.replace('\n', ' ')
                if entry == ' ' or entry == '  ':
                    continue
                else:
                    hist_post.append(entry[0:-1])

        else:
            num_sub_post = len(root[i][2])
            for j in range(1, num_sub_post - 1):
                # print(i,j, user_path)

                if user_path == 'depression2022/test_data/datos/subject1566.xml' and j == 1:
                    entry = root[i][2][1].text
                    if type(entry) == str:
                        entry = entry.replace('\n', ' ')
                        if entry == ' ' or entry == '  ':
                            continue
                        else:
                            hist_post.append(entry[0:-1])
                elif user_path == 'depression2022/test_data/datos/subject1566.xml' and j > 3 and len(
                        root[i][2][j][2]) != 0:
                    for k in range(1, len(root[i][2][j][2])):
                        entry = root[i][2][j][2][k][2].text
                        if type(entry) == str:
                            entry = entry.replace('\n', ' ')
                            if entry == ' ' or entry == '  ':
                                continue
                            else:
                                hist_post.append(entry[0:-1])

                elif user_path != 'depression2022/test_data/datos/subject1566.xml':
                    entry = root[i][2][j][2].text

                    if type(entry) == str:
                        entry = entry.replace('\n', ' ')
                        if entry == ' ' or entry == '  ':
                            continue
                        else:
                            hist_post.append(entry[0:-1])
                            ##--------concatenate post -------------#
    all_post = ""  # concatenate all post
    for i in range(len(hist_post)):
        all_post = all_post + hist_post[i] + " "

    return all_post


def get_text_test(path, list_subject):
    all_documents = []  # list with all the documents
    all_label = []  # label

    for i in range(len(list_subject)):
        subject = list_subject[i]  # for example test_subjet1005.xml
        path_subject = path + '/' + subject + '.xml'

        document = get_test_post(path_subject)  # get document with all the history of a user

        all_documents += [document]

    return all_documents


def get_text_chunk(path): 
    all_documents = [] #list with all the documents 
    all_label = [] #label 

    user_path = get_urls(path)
    for i in range(len(user_path)): 
        subject = user_path[i] #for example test_subjet1005.xml 
        path_subject = path + '/' + subject 
        document = get_text_post_train(path_subject) #get document with all the history of a user 

        all_documents += [document] 
    return all_documents
    

def get_text_test_anorexia(path, list_subject, chunk): 
    all_documents = [] #list with all the documents 
    all_label = [] #label 

    for i in range(len(list_subject)): 
        subject = list_subject[i] #for example test_subjet1005.xml 
        path_subject = path + '/chunk'+ str(chunk) + '/' + subject + '_' + str(chunk) + '.xml'
        
        document = get_text_post_train(path_subject) #get document with all the history of a user 
        
        all_documents += [document] 
    print("text extracted for all users")
    return all_documents
    

