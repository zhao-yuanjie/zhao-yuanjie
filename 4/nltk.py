#coding=gbk

import pandas as pd
import jieba
def read_file(file_name):
    data = pd.read_csv(file_name,encoding='gb18030')
    #print(data)
    
    num = data['num'].tolist()#['']
    comment = data['comment'].tolist()
    return num,comment

def split(comment_list):
    ci_list = []
    for i in comment_list:
        
        temp = jieba.lcut(i, cut_all=True)
        
        #print(temp.word)
        ci_list.append(temp)
    return ci_list
def del_stopword(ci_list):
    ci=[]
    stopwords = [line.rstrip() for line in open(r'stop_word.txt', encoding='utf-8')]
    #print(stopwords)
    for i in ci_list:
        temp = []
        for j in i :
            if j not in stopwords and j!=' ':
                temp.append(j)
        ci.append(temp)

    return ci
def get_word_list(file_name):
    x,y=read_file(file_name)
    ci_list=split(y)

    stop_ci =del_stopword(ci_list)
    word = []
    for i in stop_ci:
        data=''
        for j in i:
            data=data+j+' '
            word.append(data)     
    return word

