import pandas as pd 
import numpy as np
from nltk import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import random
comment_fu=get_word_list('fu.csv')
comment_zheng=get_word_list('zheng.csv')
comment_zhong=get_word_list('zhong.csv')
type = []
comment = []
for i in comment_fu:
    type.append(-1)
    comment.append(i)
for i in comment_zheng:
    type.append(1)
    comment.append(i)
for i in comment_zhong:
    type.append(0)
    comment.append(i)
#使用tfidf进行特征选择
#实例化tf实例
vect = TfidfVectorizer(min_df=2)
#X和Y为训练集，当然也可以用train_test_split随机分配训练集合测试集
X=vect.fit_transform(comment)
Y=np.array(type)
#将词频矩阵X统计成TF-IDF值  

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)


