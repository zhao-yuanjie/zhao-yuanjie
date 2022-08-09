

from sklearn.model_selection import train_test_split
import numpy as np
from gensim.models.word2vec import Word2Vec
from nltk import *
# 直接词向量相加求平均
#data = pd.read_csv('D:/本科课程学习/大三下/内容安全/2022春/BBC-Dataset-News-Classification-master/BBC-Dataset-News-Classification-master/dataset/data.csv')
def use_word2vec(x):
    """
    调用 gensim 中的 word2vec模型
    """
    word2vec_model = Word2Vec(x,
                              vector_size=64,  # 词向量维度
                              epochs=10,  # 训练次数
                              min_count=10  # 忽略词频小于20的词
                              )
    return word2vec_model
def get_contentVector(news_word, word2vec_model):
    """
    获取每篇评论的相关性向量，并求平均值
    :return:
    """
    # 获取单篇文章的每一个分词在 word2vec 模型的相关性向量
    vector_list_xq = [word2vec_model.wv[k] for k in news_word if k in word2vec_model.wv]

    # 通过np.array方法转成ndarray对象再对每一列求平均
    # 如果vector_list_xq为空则返回空值，如果vector_list_xq不为空则对每一列求均值然后返回数据
    if len(vector_list_xq) != 0:
        contentVector_xq = np.array(vector_list_xq).mean(axis=0)
        return contentVector_xq
    else:
        return None
def feature_engineering(news, word2vec_model, label_list=None):
    """
    特征工程，对于每一篇文章，获取文章的每一个分词在word2vec模型的相关性向量。
    然后把一篇文章的所有分词在word2vec模型中的相关性向量求和取平均数，即此篇文章在word2vec模型中的相关性向量。
    """
    contentVector_list_xq = []  # 词向量列表

    for i in range(len(news)):
        cutWords_xq = news[i]  # 取每一篇文章
        # 调用函数获得每一篇文章的向量，并将结果传入contentVector_list_xq列表
        result_xq = get_contentVector(cutWords_xq, word2vec_model)

        # 判断get_contentVector返回值是否为空, 空值就删除对应的标签，非空则保存到相关性列表中
        if result_xq is None:
            label_list.pop(i)  
        else:
            contentVector_list_xq.append(result_xq)

    X_xq = np.array(contentVector_list_xq)
    print('X shape: ', X_xq.shape)
    if label_list is not None:
        print('label list length: ', len(label_list))
        return X_xq, label_list
    else:
        return X_xq


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
word2vec_model_xq = use_word2vec(comment)
X, Y = feature_engineering(comment, word2vec_model_xq, type)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)