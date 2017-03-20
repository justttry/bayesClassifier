#encoding:UTF-8

import numpy as np
import unittest
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import mutual_info_classif 
import jieba
import csv
import os
import datetime
import sys
import shelve
import nltk
import random
from decisionTree import calcInformationGain, calcShannonEnt

csv.field_size_limit(sys.maxint)

#----------------------------------------------------------------------
def get_test_files(cwd):
    """
    查找测试文件
    Parameter:
    cwd:目标文件夹
    Return:
    files:目标文件夹里的各个子文件
    """
    rets = []
    for root, dirs, files in os.walk(cwd):
        if not dirs:
            for fn in files:
                rets.append((root+'\\'+fn, root[-4:]))
    return rets
            
#----------------------------------------------------------------------
def load_file(files, filetype='csv'):
    """加载csv文件"""
    #输入阵列
    X = np.zeros((0, 1))
    #标记阵列
    y = np.zeros((0, 1))
    #特征值向量
    keys = []
    cv = CountVectorizer(analyzer='word')
    v = DictVectorizer(sparse=False)
    for i in files:
        print i[0]
        f = open(i[0])
        if filetype == 'csv':
            items = csv.reader(f)
        else:
            items = []
            for item in f.readlines():
                items.append(item)
            items = [''.join([item.replace('\n', '') for item in items])]
        for j in items:
            #仅对标题进行处理
            if filetype == 'csv':
                string_list = jieba.cut(j[1])
            else:
                j = j.replace('\n', '')
                string_list = jieba.cut(j)
            #识别汉字
            string_list = [m for m in string_list if m[0] >= u'\u4e00' and m[0] <= u'\u9fa5' or\
                           m.isdigit()]
            if string_list:
                #对单个条目进行计数
                try:
                    cv_fit = cv.fit_transform(string_list)
                except:
                    print 'error detect!'
                    print string_list[0]
                    continue
                #生成字典{关键词:词频}
                X = np.concatenate((X, np.array([[dict(zip(cv.get_feature_names(), cv_fit.toarray().sum(axis=0)))]])), axis=0)
                y = np.concatenate((y, np.array([[i[1]]])))
        f.close()
    #根据生成的字典生成矩阵
    X = v.fit_transform(X.T[0])
    return X, y, v.get_feature_names()

#----------------------------------------------------------------------
def select_features_chi(dirs, nums=1000, filetype='csv'):
    """
    使用卡方检验选取特征值
    """
    cwd = os.getcwd()
    cwd = cwd + '\\' + dirs
    files = get_test_files(cwd)
    X, y, labels = load_file(files, filetype)
    #计算卡方
    chi2s = chi2(X, y)[0]
    dicts = dict(zip(range(np.shape(chi2s)[0]), chi2s))
    sortedlist = sorted(dicts.iteritems(), 
                        key=lambda i: i[1], 
                        reverse=True)
    columns = [i for (i, _) in sortedlist]
    return X[:, columns[:nums]], y, \
           [labels[int(i)] for i in np.array(sortedlist[:nums])[:, 0]]

#----------------------------------------------------------------------
def select_features_ig(dirs, nums=1000, filetype='csv'):
    """
    使用信息增益选取特征值
    """
    cwd = os.getcwd()
    cwd = cwd + '\\' + dirs
    files = get_test_files(cwd)
    X, y, labels = load_file(files, filetype)
    m, n = np.shape(X)
    #计算信息熵增益
    newX = X.copy()
    newX[newX!=0] = 1
    newX[newX==0] = 0
    newy = y.copy()
    ysets = set(newy.reshape((m, )))
    yrep = 0
    for i in ysets:
        newy[newy==i] = yrep
        yrep += 1
    dataSet = np.concatenate((newX, newy), axis=1)
    igs = [calcInformationGain(dataSet, i) for i in range(n)]
    dicts = dict(zip(range(n), igs))
    sortedlist = sorted(dicts.iteritems(), 
                        key=lambda i: i[1], 
                        reverse=True)
    columns = [i for (i, _) in sortedlist]
    return X[:, columns[:nums]], y, \
           [labels[int(i)] for i in np.array(sortedlist[:nums])[:, 0]]

#----------------------------------------------------------------------
def select_features_mul(dirs, nums=1000, filetype='csv'):
    """
    使用信息增益选取特征值
    """
    cwd = os.getcwd()
    cwd = cwd + '\\' + dirs
    files = get_test_files(cwd)
    X, y, labels = load_file(files, filetype)
    m, n = np.shape(X)
    #计算信息熵增益
    newX = X.copy()
    newX[newX!=0] = 1
    newX[newX==0] = 0
    igs = mutual_info_classif(newX, y.reshape((m, )))
    dicts = dict(zip(range(n), igs))
    sortedlist = sorted(dicts.iteritems(), 
                        key=lambda i: i[1], 
                        reverse=True)
    columns = [i for (i, _) in sortedlist]
    return X[:, columns[:nums]], y, \
           [labels[int(i)] for i in np.array(sortedlist[:nums])[:, 0]]
    

#----------------------------------------------------------------------
def calcTfidf(X, delta=0.0001):
    """
    Parameter:
    X:关键词词频矩阵
    delta:缺省0.0001
    Return:
    idf: inverse document frequency
    tfidf
    """
    m, n = np.shape(X)
    newX = X.copy()
    newX[newX[:, :]!=0] = 1.0
    df = newX.sum(axis=0)
    # idf(d, t) = log [ (1 + n) / (1 + df(d, t)) ] + 1.
    idf = np.log((n + 1.0) / (df + 1)) + 1
    tfidf = idf * X
    #tfidf = tfidf / (tfidf.sum(axis=1).reshape((m, 1)) + delta)
    return idf, tfidf

#----------------------------------------------------------------------
def create_features(X, y):
    """
    Parameter:
    X:tfidf矩阵
    y:类别
    Return:
    featuresets:特征集
    """
    m, n = np.shape(X)
    return [(dict(zip(range(n), X[i, :])), y[i, 0]) for i in range(m)]

#----------------------------------------------------------------------
def createClassify(trainset):
    """
    生成分类器
    Parameter:
    trainset:训练集
    Return:
    classifier:分类器
    """
    return nltk.NaiveBayesClassifier.train(trainset)

#----------------------------------------------------------------------
def clstest(classifier, test_set):
    """
    测试分类器
    classifier:分类器
    test_set:测试集
    """
    result = nltk.classify.accuracy(classifier, test_set)
    print u'精确度: ', result
    
    
    
########################################################################
class BayesTextClsTest(unittest.TestCase):
    """"""
        
    #----------------------------------------------------------------------
    def test_load_files(self):
        """"""
        #import pdb
        #pdb.set_trace()
        start = datetime.datetime.now()
        #X, y, labels = select_features_chi('test_file2', filetype='txt')
        X, y, labels = select_features_mul('test_file2', filetype='txt')
        m = np.shape(X)[0]
        print u'总样本数：', m
        train_lines = []
        test_lines = []
        for i in range(m):
            if random.random() <= 0.7:
                train_lines.append(i)
            else:
                test_lines.append(i)
        train_X = X[train_lines, :]
        train_y = y[train_lines, :]
        test_X = X[test_lines, :]
        test_y = y[test_lines, :]
        end = datetime.datetime.now()
        print 'cost time: ', end - start
        train_idf, train_tfidf = calcTfidf(train_X)
        test_tfidf = train_idf * test_X
        start = datetime.datetime.now()
        print 'cost time: ', start - end
        train_featuresets = create_features(train_tfidf, train_y)
        test_featuresets = create_features(test_tfidf, test_y)
        end = datetime.datetime.now()
        print 'cost time: ', end - start
        classifier = createClassify(train_featuresets)
        clstest(classifier, test_featuresets)
        start = datetime.datetime.now()
        print 'cost time: ', start - end
        
            
#----------------------------------------------------------------------
def suite():
    """"""
    suite = unittest.TestSuite()
    suite.addTest(BayesTextClsTest('test_load_files'))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')