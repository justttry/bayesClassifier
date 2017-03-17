#encoding:UTF-8

import numpy as np
import unittest
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
import jieba
import csv
import os
import datetime
import sys
import shelve
import nltk

csv.field_size_limit(sys.maxint)
            
#----------------------------------------------------------------------
def load_file():
    """加载csv文件"""
    cwd = os.getcwd()
    #获取csv文件列表
    files = [i for i in os.listdir(cwd) if i[-4:]=='.csv']
    #输入阵列
    X = np.zeros((0, 1))
    #标记阵列
    y = np.zeros((0, 1))
    #特征值向量
    cnt = 0
    keys = []
    cv = CountVectorizer(analyzer='word')
    v = DictVectorizer(sparse=False)
    for i in files:
        print i
        f = open(i)
        items = csv.reader(f)
        for j in items:
            #仅对标题进行处理
            print cnt
            cnt += 1
            string_list = [m for m in jieba.cut(j[1]) if m[0] >= u'\u4e00' and m[0] <= u'\u9fa5' or\
                           m.isdigit()]
            if string_list:
                #生成字典{关键词:词频}
                X = np.concatenate((X, [[' '.join(string_list)]]))
                y = np.concatenate((y, np.array([[i[:2]]])))
        f.close()
    #根据生成的字典生成矩阵
    X = cv.fit_transform(X)
    return X, y, cv.get_feature_names()

#----------------------------------------------------------------------
def select_features(nums=600, loadAct=True):
    """
    使用卡方检验选取特征值
    """
    if loadAct:
        X, y, labels = load_file()
        f = shelve.open('Arr.vn')
        f['X'] = X
        f['y'] = y
        f['labels'] = labels
        f.close()
    else:
        f = shelve.open('Arr.vn')
        X = f['X']
        y = f['y']
        labels = f['labels']
        f.close()
    #计算卡方
    chi2s = chi2(X, y)[0]
    dicts = dict(zip(range(np.shape(chi2s)[0]), chi2s))
    sortedlist = sorted(dicts.iteritems(), 
                        key=lambda i: i[1], 
                        reverse=True)
    return [labels[int(i)] for i in np.array(sortedlist[:nums])[:, 0]]
    
    
########################################################################
class BayesTextClsTest(unittest.TestCase):
    """"""
        
    #----------------------------------------------------------------------
    def test_load_files(self):
        """"""
        start = datetime.datetime.now()
        print select_features(loadAct=True)
        end = datetime.datetime.now()
        print 'cost time: ', end - start
            
#----------------------------------------------------------------------
def suite():
    """"""
    suite = unittest.TestSuite()
    suite.addTest(BayesTextClsTest('test_load_files'))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')