#encoding:UTF-8

from unittest import *
import unittest
import csv
import jieba

#----------------------------------------------------------------------
def jiebaCut():
    """"""
    f = open(u'银行-143.csv', 'rb')
    lists = csv.reader(f)
    items = []
    keys = []
    for i in lists:
        items.append(i[0])
        keys.append(i[1])
    string_list = jieba.cut(items[1])
    #识别汉字
    cabs = [i for i in string_list if i[0] >= u'\u4e00' and i[0] <= u'\u9fa5' or\
            i.isdigit()]
    print u'关键词个数：', len(cabs)
    print u'关键词种类数：', len(set(cabs))
    print ' '.join(cabs)

#----------------------------------------------------------------------
def gender_features2(name):
    """"""
    features = {}
    features["first_letter"] = name[0].lower()
    features["last_letter"] = name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count({})".format(letter)] = name.lower().count(letter)
        features["has({})".format(letter)] = (letter in name.lower())
    return features
    

if __name__ == '__main__':
    jiebaCut()
    #a = gender_features2('John')
    #print a