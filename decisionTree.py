#encoding:UTF-8

import unittest
from numpy import *
from treePlotter import *

########################################################################
class Node(object):
    """"""
    
    #----------------------------------------------------------------------
    def __init__(self, dataset=None, result=None, attr=None):
        """Constructor"""
        self.dataset = dataset
        self.result = result
        self.attr = attr
        self.childs = {}
        

#----------------------------------------------------------------------
def calcInst(dataSet):
    """"""
    results = dataSet[:, -1]
    retdict = {}
    maxnum = 0
    retclass = results[0]
    for i in results:
        result = retdict.get(i, 0)
        result += 1
        retdict[i] = result
        if result > maxnum:
            maxnum = result
            retclass = i
    return retclass


#----------------------------------------------------------------------
def calcShannonEnt(dataSet):
    """"""
    nums = len(dataSet)
    labels = {}
    for i in dataSet:
        label = i[-1]
        num = labels.get(label, 0)
        labels[label] = num + 1
    shannonEnt = 0.0
    for key in labels:
        prob = labels[key] / float(nums)
        shannonEnt -= prob * log(prob)
    return shannonEnt

#----------------------------------------------------------------------
def calcInformationGain(dataSet, j):
    """"""
    print j
    symbols = set(dataSet[:, j])
    length = len(dataSet)
    infG = 0.0
    for symbol in symbols:
        subset = dataSet[dataSet[:, j]==symbol]
        prob = len(subset) / float(length)
        infG += prob * calcShannonEnt(subset)
    return calcShannonEnt(dataSet) - infG

#----------------------------------------------------------------------
def splitDataSet(dataSet, axis, value):
    """"""
    length = len(dataSet[0])
    ret = dataSet[dataSet[:, axis]==value]
    return ret[:, [i for i in range(length) if i != axis]]
    

#----------------------------------------------------------------------
def chooseBestFeatureToSplit(dataSet):
    """"""
    numFeatures = len(dataSet[0]) - 1
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        infoGain = calcInformationGain(dataSet, i)
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

#----------------------------------------------------------------------
def build_tree(D, A, threshold=0.0001, algo='ID3'):
    """"""
    if len(set(D[:, -1])) == 1:
        return Node(dataset=D, result=D[0, -1])
    elif not A.any():
        return Node(dataset=D, result=calcInst(D))
    else:
        bestFeature = chooseBestFeatureToSplit(D)
        infoGain = calcInformationGain(D, bestFeature)
        if infoGain < threshold:
            return Node(dataset=D, result=calcInst(D))
        else:
            newnode = Node(dataset=D)
            newnode.attr = bestFeature
            for ai in set(D[:, bestFeature]):
                subdataset = splitDataSet(D, bestFeature, ai)
                subtree = build_tree(subdataset, 
                                     A[[i for i in range(len(A)) if i != bestFeature]])
                newnode.childs[ai] = subtree
            return newnode
        
#----------------------------------------------------------------------
def classify0(myTree, instance):
    """"""
    if myTree.result is not None:
        return myTree.result
    else:
        return classify0(myTree.childs[instance[myTree.attr]], instance)

#----------------------------------------------------------------------
def classify(D, instance):
    """"""
    myTree = build_tree(D, array(range(len(D[0]) - 1)))
    return classify0(myTree, instance)



########################################################################
class CalcShannonEntTest(unittest.TestCase):
    """"""
    
    #----------------------------------------------------------------------
    def test_0(self):
        a = 101
        nums = []
        maxnum = 100
        dataSet = []
        shannonEnt = 0.0
        while(a>1):
            b = random.randint(1, a)
            if b not in nums:
                nums.append(b)
            a = 101 - sum(nums)
        for i in nums:
            dataSet += [[i]] * i
            shannonEnt -= i/100.0 * log(i/100.0)
        self.assertEqual(shannonEnt, calcShannonEnt(dataSet))
        
########################################################################
class CalcInformationGainTest(unittest.TestCase):
    """"""
    
    #----------------------------------------------------------------------
    def test_0(self):
        a = 101
        nums1 = []
        nums2 = []
        maxnum = 100
        dataSet = []
        informationGain = 0.0
        maxnum = 0
        while(a>1 and maxnum < 100):
            b = random.randint(1, a)
            if b not in nums1:
                nums1.append(b)
            a = 101 - sum(nums1)
            maxnum += 1
        for i in nums1:
            dataSet += [[i]] * i        
        for i in range(sum(nums1)):
            b = random.choice([-1, 1])
            nums2.append([b])
        dataSet = array(dataSet)
        nums2 = array(nums2)
        nums = append(dataSet, nums2, axis=1)
        IG = calcInformationGain(nums, 0)
        length = len(nums)
        for i in nums1:
            subset = nums[nums[:, 0] == i]
            prob = len(subset) / float(length)
            informationGain += prob * calcShannonEnt(subset)
        self.assertAlmostEqual(IG, calcShannonEnt(nums) - informationGain, delta=0.00001)
    
    #----------------------------------------------------------------------
    def test_1(self):
        """"""
        for i in range(10000):
            self.test_0()
            
########################################################################
class ChooseBestFeatureToSplitTest(unittest.TestCase):
    """"""
    
    #----------------------------------------------------------------------
    def test_0(self):
        pass
    

########################################################################
class SplitDataSetTest(unittest.TestCase):
    """"""
    
    #----------------------------------------------------------------------
    def test_0(self):
        a = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        ret = splitDataSet(a, 1, 5)
        self.assertListEqual(ret.tolist(), array([[4, 6]]).tolist())
    

########################################################################
class CalcInstTest(unittest.TestCase):
    """"""
    
    #----------------------------------------------------------------------
    def test_0(self):
        n = 101
        cnt = 0
        result = 1
        a = random.choice([-1, 1], n)
        for i in a:
            if i == 1:
                cnt += 1
        if 2 * cnt < n:
            result = -1
        b = a.reshape((n, 1))
        ret = calcInst(b)
        self.assertEqual(result, ret)
        
    #----------------------------------------------------------------------
    def test_1(self):
        """"""
        for _ in range(10000):
            self.test_0()
            

########################################################################
class ClassifyTest(unittest.TestCase):
    """"""
    
    #----------------------------------------------------------------------
    def test_0(self):
        dataset = [  
           ("青年", "NO", "NO", "一般", "NO"),
           ("青年", "NO", "NO", "好", "NO"),
           ("青年", "YES", "NO", "好", "YES"),
           ("青年", "YES", "YES", "一般", "YES"),
           ("青年", "NO", "NO", "一般", "NO"),
           ("中年", "NO", "NO", "一般", "NO"),
           ("中年", "NO", "NO", "好", "NO"),
           ("中年", "YES", "YES", "好", "YES"),
           ("中年", "NO", "YES", "非常好", "YES"),
           ("中年", "NO", "YES", "非常好", "YES"),
           ("老年", "NO", "YES", "非常好", "YES"),
           ("老年", "NO", "YES", "好", "YES"),
           ("老年", "YES", "NO", "好", "YES"),
           ("老年", "YES", "NO", "非常好", "YES"),
           ("老年", "NO", "NO", "一般", "NO")  
        ]
        
        print classify(array(dataset), ("老年", "NO", "NO", "一般"))
        print 'end'
        
    
        
        
#----------------------------------------------------------------------
def suite():
    """"""
    suite = unittest.TestSuite()
    #suite.addTest(CalcShannonEntTest('test_0'))
    #suite.addTest(CalcInformationGainTest('test_0'))
    #suite.addTest(CalcInformationGainTest('test_1'))
    #suite.addTest(SplitDataSetTest('test_0'))
    #suite.addTest(CalcInstTest('test_0'))
    #suite.addTest(CalcInstTest('test_1'))
    suite.addTest(ClassifyTest('test_0'))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')