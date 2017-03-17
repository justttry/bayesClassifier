#encoding:UTF-8

import matplotlib.pyplot as plt
#from AVLTree import AVLTree

decisionNode = dict(boxstyle='sawtooth', fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')

#----------------------------------------------------------------------
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    """"""
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va='center', ha='center', bbox=nodeType,
                            arrowprops=arrow_args)

#----------------------------------------------------------------------
def getNumLeafs(myTree):
    """"""
    return myTree.get_numleafs()

#----------------------------------------------------------------------
def getTreeDepth(myTree):
    """"""
    return myTree.get_treedepth()

#----------------------------------------------------------------------
def plotMidText(cntrPt, parentPt, txtString):
    """"""
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)
    
#----------------------------------------------------------------------
def plotTree(myTree, parentPt, xOff, init=None):
    """"""
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    value = myTree.node.val
    if init:
        cntrPt = parentPt
    else:
        cntrPt = (parentPt[0] + xOff, parentPt[1] + plotTree.yOff)
    plotNode(value, cntrPt, parentPt, decisionNode)
    if myTree.node.left.height >= 1:
        newxOff = -abs(xOff) / 2.0
        plotTree(myTree.node.left, 
                 cntrPt, 
                 newxOff)
    if myTree.node.right.height >= 1:
        newxOff = abs(xOff) / 2.0
        plotTree(myTree.node.right, 
                 cntrPt, 
                 newxOff)
        
        
#----------------------------------------------------------------------
def createPlot(inTree):
    """"""
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.yOff = -1.0 / plotTree.totalD
    xOff = 0.5
    plotTree(inTree, (0.5, 1.0), xOff, True)
    plt.show()
    

if __name__ == '__main__':
    myTree = AVLTree()
    myTree.insert(1)
    myTree.insert(2)
    myTree.insert(3)
    createPlot(myTree)