from math import log
import operator
import pickle

import numpy as np

from treePlotter import getLeafBestCls
import matplotlib.pyplot as plt

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

# 计算信息熵
def calcShannonEnt(dataSet, method = 'none'):
    numEntries = len(dataSet)
    labelCount = {}
    for feature in dataSet:
        if method =='prob':
            label =  feature
        else:
            label = feature[-1]
        if label not in labelCount.keys():
            labelCount[label]=1
        else:
            labelCount[label]+=1
    shannonEnt = 0.0
    for key in labelCount:
        numLabels = labelCount[key]
        prob = numLabels/numEntries
        shannonEnt -= prob*(log(prob,2))
    return shannonEnt




def calcGini(dataset):
    feature = [example[-1] for example in dataset]
    uniqueFeat = set(feature)
    sumProb =0.0
    for feat in uniqueFeat:
        prob = feature.count(feat)/len(uniqueFeat)
        sumProb += prob*prob
    sumProb = 1-sumProb
    return sumProb

def calcGini(dataset):
    feature = [example[-1] for example in dataset]
    uniqueFeat = set(feature)
    sumProb =0.0
    for feat in uniqueFeat:
        prob = feature.count(feat)/len(uniqueFeat)
        sumProb += prob*prob
    sumProb = 1-sumProb
    return sumProb
def chooseBestFeatureToSplit2(dataSet): #使用基尼系数进行划分数据集
    numFeatures = len(dataSet[0]) -1 #最后一个位置的特征不算
    bestInfoGain = 0.0
    bestFeature = np.Inf
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            # 通过不同的特征值划分数据子集
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob *calcGini(subDataSet)
        infoGain = newEntropy
        if(infoGain < bestInfoGain): # 选择最小的基尼系数作为划分依据
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature #返回决策属性的最佳索引



def chooseBestFeatureToSplit3(dataSet): #使用信息增益率进行划分数据集
    numFeatures = len(dataSet[0]) -1 #最后一个位置的特征不算
    baseEntropy = calcShannonEnt(dataSet) #计算数据集的总信息熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        newEntropyProb = calcShannonEnt(featList, method='prob')
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            # 通过不同的特征值划分数据子集
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob *calcGini(subDataSet)
        newEntropy  = newEntropy/newEntropyProb
        infoGain = baseEntropy - newEntropy #计算每个信息值的信息增益
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature #返回信息增益的最佳索引



def splitDataSet(dataSet, axis, value): #划分数据集
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = np.delete(featVec,axis)
            # 去掉对应位置的特征
            retDataSet.append(reducedFeatVec)
    return np.array(retDataSet)
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) -1 #最后一个位置的特征不算
    baseEntropy = calcShannonEnt(dataSet) #计算数据集的总信息熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            # 通过不同的特征值划分数据子集
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob *calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy #计算每个信息值的信息增益
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature #返回信息增益的最佳索引


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
            classCount[vote] +=1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0] #返回类别最多的类




def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 获取数据集的对应标签序列
    if classList.count(classList[0]) == len(classList):
        return classList[0] # 当所有特征的类别都相同时停止划分
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit2(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat]) # 删除对应位置的标签
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues) #获取该列特征值的所有不同的特征值
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def classify(inputTrees, featLabels, testVec):
    firstStr = list(inputTrees.keys())[0]
    secondDict = inputTrees[firstStr]
    featIndex = featLabels.index(firstStr)
    classLabel = -1
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key],featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return getLeafBestCls(inputTrees) if classLabel == -1 else classLabel



def storeTree(inputTree,filename):
    with open(filename,'wb') as f:
        pickle.dump(inputTree,f)
def reloadTree(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)




