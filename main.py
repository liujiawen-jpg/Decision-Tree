# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
from tree import *
from treePlotter import *
import pandas as pd
import numpy as np
import sys
sys.setrecursionlimit(100000) #例如这里设置为十万
def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。

def load_IRIS():
    dataSet = pd.read_csv('iris/iris.csv')
    labels = dataSet.columns[1:-1]
    featList = [ dataSet[label] for label in labels ]
    train_Data = [[np.round(feat[i]) for feat in featList]for i in range(len(dataSet))]
    test_labels = dataSet['Species']
    for i in range(len(dataSet)):
        train_Data[i].append(test_labels[i])
    return train_Data,list(labels)
def load_Cora():
    dataSet=pd.read_csv('cora/cora.content',sep = '\t',header=None)
    feature = dataSet.iloc[:,1:]
    feature = np.array(feature)
    dataList = feature.tolist()
    label = [i for i in range(len(dataList[0])-1)]
    return dataList,label


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    np.random.seed(1) #固定种子使得每次打乱的结果都一致，方便测试
    dataSet,label = load_Cora()
    train_num = int(len(dataSet)*0.8)
    train_data = dataSet[:train_num]
    test_data = dataSet[train_num+1:]
    # trainTree = createTree(train_data,label)
    # storeTree(trainTree,'CORATree3.txt')
    trainTree = reloadTree('CORATree2.txt')
    # print(trainTree)

    createPlot(trainTree)
    test_labels = [i for i in range(1433)]
    errCount = 0.0
    for data in test_data:
        testVec = data[:-1]
        result = classify(trainTree,test_labels, testVec)
        if  result!=data[-1]:
            errCount+=1.0
    prob = (1-(errCount/len(test_data)))*100
    print(prob)



    # dataSet,label = load_IRIS()
    # num = 100
    # np.random.shuffle(dataSet)
    # train_Data = dataSet[:num]
    # test_data = dataSet[num+1:]
    # testTree = createTree(dataSet, label)
    # test_labels = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']
    # createPlot(testTree)
    # errCount = 0.0
    # for data in test_data:
    #     testVec = data[:-1]
    #     result = classify(testTree,test_labels, testVec)
    #     if result!=data[-1]:
    #         errCount+=1.0
    # prob = (1-(errCount/len(result)))*100
    # print(prob)
    # print(dataSet.head())
    # testTree = createTree(dataSet,label)
    # print(label.head())

    # myTree = retrieveTree(0)
    # myTree['no surfacing'][3] = 'maybe'
    # createPlot(myTree)
    # createPlot()
    # storeTree(myTree,'1.txt')
    # treess = reloadTree('1.txt')
    # print(treess)
    # dataSet, labels = createDataSet()
    # print(classify(myTree, labels,[1,0]))
    # # testTree = createTree(dataSet, labels)
    # # print(testTree)
    # fr = open('lenses.txt')
    # lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    # lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    # lensesTree = createTree(lenses, lensesLabels)
    # createPlot(lensesTree)
    # bestSplitIndex = chooseBestFeatureToSplit3(dataSet)
    # print(bestSplitIndex)
    # print(dataSet)
    # returnVec = splitDataSet(dataSet,0,1)
    # print(returnVec)
    # bestSplitIndex = chooseBestFeatureToSplit(dataSet)
    # print(bestSplitIndex)
    # shanEnt = calcShannonEnt(dataSet)
    # print(shanEnt)


# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
