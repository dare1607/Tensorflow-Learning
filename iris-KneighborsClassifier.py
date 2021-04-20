# 分类器对象，K-近邻算法，搜索训练集，寻找与测试集相符合的观测记录
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(100)
x = datasets.load_iris().data
y = datasets.load_iris().target
i = np.random.permutation(len(datasets.load_iris().data))
x_train = x[i[:-10]]
y_train = y[i[:-10]]
x_test = x[i[-10:]]
y_test = y[i[-10:]]

KNeighborsClassifier().fit(x_train, y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_neighbors=5, p=2, weights='uniform')
print(KNeighborsClassifier.predict(KNeighborsClassifier().fit(x_train, y_train), x_test))
print(y_test)
