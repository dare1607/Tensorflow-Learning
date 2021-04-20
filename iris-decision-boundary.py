import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap

x = datasets.load_iris().data[:, :2]
y = datasets.load_iris().target
x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5

colors = ListedColormap(['#008c8c', '#AAFFAA', '#F2F4F9'])
h = .02
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
KNeighborsClassifier().fit(x, y)
Z = KNeighborsClassifier.predict(KNeighborsClassifier().fit(x, y), np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=colors)
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()
