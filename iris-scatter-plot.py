import matplotlib.pyplot as plt
import matplotlib.patches as patch
from sklearn import datasets

x = datasets.load_iris().data[:, 0]
y = datasets.load_iris().data[:, 1]
species = datasets.load_iris().target

x_min, x_max = x.min() - .5, x.max() + .5
y_min, y_max = y.min() - .5, y.max() + .5

plt.figure()
plt.title('Iris Dataset')
plt.scatter(x, y, c=species)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
