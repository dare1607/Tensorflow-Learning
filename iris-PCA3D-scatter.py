import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

species = datasets.load_iris().target
# 降维，n_components设置维数,fit_transform设置降维数据
x_reduce = PCA(n_components=3).fit_transform(datasets.load_iris().data)

fig = plt.figure()
ax = Axes3D(fig)
ax.set_title('Iris Dataset PCA 3D', size=14)
ax.scatter(x_reduce[:, 0], x_reduce[:, 1], x_reduce[:, 2], c=species)
ax.set_xlabel('first')
ax.set_ylabel('second')
ax.set_zlabel('three')
ax.w_xaxis.set_ticklabels(())
ax.w_yaxis.set_ticklabels(())
ax.w_zaxis.set_ticklabels(())
plt.show()
