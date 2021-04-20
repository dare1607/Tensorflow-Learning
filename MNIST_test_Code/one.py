# import os
# import tensorflow as tf
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# print(tf.__version__)

# x = tf.constant(4)
# x = tf.ones((3, 3))
# x = tf.zeros((2, 3))
# x = tf.eye(3)
# x = tf.random.normal((3, 3), mean=0, stddev=1)  # 正态分布
# x = tf.random.uniform((1, 3), minval=0, maxval=1)  # 均匀分布
# x = tf.range(start=1, limit=10, delta=2)
# x = tf.cast(x, dtype=tf.float64)
# print(x)

# knn最邻近分类
# import numpy as np
# from sklearn import neighbors
# knn = neighbors.KNeighborsClassifier()
# data = np.array([[3, 104], [2, 100], [1, 81], [101, 10], [99, 5], [98, 2]])
# labels = np.array([1, 1, 1, 2, 2, 2])
# knn.fit(data, labels)
# print(knn.predict([[10, 92]]))

# SVM向量机算法
# from sklearn import svm
# import numpy as np
# import matplotlib.pyplot as plt
#
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 7))
# ax0, ax1, ax2, ax3 = axes.flatten()
# x = [[1, 8], [3, 20], [1, 15], [3, 35], [5, 35], [4, 40], [7, 80], [6, 49]]
# y = [1, 1, -1, -1, 1, -1, -1, 1]
#
# title = ['LinearSVC (linear kernel)', 'SVC with polynomial (degree 3) kernel',
#           'SVC with RBF kernel', 'SVC with Sigmoid kernel']
# rdm_arr = np.random.randint(1, 15, size=(15, 2))
# def drawPoint(ax, clf, tn):
#     for i in x:
#         ax.set_title(title[tn])
#         res = clf.predict(np.array(i).reshape(1, -1))
#         if res > 0:
#             ax.scatter(i[0], i[1], c='r', marker='*')
#         else:
#             ax.scatter(i[0], i[1], c='g', marker='*')
#     for i in rdm_arr:
#         res = clf.predict(np.array(i).reshape(1, -1))
#         if res > 0:
#             ax.scatter(i[0], i[1], c='r', marker='.')
#         else:
#             ax.scatter(i[0], i[1], c='g', marker='.')
#
# if __name__ == '__main__':
#     for n in range(0, 4):
#         if n==0:
#             clf = svm.SVC(kernel='linear').fit(x, y)
#             drawPoint(ax0, clf, 0)
#         elif n==1:
#             clf = svm.SVC(kernel='poly', degree=3).fit(x, y)
#             drawPoint(ax1, clf, 1)
#         elif n==2:
#             clf = svm.SVC(kernel='rbf').fit(x, y)
#             drawPoint(ax2, clf, 2)
#         else:
#             clf = svm.SVC(kernel='sigmoid').fit(x, y)
#             drawPoint(ax3, clf, 3)
#     plt.show()