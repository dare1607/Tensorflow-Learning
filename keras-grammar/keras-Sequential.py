import tensorflow as tf
import numpy as np
from sklearn import datasets

features = datasets.load_iris().data
labels = datasets.load_iris().target

# 随机打乱顺序
np.random.seed(110)
np.random.shuffle(features)
np.random.seed(110)
np.random.shuffle(labels)
tf.random.set_seed(110)

# Sequential是一个容器，用来描述神经网络的网络结构
# 拉直层tf.keras.layers.Flatten()将输入特征展成一维数组
# 全连接层tf.keras.layers.Dense(神经元个数，activate='激活函数'，kernel_regularizer='正则化方式‘)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())
])

# compile可配置神经网络的训练方法，告知训练所用优化器、损失函数和准确率评测标准
# model.compile(optimizer=优化器, loss=损失函数， metric=['准确率'])
model.compile(
    optimizer=tf.keras.optimizers.SGD(lr=0.1),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['sparse_categorical_accuracy']
)

# fit函数执行训练过程
model.fit(features, labels, batch_size=32, epochs=500, validation_split=0.2, validation_freq=20)

# summary 函数用于打印网络结构和参数统计
model.summary()
