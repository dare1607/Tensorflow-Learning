import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
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


# 使用class封装网络结构，call()函数中调用__init__()函数中初始化的网络快
# 实现前向传播并返回推理值
class IrisModel(Model):
    def get_config(self):
        pass

    def __init__(self):
        super(IrisModel, self).__init__()
        self.d1 = Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, x, training=None, mask=None):
        y = self.d1(x)
        return y


model = IrisModel()

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
