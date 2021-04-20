import tensorflow as tf
import numpy as np
# 加入忽略
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = np.arange(0, 5)  # 生成步长为1的等差一维数组
b = tf.convert_to_tensor(a, dtype=tf.int64)  # 将a转化为张量
c = tf.fill([2, 2], 3)
d = tf.constant([1, 5], dtype=tf.int64)  # 创建张量
d_change = tf.cast(d, dtype=tf.float64)  # 强制转换数据类型
print('a: ', a, 'b: ', b)
print('d:', d, 'd_change:', d_change)
# 创建均值为0.5， 标准差为1的正态分布
e = tf.random.normal([2, 2], mean=0.5, stddev=1)
# 生成随机数在(0.5-2*0.6, 0.5+2*0.6)之间
f = tf.random.truncated_normal([3, 2], mean=0.6, stddev=0.5)
g = tf.random.uniform([4, 2], minval=0, maxval=1)  # 随机生成均匀分布数值
print('e:', e, 'f:', f, 'g:', g)
# tf.reduce_min(x)返回最小值，tf.reduce_max(x)返回最大值
# tf.reduce_mean(x)获取x中所有数的中值
# tf.matmul(x, y)矩阵相乘
# tf.data.Dataset.from_tensor_slices((x,y))加载数据，合并特征及标签
# tf.GradientTape().gradient()求导，tf.assign_sub()自减,enumerate返回索引
# tf.one_hot()将input转换为one-hot类型输出，即将多个数值联合放在一起作为多个相同类型的向量
# tf.argmax(x, axis=0/1)返回每一行/列最大值的索引
# tf.where(tf.greater(x, y), x, y)若x>y，返回x对应位置的元素，否则返回y对应位置的元素
# np.random.RandomState().rand生成[0,1)的随机数
