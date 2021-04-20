import tensorflow as tf
import numpy as np

rdm = np.random.RandomState(seed=200)  # 生成[0,1)之间的随机数
x = rdm.rand(32, 2)  # 生成32行2列的矩阵
# print(x)
y_ = [[x1 + x2 + (rdm.rand() / 10.0 - 0.05)] for (x1, x2) in x]  # 生成噪声[0,1)/10=[0,0.1); [0,0.1)-0.05=[-0.05,0.05)
x = tf.cast(x, dtype=tf.float32)
w1 = tf.Variable(tf.random.normal([2, 1], stddev=1, seed=2))
b1 = tf.Variable(tf.random.normal([1], stddev=1, seed=2))

epoch = 20000
lr_start = 0.002
lr_end = 0.999
lr_step = 1

for epoch in range(epoch):
    lr = lr_start * lr_end ** (epoch / lr_step)
    with tf.GradientTape() as tape:
        y = tf.matmul(x, w1) + b1
        loss_mean = tf.reduce_mean(tf.square(y_ - y))  # 均方根损失函数
        # loss = tf.reduce_sum(tf.where(tf.greater(y, y_) * Cost), (y_ -y)*PROFIT)自定义损失函数
        # tf.losses.categorical_crossentropy(x, y)交叉损失函数
    grads = tape.gradient(loss_mean, [w1, b1])
    # 实现梯度更新 w1 = w1 - lr * w1_grad    b = b - lr * b_grad
    w1.assign_sub(lr * grads[0])
    b1.assign_sub(lr * grads[1])

    if epoch % 500 ==0:
        print('Epoch: {}, loss: {}, w: {}'.format(epoch, loss_mean, w1.numpy()))
print("Final w1 is: ", w1.numpy())
