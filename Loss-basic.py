# 梯度下降法
import tensorflow as tf

w = tf.Variable(tf.constant(5, dtype=tf.float32))
lr = 0.2  # 学习率，可调节学习率，查看不同学习率对损失的影响
epoch = 50  # 循环次数

for epoch in range(epoch):
    # lr = LR_BASE * LR_DECAY ** (epoch / LR_STEP)学习率自更新
    # tf.GradientTape()求导，进行梯度计算，使用with实现自动关闭
    with tf.GradientTape() as tape:
        loss = tf.square(w + 1)  # 求平方
    grads = tape.gradient(loss, w)

    w.assign_sub(lr * grads)
    print('After %s epoch, w is %f, loss is %f' % (epoch+1, w.numpy(), loss))
