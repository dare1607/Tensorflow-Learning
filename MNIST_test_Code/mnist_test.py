import numpy as np
import struct
import matplotlib.pyplot as plt
import tensorflow as tf

train_images_idx3_ubyte_file = 'fashion-mnist-master/data/fashion/train-images.idx3-ubyte'
train_labels_idx1_ubyte_file = 'fashion-mnist-master/data/fashion/train-labels.idx1-ubyte'
test_images_idx3_ubyte_file = 'fashion-mnist-master/data/fashion/t10k-images.idx3-ubyte'
test_labels_idx1_ubyte_file = 'fashion-mnist-master/data/fashion/t10k-labels.idx1-ubyte'


def decode_idx3_ubyte(idx3_ubyte_file):
    # 解析idx3文件的通用函数
    bin_date = open(idx3_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_row, num_cols = struct.unpack_from(fmt_header, bin_date, offset)
    print('模数:%d, 图片数量:%d张, 图片大小:%d*%d' % (magic_number, num_images, num_row, num_cols))

    image_size = num_row * num_cols
    offset += struct.calcsize(fmt_header)
    print('偏移位置：', offset)
    fmt_image = '>' + str(image_size) + 'B'  # B表示的是一个字节 > 表示的是大端法则，content_numb表示的是多少个字节
    print(fmt_image, offset, struct.calcsize(fmt_image))
    images = np.empty((num_images, num_row, num_cols))
    # plt.figure()
    for m in range(num_images):
        if (m + 1) % 10000 == 0:
            print('已解析 %d' % (m + 1) + '张')
            print(offset)
        images[m] = np.array(struct.unpack_from(fmt_image, bin_date, offset)).reshape((num_row, num_cols))
        # print(images[m])
        offset += struct.calcsize(fmt_image)
    #         plt.imshow(images[m], 'gray')
    #         plt.pause(0.00001)
    #         plt.show()
    # plt.show()
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    # 解析idx1文件的通常函数
    bin_data = open(idx1_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('模数:%d, 图片数量:%d张' % (magic_number, num_images))

    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for m in range(num_images):
        if (m + 1) % 10000 == 0:
            print('已解析 %d' % (m + 1) + '张')
        labels[m] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)


if __name__ == '__main__':
    train_images = load_train_images()
    train_labels = load_train_labels()
    test_images = load_test_images()
    test_labels = load_test_labels()

    # 查看前十个数据集激起标签读取是否正确
    for i in range(10):
        print(train_labels[i])
        plt.imshow(train_images[i], cmap='gray')
        plt.pause(0.000001)
        plt.show()
    print('done')

    # 处理成tensor
    train_images = tf.expand_dims(train_images, axis=-1)
    print(train_images)
    train_labels = tf.keras.utils.to_categorical(y=train_labels, num_classes=10)
    print(train_labels)
    train_data = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(128)

    # 构建网络模型
    input_x = tf.keras.Input(shape=[28, 28, 1], name='in_1')
    conv = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], padding='VALID', activation=tf.nn.sigmoid)(input_x)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.MaxPool2D()(conv)
    conv = tf.keras.layers.Conv2D(filters=16, kernel_size=[3, 3], padding='VALID', activation=tf.nn.sigmoid)(conv)
    flat = tf.keras.layers.Flatten()(conv)
    process = tf.keras.layers.Dense(128, activation='sigmoid', name='d_1')(conv)
    outputs_label = tf.keras.layers.Dense(10, activation='softmax', name='output')(flat)
    model = tf.keras.Model(inputs=input_x, outputs=outputs_label)

    # 训练
    opt = tf.keras.optimizers.Adam(1e-3)
    model.compile(optimizer=opt, loss=tf.keras.losses.mse, metrics='accuracy')
    model.fit(train_data, epochs=5)
    model.save('./mnist_test_conv_model.h5')

    # 测试
    mnist = tf.keras.datasets.mnist
    new_model = tf.keras.models.load_model('./mnist_test_conv_model.h5')
    x_test = tf.expand_dims(test_images, axis=-1)
    y_test = tf.keras.utils.to_categorical(y=test_labels, num_classes=10)
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)
    score = new_model.evaluate(test_data)
    print(score)
