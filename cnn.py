# coding:utf-8
from time import sleep

from sklearn.manifold import TSNE
from tensorflow import keras
import preprocess
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from datetime import datetime
from tensorflow_core.python.keras import layers
import numpy as np
import tensorflow as tf
#如果是GPU，需要去掉注释，如果是CPU，则注释
# gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
# assert len(gpu) == 1
# tf.config.experimental.set_memory_growth(gpu[0], True)


def subtime(date1, date2):
    return date2 - date1


num_classes = 10    # 样本类别
length = 784        # 样本长度
number = 300  # 每类样本的数量
normal = True  # 是否标准化
rate = [0.5, 0.25, 0.25]  # 测试集验证集划分比例

path = r'data\0HP'
x_train, y_train, x_valid, y_valid, x_test, y_test = preprocess.prepro(
    d_path=path,
    length=length,
    number=number,
    normal=normal,
    rate=rate,
    enc=False, enc_step=28)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_valid = np.array(x_valid)
y_valid = np.array(y_valid)
x_test = np.array(x_test)
y_test = np.array(y_test)


print(x_train.shape)
print(x_valid.shape)
print(x_test.shape)
print(y_train.shape)
print(y_valid.shape)
print(y_test.shape)


y_train = [int(i) for i in y_train]
y_valid = [int(i) for i in y_valid]
y_test = [int(i) for i in y_test]

# 打乱顺序
index = [i for i in range(len(x_train))]
random.seed(1)
random.shuffle(index)
x_train = np.array(x_train)[index]
y_train = np.array(y_train)[index]

index1 = [i for i in range(len(x_valid))]
random.shuffle(index1)
x_valid = np.array(x_valid)[index1]
y_valid = np.array(y_valid)[index1]

index2 = [i for i in range(len(x_test))]
random.shuffle(index2)
x_test = np.array(x_test)[index2]
y_test = np.array(y_test)[index2]

print(x_train.shape)
print(x_valid.shape)
print(x_test.shape)
print(y_train)
print(y_valid)
print(y_test)
print("x_train的最大值和最小值：", x_train.max(), x_train.min())
print("x_test的最大值和最小值：", x_test.max(), x_test.min())

x_train = tf.reshape(x_train, (len(x_train), 784, 1))
x_valid = tf.reshape(x_valid, (len(x_valid), 784, 1))
x_test = tf.reshape(x_test, (len(x_test), 784, 1))


# 保存最佳模型
class CustomModelCheckpoint(keras.callbacks.Callback):
    def __init__(self, model, path):
        self.model = model
        self.path = path
        self.best_loss = np.inf

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs['val_loss']
        if val_loss < self.best_loss:
            print("\nValidation loss decreased from {} to {}, saving model".format(self.best_loss, val_loss))
            self.model.save_weights(self.path, overwrite=True)
            self.best_loss = val_loss

# 模型定义
def mymodel():
    inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))
    h1 = layers.Conv1D(filters=8, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)
    h1 = layers.MaxPool1D(pool_size=2, strides=2, padding='same')(h1)

    h1 = layers.Conv1D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu')(h1)
    h1 = layers.MaxPool1D(pool_size=2, strides=2, padding='same')(h1)

    h1 = layers.Flatten()(h1)
    h1 = layers.Dropout(0.6)(h1)
    h1 = layers.Dense(32, activation='relu')(h1)
    h1 = layers.Dense(10, activation='softmax')(h1)

    deep_model = keras.Model(inputs, h1, name="cnn")
    return deep_model

model = mymodel()

model.summary()
startdate = datetime.utcnow()  # 获取当前时间

# 编译模型
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

history = model.fit(x_train, y_train,
              batch_size=256, epochs=150, verbose=1,
              validation_data=(x_valid, y_valid),
              callbacks=[CustomModelCheckpoint(
  model, r'bestcnn.h5')])

#加载模型
filepath = r'bestcnn.h5'
model.load_weights(filepath=filepath)
# 编译模型
model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
# 评估模型
scores = model.evaluate(x_test, y_test, verbose=1)
print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))

y_predict = model.predict(x_test)
y_pred_int = np.argmax(y_predict, axis=1)
print(y_pred_int[0:5])
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_int, digits=4))

def acc_line():
    # 绘制acc和loss曲线
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))  # Get number of epochs

    # 画accuracy曲线
    plt.plot(epochs, acc, 'r', linestyle='-.')
    plt.plot(epochs, val_acc, 'b', linestyle='dashdot')
    plt.title('Training and validation accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["Accuracy", "Validation Accuracy"])

    plt.figure()

    # 画loss曲线
    plt.plot(epochs, loss, 'r', linestyle='-.')
    plt.plot(epochs, val_loss, 'b', linestyle='dashdot')
    plt.title('Training and validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Loss", "Validation Loss"])
    # plt.figure()
    plt.show()


acc_line()


# 绘制混淆矩阵
def confusion():
    y_pred_gailv = model.predict(x_test, verbose=1)
    y_pred_int = np.argmax(y_pred_gailv, axis=1)
    print(len(y_pred_int))
    con_mat = confusion_matrix(y_test.astype(str), y_pred_int.astype(str))
    print(con_mat)
    classes = list(set(y_train))
    classes.sort()
    plt.imshow(con_mat, cmap=plt.cm.Blues)
    indices = range(len(con_mat))
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    plt.colorbar()
    plt.xlabel('guess')
    plt.ylabel('true')
    for first_index in range(len(con_mat)):
        for second_index in range(len(con_mat[first_index])):
            plt.text(first_index, second_index, con_mat[second_index][first_index], va='center', ha='center')
    plt.show()
confusion()
