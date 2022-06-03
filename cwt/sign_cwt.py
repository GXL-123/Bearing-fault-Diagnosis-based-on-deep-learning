# 博客：https://blog.csdn.net/qq_38918049/article/details/124948664?spm=1001.2014.3001.5501
# github：https://github.com/boating-in-autumn-rain?tab=repositories
# 微信公众号：秋雨行舟
# B站：秋雨行舟
#
# 该项目涉及数据集以及相关安装包在公众号《秋雨行舟》回复轴承即可领取。
# 对于该项目有疑问的可以在上述四个平台中留言，看到了就会回复。
# 该项目对应的视频可在B站搜索《秋雨行舟》进行观看学习。
# 欢迎交流学习，共同进步

import pywt
import matplotlib.pyplot as plt
import numpy as np
from sign import preprocess

path = r'../sign/data/0HP'
x_train, y_train, x_valid, y_valid, x_test, y_test = preprocess.prepro(
    d_path=path,
    length=784,
    number=30,
    normal=True,
    rate=[0.6, 0.2, 0.2],
    enc=False, enc_step=28)

for i in range(0, len(x_train)):
    N = 784
    fs = 12000
    t = np.linspace(0, 784 / fs, N, endpoint=False)
    wavename = 'cmor3-3'
    totalscal = 256
    fc = pywt.central_frequency(wavename)
    cparam = 2 * fc * totalscal
    scales = cparam / np.arange(totalscal, 1, -1)
    [cwtmatr, frequencies] = pywt.cwt(x_train[i], scales, wavename, 1.0 / fs)
    plt.contourf(t, frequencies, abs(cwtmatr))

    plt.axis('off')
    plt.gcf().set_size_inches(784 / 100, 784 / 100)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    x = r'./cwt_picture/train/' + str(i) + '-' + str(y_train[i]) + '.jpg'
    plt.savefig(x)