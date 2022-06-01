# 博客：https://blog.csdn.net/qq_38918049/article/details/124948664?spm=1001.2014.3001.5501
# github：https://github.com/boating-in-autumn-rain?tab=repositories
# 微信公众号：秋雨行舟
# B站：秋雨行舟
#
# 该项目涉及数据集以及相关安装包在公众号《秋雨行舟》回复轴承即可领取。
# 对于该项目有疑问的可以在上述四个平台中留言，看到了就会回复。
# 该项目对应的视频可在B站搜索《秋雨行舟》进行观看学习。
# 欢迎交流学习，共同进步

import numpy as np
import os
from PIL import Image

def read_directory(directory_name,height,width,normal):
    file_list=os.listdir(directory_name)
    file_list.sort(key=lambda x: int(x.split('-')[0]))
    img = []
    label0=[]
    
    for each_file in file_list:
        img0 = Image.open(directory_name + '/'+each_file)
        img0 = img0.convert('L')
        gray = img0.resize((height,width))
        img.append(np.array(gray).astype(np.float))
        label0.append(float(each_file.split('.')[0][-1]))
    if normal:
        data = np.array(img)/255.0#归一化
    else:
        data = np.array(img)
    data=data.reshape(-1,1,height,width)
    label=np.array(label0)
    return data,label 
