# -*- encoding: utf-8 -*-

"""
@File        :  image_process_util.py    
@Contact     :  1121584497@qq.com
@Modify Time :  2020/4/2 18:03      
@Author      :  Tomax  
@Version     :  1.0
@Desciption  :  None
"""

# import lib
import cv2
import os
import numpy as np

from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import Image

def batch_process(input_folder, output_folder, handler = None, cover = False, generate_index = False):
    """
    批量图片处理(处理两层文件目录，外层为标签，内层为文件)
    :param input_folder: 输入路径
    :param output_folder: 输出路径
    :param handler: 图片处理方法，传入图片，返回处理的图片
    :param cover: 是否覆盖历史处理数据
    :param generate_index: 是否按index自增输出文件，否的话保持图像原名
    :return:
    """
    for label in os.listdir(input_folder):
        if os.path.exists(output_folder + '/' + label):
            continue
        else:
            os.makedirs(output_folder + '/' + label)
        index = 0
        for item in os.listdir(input_folder + '/' + label):
            if '.jpg' in item or '.png' in item:
                img = cv2.imread(input_folder + '/' + label + '/' + item)
                img = handler(img)
                out_name = ''
                if generate_index:
                    out_name = str(index) + '.png'
                else:
                    out_name = item
                cv2.imwrite(output_folder + '/' + label + '/' + out_name, img)


def batch_process_without_output(input_folder, handler = None):
    """
    批量图片处理(处理两层文件目录，外层为标签，内层为文件)
    :param input_folder: 输入路径
    :param handler: 图片处理方法，传入图片，返回处理的图片
    :return:
    """
    for label in os.listdir(input_folder):
        for item in os.listdir(input_folder + '/' + label):
            if '.jpg' in item or '.png' in item:
                print('current process image:', label + '/' + item)
                img = cv2.imread(input_folder + '/' + label + '/' + item)
                handler(img, label)

def batch_process_with_result(input_folder, handler = None):
    """
        批量图片处理(处理两层文件目录，外层为标签，内层为文件)
        :param input_folder: 输入路径
        :param handler: 图片处理方法，传入图片，返回处理的图片
        :return: results
        """
    results = []
    for label in os.listdir(input_folder):
        for item in os.listdir(input_folder + '/' + label):
            if '.jpg' in item or '.png' in item:
                print('current process image:', label + '/' + item)
                img = cv2.imread(input_folder + '/' + label + '/' + item)
                result = handler(img, label, item)
                results.append(result)
    return results

def preprocess(img):
    """
    图像预处理，使可以给resnet操作
    :param img:
    :return:
    """
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)  # 去均值中心化，preprocess_input函数详细功能见注释
    return x


def cv_2_pil(cv2_image):
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))


def pil_2_cv(pil_image):
    return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)