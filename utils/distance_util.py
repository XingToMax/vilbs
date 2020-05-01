# -*- encoding: utf-8 -*-

"""
@File        :  distance_util.py    
@Contact     :  1121584497@qq.com
@Modify Time :  2020/4/2 18:19      
@Author      :  Tomax  
@Version     :  1.0
@Desciption  :  None
"""

# import lib
import numpy as np
from scipy.spatial.distance import pdist

def cos_dis(v1, v2):
    """
    计算余弦距离
    :param v1:
    :param v2:
    :return:
    """
    return pdist(np.stack([v1, v2]), 'cosine')[0]

def euc_dis(v1, v2):
    """
    计算欧式距离
    :param v1:
    :param v2:
    :return:
    """
    return np.sqrt(np.sum(np.square(v1 - v2)))