# -*- encoding: utf-8 -*-

"""
@File        :  database_util.py    
@Contact     :  1121584497@qq.com
@Modify Time :  2020/4/2 20:27      
@Author      :  Tomax  
@Version     :  1.0
@Desciption  :  None
"""

# import lib
import json
import numpy as np
from sklearn.decomposition import PCA

def read_database(database_path):
    with open(database_path, "r") as f:
        labels = []
        features = []
        data = f.readlines()
        for data_ in data:
            feature_s = data_.split(' ')
            feature = feature_s[:2048]
            label = feature_s[2048]
            label = label.replace('\n', '')
            for i in range(len(feature)):
                feature[i] = float(feature[i])
            labels.append(label)
            features.append(feature)
    return features, labels

def read_id_map(id_map_path):
    with open(id_map_path, "r") as f:
        id_dict = {}
        data = f.readlines()
        for data_ in data:
            if len(data_) <= 0:
                continue
            items = data_.split(" ")
            id = items[0]
            label = items[1]
            label = label.replace('\n', '')
            id_dict[id] = label
        return id_dict

def generate_database(database_path, imgs_path):
    # with open(database_path, "w") as f:
    #     for fo in os.listdir(imgs_path):
    #         img1 = cv2.imread(imgs_path + '/' + fo + '/1.png')
    #         img2 = cv2.imread(imgs_path + '/' + fo + '/2.png')
    #         feature1 = extract_feature(img1)
    #         feature2 = extract_feature(img2)
    #         for x in feature1:
    #             f.write(str(x))
    #             f.write(' ')
    #         f.write(fo)
    #         f.write('\n')
    #
    #         for x in feature2:
    #             f.write(str(x))
    #             f.write(' ')
    #         f.write(fo)
    #         f.write('\n')
    pass

def result_save(path, results):
    with open(path, "w") as f:
        for result in results:
            res_str = json.dumps(result.__dict__)
            print(res_str)
            f.write(res_str + '\n')

def result_read(path):
    data_list = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            # print(line)
            data_list.append(data)
    return data_list

class PCAOptimize(object):
    def __init__(self, features, n=100):
        self.pca = PCA(n_components=n)
        self.pca.fit(features)

    def reduce(self, feature):
        data = np.expand_dims(feature, axis=0)
        data = self.pca.transform(data)
        data = np.squeeze(data)
        return data

    def reduce_batch(self, features):
        return self.pca.transform(features)

if __name__ == '__main__':
    features, labels = read_database('E:/ml/datasets/indoor_sfdb_standard.txt')
    optimizer = PCAOptimize(features)

    data = np.ones([2, 2048])
    data = optimizer.reduce_batch(data)
    print(np.shape(data))
    print(data)