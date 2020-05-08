# -*- encoding: utf-8 -*-

"""
@File        :  indoor_locate.py    
@Contact     :  1121584497@qq.com
@Modify Time :  2020/4/2 15:37      
@Author      :  Tomax  
@Version     :  1.0
@Desciption  :  None
"""

# import lib

from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet import MobileNet

from utils.distance_util import cos_dis
from utils.config_util import get_config
from utils.image_process_util import preprocess, cv_2_pil, pil_2_cv, batch_process_without_output, batch_process_with_result
from utils.database_util import read_database, read_id_map, result_read, result_save, PCAOptimize
from common.entity import Cell, SingleResult, Context
from inpaint.inpaint_model import InpaintCAModel
from detect.yolo3.yolo import YOLO

import tensorflow as tf
import time
import cv2
import numpy as np
import os

class IndoorLocate(object):

    """
    提供室内定位服务

    @:param config: 配置
    @:param features: 特征数据库
    @:param id_labels: 标注特征数据库的场景id
    @:param id_name_dict: 场景id和场景名称的映射集合
    @:param feature_extractor: 特征提取器
    @:param person_detector: 人形检测器
    @:param inpaintor: 图像修复器
    """

    def __init__(self, config):
        self.config = config
        self.image_height = 224
        self.image_width = 224
        sess_config = tf.ConfigProto()
        self.sess = tf.Session(config=sess_config)
        self.init_feature_extract_model()
        self.init_detect_model()
        self.init_inpaint_model()
        self.init_feature_database()
        self.init_name_id_file()

    def init_feature_extract_model(self):
        """
        加载特征提取模型
        :return:
        """
        self.feature_extractor = ResNet50(weights=self.config.feature_net.model, include_top=False)
        print('feature extract model loaded.')

    def init_detect_model(self):
        self.person_detector = YOLO(**vars(self.config.yolo))
        print('detect model loaded.')

    def init_inpaint_model(self):
        inpaint_model = InpaintCAModel()
        self.input_image_ph = tf.placeholder(
            tf.float32, shape=(1, self.image_height, self.image_width * 2, 3))
        stage1_output, stage2_output = inpaint_model.build_server_graph_rough(self.input_image_ph)
        stage1_output = (stage1_output + 1.) * 127.5
        stage2_output = (stage2_output + 1.) * 127.5
        stage1_output = tf.reverse(stage1_output, [-1])
        stage2_output = tf.reverse(stage2_output, [-1])
        stage1_output = tf.saturate_cast(stage1_output, tf.uint8)
        stage2_output = tf.saturate_cast(stage2_output, tf.uint8)
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'inpaint_net')
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(
                self.config.deepfill.checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        self.sess.run(assign_ops)
        self.deepfill_stage_1 = stage1_output
        self.deepfill_stage_2 = stage2_output
        print('inpaint model loaded.')

    def init_feature_database(self):
        self.features, self.id_labels = read_database(self.config.feature_net.database)
        self.reducer = PCAOptimize(self.features)
        self.low_features = self.reducer.reduce_batch(self.features)


    def init_name_id_file(self):
        self.id_name_dict = read_id_map(self.config.feature_net.id_name_map)

    def search(self, img, mode = 0):
        if mode > 0:
            img, detect_cost, inpaint_cost = self.image_optimize(img, mode)

        start_time = time.time()
        feature_ = self.feature_extract(img)
        target_dis = 1000
        target_id = None
        for feature, label in zip(self.features, self.id_labels):
            dis = cos_dis(feature_, feature)
            if dis < target_dis:
                target_id = label
                target_dis = dis
        return target_id, self.id_name_dict[target_id], target_dis, time.time() - start_time

    def search_top_k(self, img, k = 2, mode = 0):
        if mode > 0:
            img, detect_cost, inpaint_cost = self.image_optimize(img, mode)

        start_time = time.time()
        feature_ = self.feature_extract(img)
        cell_list = []
        for feature, label in zip(self.features, self.id_labels):
            dis = cos_dis(feature_, feature)
            cell_list.append(Cell(label, dis, time.time() - start_time))
        cell_list.sort(key=lambda cell: cell.dis, reverse=False)
        return cell_list[ : k]

    def search_for_single_result(self, img, origin_id, mode = 0, low = False):
        if mode > 0:
            img, detect_cost, inpaint_cost = self.image_optimize(img, mode)
        start_time = time.time()
        feature_ = self.feature_extract(img)
        if low:
            feature_ = self.reducer.reduce(feature_)
        answer_image_min_dis = 1000
        answer_image_avg_dis = 0
        cell_list = []
        feature_database = self.features
        if low:
            feature_database = self.low_features
        for feature, label in zip(feature_database, self.id_labels):
            dis = cos_dis(feature_, feature)
            if label == origin_id:
                if dis < answer_image_min_dis:
                    answer_image_min_dis = dis
                answer_image_avg_dis += dis
            cell_list.append(Cell(label, dis, 0))
        answer_image_avg_dis /= 2
        search_time = time.time() - start_time
        cell_list.sort(key=lambda cell: cell.dis, reverse=False)
        result = SingleResult()
        result.origin_id = origin_id
        result.current_id = cell_list[0].id
        result.result = (origin_id == result.current_id)
        result.distance = cell_list[0].dis
        result.search_time = search_time
        result.first_distance = cell_list[0].dis
        result.first_label = cell_list[0].id
        result.second_distance = cell_list[1].dis
        result.second_label = cell_list[1].id
        result.third_distance = cell_list[2].dis
        result.third_label = cell_list[2].id
        result.mode = mode
        result.answer_image_min_dis = answer_image_min_dis
        result.answer_image_avg_dis = answer_image_avg_dis
        if mode > 0:
            result.detect_time = detect_cost
            result.inpaint_time = inpaint_cost
        return result


    def feature_extract(self, img):
        """
        特征提取
        :param img: 待提取特征的图片
        :return: 提取到的特征
        """
        img = preprocess(img)
        feature = self.feature_extractor.predict(img)
        return feature.squeeze((0, 1, 2))

    def image_optimize(self, img, level = 3):
        """
        :param img:
        :param level:
        :return:
        """
        if level == 1:
            img, mask, flag = self.get_human_remove(img)
            return img, 0, 0
        if level == 2:
            return self.rough_remove(img)
        if level >= 3:
            return self.full_remove(img)

    def rough_remove(self, image):
        begin_time = time.time()
        image_, mask,flag = self.get_human_remove(image)
        # print('detect human cost: ', str(time.time() - begin_time))
        detect_cost = time.time() - begin_time
        begin_time = time.time()
        if flag:
            out = self.rough_inpaint_image(image_, mask)
            out = out[0][:, :, ::-1]
        else:
            out = image
        # print('rough_remove, cost :', str(time.time() - begin_time))
        inpaint_cost = time.time() - begin_time
        return out, detect_cost, inpaint_cost

    def full_remove(self, image):
        begin_time = time.time()
        image_, mask, flag = self.get_human_remove(image)
        detect_cost = time.time() - begin_time
        # print('detect, cost :', str(time.time() - begin_time))
        begin_time = time.time()
        out = None
        if flag:
            out = self.full_inpaint_image(image_, mask)
            out = out[0][:, :, ::-1]
        else:
            out = image
        # print('full_remove, cost :', str(time.time() - begin_time))
        inpaint_cost = time.time() - begin_time
        return out, detect_cost, inpaint_cost

    def get_human_remove(self, image):
        pil_image = cv_2_pil(image)
        coors = self.person_detector.detect_image_coors(pil_image)
        mask = np.zeros(np.shape(image))
        flag = len(coors) > 0
        for coor in coors:
            top, left, bottom, right = coor
            top = max(top - 5, 0)
            left = max(left - 5, 0)
            bottom = min(bottom + 5, np.shape(image)[0])
            right = min(right + 5, np.shape(image)[1])
            image[top : bottom, left : right] = np.ones([bottom - top, right - left, 3]) * 255
            mask[top : bottom, left : right] = np.ones([bottom - top, right - left, 3]) * 255
        return image, mask, flag

    def full_inpaint_image(self, image, mask):
        assert image.shape == mask.shape
        h, w, _ = image.shape
        grid = 4
        image = image[:h // grid * grid, :w // grid * grid, :]
        mask = mask[:h // grid * grid, :w // grid * grid, :]
        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)
        return self.sess.run(self.deepfill_stage_2, feed_dict={self.input_image_ph: input_image})

    def rough_inpaint_image(self, image, mask):
        assert image.shape == mask.shape
        h, w, _ = image.shape
        grid = 4
        image = image[:h // grid * grid, :w // grid * grid, :]
        mask = mask[:h // grid * grid, :w // grid * grid, :]
        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)
        return self.sess.run(self.deepfill_stage_1, feed_dict={self.input_image_ph: input_image})

    def generate_database(self, input_path, output_path):
        with open(output_path, "w") as f:
            for fo in os.listdir(input_path):
                img1 = cv2.imread(input_path + '/' + fo + '/1.png')
                img2 = cv2.imread(input_path + '/' + fo + '/2.png')
                feature1 = self.feature_extract(img1)
                feature2 = self.feature_extract(img2)
                for x in feature1:
                    f.write(str(x))
                    f.write(' ')
                f.write(fo)
                f.write('\n')

                for x in feature2:
                    f.write(str(x))
                    f.write(' ')
                f.write(fo)
                f.write('\n')

application_config = get_config('application.yml')
processor = IndoorLocate(application_config)
ctx = Context()

def search_img_test(img, label):
    id, name, dis, cost = processor.search(img)
    if label == id:
        ctx.success_increment()
        print('success, origin id:', label, 'current id:', id, 'current name:', name, 'dis:', dis, 'cost:', cost)
    else:
        ctx.fail_increment()
        print('fail, origin id:', label, 'current id:', id, 'current name:', name, 'dis:', dis, 'cost:', cost)

def search_img_test_top2(img, label):
    cells = processor.search_top_k(img, k=3, mode=2)
    flag = False
    for cell in cells:
        if label == cell.id:
            ctx.success_increment()
            print('success, origin id:', label, 'current id:', id, 'dis:', cell.dis, 'cost:', cell.cost)
            flag = True
            break
    if not flag:
        ctx.fail_increment()
        print('fail, origin id:', label)

def record_full_optimize_img_test(img, label, item):
    result = processor.search_for_single_result(img, label, mode=3, low=False)
    result.pic_name = item
    return result

def record_rough_optimize_img_test(img, label, item):
    result = processor.search_for_single_result(img, label, mode=2, low=False)
    result.pic_name = item
    return result

def record_remove_img_test(img, label, item):
    result = processor.search_for_single_result(img, label, mode=1, low=False)
    result.pic_name = item
    return result

def record_no_optimize_img_test(img, label, item):
    result = processor.search_for_single_result(img, label, mode=0, low=False)
    result.pic_name = item
    return result

def record_no_human_img_test(img, label, item):
    result = processor.search_for_single_result(img, label, mode=0, low=False)
    result.pic_name = item
    return result

if __name__ == '__main__':
    # batch_process_without_output('E:\\ml\\datasets\\indoor_human', search_img_test)
    # processor.generate_database('E:/ml/datasets/indoor_2', 'E:/ml/datasets/indoor_sfdb_standard.txt')
    # ctx.log()

    # results = batch_process_with_result('E:/ml/datasets/indoor_human', record_full_optimize_img_test)
    # print(len(results))
    # result_save('E:/ml/datasets/full_inpaint_with_human_result(reduce).txt', results)

    # results = batch_process_with_result('E:/ml/datasets/indoor_2', record_no_human_img_test)
    # print(len(results))
    # result_save('E:/ml/datasets/no_human_result.txt', results)

    # results = batch_process_with_result('E:/ml/datasets/indoor_human', record_no_optimize_img_test)
    # print(len(results))
    # result_save('E:/ml/datasets/no_optimize_with_human_result.txt', results)

    # results = batch_process_with_result('E:/ml/datasets/indoor_human', record_remove_img_test)
    # print(len(results))
    # result_save('E:/ml/datasets/remove_human_with_human_result.txt', results)

    results = batch_process_with_result('E:/ml/datasets/indoor_human', record_rough_optimize_img_test)
    print(len(results))
    result_save('E:/ml/datasets/rough_inpaint_with_human_result.txt', results)