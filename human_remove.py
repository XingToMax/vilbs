# -*- encoding: utf-8 -*-

"""
@File        :  human_remove.py    
@Contact     :  1121584497@qq.com
@Modify Time :  2020/5/8 19:23      
@Author      :  Tomax  
@Version     :  1.0
@Desciption  :  None
"""

# import lib

import os
import time
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from detect.yolo3.yolo import YOLO
from inpaint.inpaint_model import InpaintCAModel
from utils.config_util import get_config

class HumanRemove:
    def __init__(self, config, image_height = 224, image_width = 224):
        self.config = config
        self.image_height = image_height
        self.image_width = image_width
        self.init_yolo()
        self.init_deepfill()

    def rough_remove(self, image):
        begin_time = time.time()
        image_, mask,flag = self.get_human_remove(image)
        print('detect human cost: ', str(time.time() - begin_time))
        begin_time = time.time()
        if flag:
            out = self.rough_inpaint_image(image_, mask)
            out = out[0][:, :, ::-1]
        else:
            out = image
        print('rough_remove, cost :', str(time.time() - begin_time))
        return out

    def full_remove(self, image):
        begin_time = time.time()
        image_, mask, flag = self.get_human_remove(image)
        print('detect, cost :', str(time.time() - begin_time))
        begin_time = time.time()
        out = None
        if flag:
            out = self.full_inpaint_image(image_, mask)
            out = out[0][:, :, ::-1]
        else:
            out = image
        print('full_remove, cost :', str(time.time() - begin_time))
        return out

    def remove(self, image):
        begin_time = time.time()
        image, mask, flag = self.get_human_remove(image)
        out1, out2 = self.inpaint_image(image, mask)
        print('full_remove, cost :', str(time.time() - begin_time))
        return out1[0][:, :, ::-1], out2[0][:, :, ::-1]

    def init_deepfill(self):
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess = tf.Session(config=sess_config)

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
        sess.run(assign_ops)
        print('deepfill model loaded.')
        self.sess = sess
        self.deepfill_stage_1 = stage1_output
        self.deepfill_stage_2 = stage2_output

    def init_yolo(self):
        self.yolo = YOLO(**vars(self.config.yolo))

    def get_human_remove(self, image):
        pil_image = HumanRemove.cv_2_pil(image)
        coors = self.yolo.detect_image_coors(pil_image)
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

    def inpaint_image(self, image, mask):
        assert image.shape == mask.shape
        h, w, _ = image.shape
        grid = 4
        image = image[:h // grid * grid, :w // grid * grid, :]
        mask = mask[:h // grid * grid, :w // grid * grid, :]
        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)
        return self.sess.run(self.deepfill_stage_1, feed_dict={self.input_image_ph: input_image}), self.sess.run(self.deepfill_stage_2, feed_dict={self.input_image_ph: input_image})

    @staticmethod
    def cv_2_pil(cv2_image):
        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

    @staticmethod
    def pil_2_cv(pil_image):
        return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)

    @staticmethod
    def mask_generate(mask):
        mask_ = np.array(mask, np.float)
        h, w, _ = mask.shape
        for i in range(h):
            for j in range(w):
                mask_[i][j] = mask[i][j] > 127.5
        return mask_


application_config = get_config('application.yml')
processor = HumanRemove(config=application_config, image_height=224, image_width=224)

def human_remove_demo(image):
    return processor.full_remove(image)

if __name__ == '__main__':
    path = 'E:\\ml\\test-data\\metric\\8.png'
    image = human_remove_demo(cv2.resize(cv2.imread(path), (224, 224)))
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
