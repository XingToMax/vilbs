# -*- encoding: utf-8 -*-

"""
@File        :  entity.py    
@Contact     :  1121584497@qq.com
@Modify Time :  2020/4/6 15:49      
@Author      :  Tomax  
@Version     :  1.0
@Desciption  :  None
"""

# import lib

class Context(object):
    def __init__(self):
        self.cnt = 0
        self.success = 0
    def success_increment(self):
        self.cnt = self.cnt + 1
        self.success = self.success + 1
    def fail_increment(self):
        self.cnt = self.cnt + 1
    def log(self):
        print(self.success, '/', self.cnt, 'rate: ', format(self.success / self.cnt * 100, ".2f") + '%')
    def refresh(self):
        self.cnt = 0
        self.success = 0

class SingleResult(object):
    def __init__(self, dict_data = None):
        if not dict_data:
            self.origin_id = 0
            self.current_id = 0
            self.mode = 0
            self.result = True
            self.distance = 0
            self.first_distance = 0
            self.second_distance = 0
            self.third_distance = 0
            self.first_label = 0
            self.second_label = 0
            self.third_label = 0
            self.detect_time = 0
            self.inpaint_time = 0
            self.search_time = 0
            self.answer_image_min_dis = 0
            self.answer_image_avg_dis = 0
            self.pic_name = ''
        else:
            self.__dict__.update(dict_data)

class Cell(object):
    def __init__(self, id, dis, cost):
        self.id = id
        self.dis = dis
        self.cost = cost