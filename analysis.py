# -*- encoding: utf-8 -*-

"""
@File        :  analysis.py    
@Contact     :  1121584497@qq.com
@Modify Time :  2020/4/6 15:28      
@Author      :  Tomax  
@Version     :  1.0
@Desciption  :  None
"""

# import lib
import json
import numpy as np

from pyecharts.charts import Bar, Pie
from pyecharts import options as opts

from common.entity import SingleResult
from utils.database_util import result_read

class Ana(object):
    def __init__(self):
        self.cnt = 0
        self.suc = 0
    def success_increment(self):
        self.cnt = self.cnt + 1
        self.suc = self.suc + 1
    def fail_increment(self):
        self.cnt = self.cnt + 1
    def log(self):
        print(self.suc, '/', self.cnt, 'rate: ', format(self.suc / self.cnt * 100, ".4f") + '%')
    def refresh(self):
        self.cnt = 0
        self.suc = 0
    def batch_update(self, results):
        for result in results:
            if result.result:
                self.success_increment()
            else:
                self.fail_increment()

no_human_results = [SingleResult(result) for result in result_read('E:/ml/datasets/no_human_result.txt')]
no_optimize_results = [SingleResult(result) for result in result_read('E:/ml/datasets/no_optimize_with_human_result.txt')]
remove_human_results = [SingleResult(result) for result in result_read('E:/ml/datasets/remove_human_with_human_result.txt')]
rough_optimize_results = [SingleResult(result) for result in result_read('E:/ml/datasets/rough_inpaint_with_human_result.txt')]
full_optimize_results = [SingleResult(result) for result in result_read('E:/ml/datasets/full_inpaint_with_human_result(reduce).txt')]

def float_render(val):
    return float(format(val, ".4f"))

def search_rate_count():
    cnt_list = []
    suc_list = []
    ana = Ana()
    ana.batch_update(no_human_results)
    ana.suc -= 124
    ana.cnt -= 124
    cnt_list.append(ana.cnt)
    suc_list.append(ana.suc)
    ana.log()
    ana.refresh()
    ana.batch_update(no_optimize_results)
    cnt_list.append(ana.cnt)
    suc_list.append(ana.suc)
    ana.log()
    ana.refresh()
    ana.batch_update(remove_human_results)
    cnt_list.append(ana.cnt)
    suc_list.append(ana.suc)
    ana.log()
    ana.refresh()
    ana.batch_update(rough_optimize_results)
    cnt_list.append(ana.cnt)
    suc_list.append(ana.suc)
    ana.log()
    ana.refresh()
    ana.batch_update(full_optimize_results)
    cnt_list.append(ana.cnt)
    suc_list.append(ana.suc)
    ana.log()
    ana.refresh()
    print(cnt_list)
    print(suc_list)

    bar = (
        Bar()
            .add_xaxis(["无行人图片", "有行人图片", "去除行人", "粗糙修复", "精细修复"])
            .add_yaxis("", [float(format(x / y, ".4f")) for x, y in zip(suc_list, cnt_list)])
            .set_global_opts(title_opts=opts.TitleOpts(title="准确率对比", subtitle="图片检索准确率对比图"))
    )
    bar.render(path='charts/rate_count.html')

def time_analysis():
    search_times = [result.search_time for result in full_optimize_results]
    detect_times = [result.detect_time for result in full_optimize_results]
    rough_times = [result.inpaint_time for result in rough_optimize_results if result.inpaint_time > 0]
    full_times = [result.inpaint_time for result in full_optimize_results if result.inpaint_time > 0]
    search_times.remove(np.max(search_times))
    detect_times.remove(np.max(detect_times))
    rough_times.remove(np.max(rough_times))
    full_times.remove(np.max(full_times))
    print('检索时间(ms)', float_render(np.mean(search_times)), float_render(np.max(search_times)), float_render(np.min(search_times)))
    print('检测时间(ms)', float_render(np.mean(detect_times)), float_render(np.max(detect_times)), float_render(np.min(detect_times)))
    print('粗糙修复时间(ms)', float_render(np.mean(rough_times)), float_render(np.max(rough_times)), float_render(np.min(rough_times)))
    print('精细修复时间(ms)', float_render(np.mean(full_times)), float_render(np.max(full_times)), float_render(np.min(full_times)))

def dis_analysis():
    no_human_dis_list = [result.answer_image_avg_dis for result in no_human_results if result.distance > 1e-10]
    no_optimize_dis_list = [result.answer_image_avg_dis for result in no_optimize_results]
    remove_human_dis_list = [result.answer_image_avg_dis for result in remove_human_results]
    rough_optimize_dis_list = [result.answer_image_avg_dis for result in rough_optimize_results]
    full_optimize_dis_list = [result.answer_image_avg_dis for result in full_optimize_results]

    print('无行人', float_render(np.mean(no_human_dis_list)), float_render(np.max(no_human_dis_list)), float_render(np.min(no_human_dis_list)))
    print('有行人无优化', float_render(np.mean(no_optimize_dis_list)), float_render(np.max(no_optimize_dis_list)), float_render(np.min(no_optimize_dis_list)))
    print('直接去除行人', float_render(np.mean(remove_human_dis_list)), float_render(np.max(remove_human_dis_list)), float_render(np.min(remove_human_dis_list)))
    print('粗糙修复', float_render(np.mean(rough_optimize_dis_list)), float_render(np.max(rough_optimize_dis_list)), float_render(np.min(rough_optimize_dis_list)))
    print('精细修复', float_render(np.mean(full_optimize_dis_list)), float_render(np.max(full_optimize_dis_list)), float_render(np.min(full_optimize_dis_list)))

    columns = ['0~0.1', '0.1~0.2', '0.2~0.3', '0.3~0.4', '0.4~0.5', '0.5~0.6']
    data1 = [0, 0, 0, 0, 0, 0]
    data2 = [0, 0, 0, 0, 0, 0]
    data3 = [0, 0, 0, 0, 0, 0]
    data4 = [0, 0, 0, 0, 0, 0]
    data5 = [0, 0, 0, 0, 0, 0]

    no_human_dis_list =       [int(float_render(dis) * 10000) // 1000 for dis in no_human_dis_list]
    no_optimize_dis_list =    [int(float_render(dis) * 10000) // 1000 for dis in no_optimize_dis_list]
    remove_human_dis_list =   [int(float_render(dis) * 10000) // 1000 for dis in remove_human_dis_list]
    rough_optimize_dis_list = [int(float_render(dis) * 10000) // 1000 for dis in rough_optimize_dis_list]
    full_optimize_dis_list =  [int(float_render(dis) * 10000) // 1000 for dis in full_optimize_dis_list]
    for dis in no_human_dis_list:
        data1[dis] += 1
    for dis in no_optimize_dis_list:
        data2[dis] += 1
    for dis in remove_human_dis_list:
        data3[dis] += 1
    for dis in rough_optimize_dis_list:
        data4[dis] += 1
    for dis in full_optimize_dis_list:
        data5[dis] += 1
    print(data1, np.sum(data1))
    print(data2, np.sum(data2))
    print(data3, np.sum(data3))
    print(data4, np.sum(data4))
    print(data5, np.sum(data5))

    pie1 = (
        Pie()
            .add("无行人", [[label, value] for label, value in zip(columns, data1)])  # 加入数据
            .set_global_opts(title_opts=opts.TitleOpts(title="无行人距离分布图"),
                             legend_opts=opts.LegendOpts(pos_left=160))  # 全局设置项
            .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}")))  # 样式设置项
    pie1.render('charts/dis_distribution_1.html')

    pie1 = (
        Pie()
            .add("有行人无优化", [[label, value] for label, value in zip(columns, data2)])  # 加入数据
            .set_global_opts(title_opts=opts.TitleOpts(title="有行人距离分布图"),
                             legend_opts=opts.LegendOpts(pos_left=160))  # 全局设置项
            .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}")))  # 样式设置项
    pie1.render('charts/dis_distribution_2.html')

def top_2_analysis():
    ana = Ana()
    for result in no_human_results:
        if result.origin_id == result.first_label or result.origin_id == result.second_label:
            ana.success_increment()
        else:
            ana.fail_increment()
    ana.suc -= 124
    ana.cnt -= 124
    ana.log()
    ana.refresh()

    for result in no_optimize_results:
        if result.origin_id == result.first_label or result.origin_id == result.second_label:
            ana.success_increment()
        else:
            ana.fail_increment()
    ana.log()
    ana.refresh()

    for result in remove_human_results:
        if result.origin_id == result.first_label or result.origin_id == result.second_label:
            ana.success_increment()
        else:
            ana.fail_increment()
    ana.log()
    ana.refresh()

    for result in rough_optimize_results:
        if result.origin_id == result.first_label or result.origin_id == result.second_label:
            ana.success_increment()
        else:
            ana.fail_increment()
    ana.log()
    ana.refresh()

    for result in full_optimize_results:
        if result.origin_id == result.first_label or result.origin_id == result.second_label:
            ana.success_increment()
        else:
            ana.fail_increment()
    ana.log()
    ana.refresh()

def top_3_analysis():
    ana = Ana()
    for result in no_human_results:
        if result.origin_id == result.first_label or result.origin_id == result.second_label or result.origin_id == result.third_label:
            ana.success_increment()
        else:
            ana.fail_increment()
    ana.suc -= 124
    ana.cnt -= 124
    ana.log()
    ana.refresh()

    for result in no_optimize_results:
        if result.origin_id == result.first_label or result.origin_id == result.second_label or result.origin_id == result.third_label:
            ana.success_increment()
        else:
            ana.fail_increment()
    ana.log()
    ana.refresh()

    for result in remove_human_results:
        if result.origin_id == result.first_label or result.origin_id == result.second_label or result.origin_id == result.third_label:
            ana.success_increment()
        else:
            ana.fail_increment()
    ana.log()
    ana.refresh()

    for result in rough_optimize_results:
        if result.origin_id == result.first_label or result.origin_id == result.second_label or result.origin_id == result.third_label:
            ana.success_increment()
        else:
            ana.fail_increment()
    ana.log()
    ana.refresh()

    for result in full_optimize_results:
        if result.origin_id == result.first_label or result.origin_id == result.second_label or result.origin_id == result.third_label:
            ana.success_increment()
        else:
            ana.fail_increment()
    ana.log()
    ana.refresh()

if __name__ == '__main__':
    search_rate_count()
    # time_analysis()
    # dis_analysis()
    # top_2_analysis()
    # top_3_analysis()