# -*- encoding: utf-8 -*-

"""
@File        :  model_converter.py    
@Contact     :  1121584497@qq.com
@Modify Time :  2020/4/27 15:46      
@Author      :  Tomax  
@Version     :  1.0
@Desciption  :  None
"""

# import lib
from inpaint.inpaint_model import InpaintCAModel
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import cv2
import numpy as np
from keras.applications.resnet50 import ResNet50

def export_inpaint_model(input_checkpoint):
    sess_config = tf.ConfigProto()
    sess = tf.Session(config=sess_config)
    inpaint_model = InpaintCAModel()
    input_image_ph = tf.placeholder(
        tf.float32, shape=(1, 224, 224 * 2, 3), name='input_image')
    stage1_output, stage2_output = inpaint_model.build_server_graph_rough(input_image_ph)
    stage1_output = (stage1_output + 1.) * 127.5
    stage2_output = (stage2_output + 1.) * 127.5
    stage1_output = tf.reverse(stage1_output, [-1])
    stage2_output = tf.reverse(stage2_output, [-1])
    stage1_output = tf.saturate_cast(stage1_output, tf.uint8, name='stage_1_output')
    stage2_output = tf.saturate_cast(stage2_output, tf.uint8, name='stage_2_output')
    graph = tf.get_default_graph()
    [print(n.name) for n in tf.get_default_graph().as_graph_def().node]

    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'inpaint_net')
    assign_ops = []
    for var in vars_list:
        vname = var.name
        from_name = vname
        var_value = tf.contrib.framework.load_variable(
            input_checkpoint, from_name)
        assign_ops.append(tf.assign(var, var_value))
    sess.run(assign_ops)

    tf.train.write_graph(sess.graph_def, "E:/ml/vilbs/models/deepfill/pb/", "deepfill.pb", as_text=False)
    freeze_graph.freeze_graph("E:/ml/vilbs/models/deepfill/pb/deepfill.pb", '', True, 'E:/ml/vilbs/models/deepfill/snap-0', 'stage_1_output, stage_2_output','save/restore_all', 'save/Const:0', 'E:/ml/vilbs/models/deepfill/pb/deepfill_.pb', False, "")
    # writer = tf.summary.FileWriter("E://ml/tb_tmp/graph", sess.graph)
    # writer.flush()

def use_tflite(input_path, image_path, mask_path):
    # prepare
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)
    image = cv2.resize(image, (224, 224))
    mask = cv2.resize(mask, (224, 224))
    h, w, _ = image.shape
    grid = 4
    image = image[:h // grid * grid, :w // grid * grid, :]
    mask = mask[:h // grid * grid, :w // grid * grid, :]
    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)

    # interpret
    interpreter = tf.lite.Interpreter(input_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    result = np.squeeze(output_data)
    print(np.shape(result))
    cv2.imshow('image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def convert_model_from_session(f_path, t_path):
    sess_config = tf.ConfigProto()
    sess = tf.Session(config=sess_config)
    inpaint_model = InpaintCAModel()
    input_image_ph = tf.placeholder(
        tf.float32, shape=(1, 224, 224 * 2, 3), name='input_image')
    stage1_output, stage2_output = inpaint_model.build_server_graph_rough(input_image_ph)
    stage1_output = (stage1_output + 1.) * 127.5
    stage2_output = (stage2_output + 1.) * 127.5
    stage1_output = tf.reverse(stage1_output, [-1])
    stage2_output = tf.reverse(stage2_output, [-1])
    stage1_output = tf.saturate_cast(stage1_output, tf.uint8, name='stage_1_output')
    stage2_output = tf.saturate_cast(stage2_output, tf.uint8, name='stage_2_output')
    graph = tf.get_default_graph()
    [print(n.name) for n in tf.get_default_graph().as_graph_def().node]

    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'inpaint_net')
    assign_ops = []
    for var in vars_list:
        vname = var.name
        from_name = vname
        var_value = tf.contrib.framework.load_variable(
            f_path, from_name)
        assign_ops.append(tf.assign(var, var_value))
    sess.run(assign_ops)

    converter = tf.lite.TFLiteConverter.from_session(sess, [input_image_ph], [stage1_output, stage2_output])

    converter.allow_custom_ops = True
    tflite_model = converter.convert()
    open(t_path, "wb").write(tflite_model)

def convertResnet(resnet_path, outpath):
    sess_config = tf.ConfigProto()
    sess = tf.Session(config=sess_config)
    feature_extractor = ResNet50(weights=resnet_path, include_top=False)
    [print(n.name) for n in tf.get_default_graph().as_graph_def().node]

    converter = tf.lite.TFLiteConverter.from_keras_model_file(sess, resnet_path, input_arrays=["input_1"], output_arrays=[])
    converter = tf.lite.TFLiteConverter.from_saved_model(sess, None, None)
    converter.convert()

# def freeze_graph(input_checkpoint, output_graph):
#     output_node_names = 'in`paint_net'
#     saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
#     graph = tf.get_default_graph()  # 获得默认的图
#     input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图
#     with tf.Session() as sess:
#         saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
#         output_graph_def = tf.graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
#             sess=sess,
#             input_graph_def=input_graph_def,  # 等于:sess.graph_def
#             output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

        # with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
        #     f.write(output_graph_def.SerializeToString())  # 序列化输出
        #     # print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点

def convert_tflite(pb_path, tflite_path):
    convert = tf.lite.TFLiteConverter.from_frozen_graph(pb_path, input_arrays=["input_image"], output_arrays=["stage_1_output", "stage_2_output"])

    convert.post_training_quantize = True
    tflite_model = convert.convert()
    open(tflite_path, "wb").write(tflite_model)


if __name__ == '__main__':
    # freeze_graph('E:/ml/vilbs/models/deepfill/snap-0', 'E:/ml/vilbs/models/deepfill/pb/frozen.pb')
    # export_inpaint_model('E:/ml/vilbs/models/deepfill/snap-0')
    # convert_tflite('E:\\ml\\vilbs\\models\\deepfill\\pb\\deepfill_.pb', 'E:\\ml\\vilbs\\models\\deepfill\\tflite\\deepfill.tflite')
    # convert_model_from_session('E:/ml/vilbs/models/deepfill/snap-0', 'E:\\ml\\vilbs\\models\\deepfill\\tflite\\deepfill.tflite')
    # print(str(b"'toco_from_protos' \xb2\xbb\xca\xc7\xc4\xda\xb2\xbf\xbb\xf2\xcd\xe2\xb2\xbf\xc3\xfc\xc1\xee\xa3\xac\xd2\xb2\xb2\xbb\xca\xc7\xbf\xc9\xd4\xcb\xd0\xd0\xb5\xc4\xb3\xcc\xd0\xf2\r\n\xbb\xf2\xc5\xfa\xb4\xa6\xc0\xed\xce\xc4\xbc\xfe\xa1\xa3\r\n", encoding='gbk'))
    convertResnet('E:/ml/vilbs/models/resnet/resnet50un.h5', '')
    # use_tflite('E:/ml/vilbs/models/deepfill/tflite/deepfill.tflite', 'E:/ml/test-data/gi/1_input.png', 'E:/ml/test-data/gi/1_mask.png')

"""
toco --graph_def_file=E:/ml/vilbs/models/deepfill/pb/deepfill_.pb --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE --output_file=E:/ml/vilbs/models/deepfill/tflite/deepfill.tflite --inference_type=FLOAT --input_type=FLOAT --input_arrays=input_image --output_arrays=stage_1_output,stage_2_output --input_shapes=1,224,448,3
toco --graph_def_file=/home/ubuntu/test/vilbs/models/deepfill/pb/deepfill_.pb --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE --output_file=/home/ubuntu/test/vilbs/models/deepfill/tflite/deepfill.tflite --inference_type=FLOAT --input_type=FLOAT --input_arrays=input_image --output_arrays=stage_1_output,stage_2_output --input_shapes=1,224,448,3

"""
