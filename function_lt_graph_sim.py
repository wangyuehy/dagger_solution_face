from __future__ import division, print_function
import os
import math
import random
import numpy as np
import shutil
import tensorflow as tf
tf.enable_eager_execution()
slim = tf.contrib.slim

import lt_sdk as lt
import pdb
from lt_sdk.proto import hardware_configs_pb2, graph_types_pb2
from lt_sdk.graph.transform_graph import utils as lt_utils
from calibration_data import get_calibration_data

import argparse
import cv2
from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms, cpu_nms
from utils.plot_utils import get_color_table, plot_one_box
from model.yolov3 import yolov3
from model.yolov3_tiny import yolov3_tiny

config = lt.get_default_config(hw_cfg=hardware_configs_pb2.DAGGER,graph_type=graph_types_pb2.TFSavedModel)

batch_size = 1
input_name = "input_data"  # 模型输入名
input_h_w = 224
def preprocess_input(anchor_path, class_name_path, input_image):
    anchors = parse_anchors(anchor_path)
    classes = read_class_names(class_name_path)
    num_class = len(classes)

    color_table = get_color_table(num_class)
    img_ori = cv2.imread(input_image)
    height_ori, width_ori = img_ori.shape[:2]
    img = cv2.resize(img_ori, tuple([input_h_w, input_h_w]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, np.float32)
    img = img[np.newaxis, :] / 255.
    return img, anchors, classes, num_class, height_ori, width_ori, img_ori, color_table

def random_split_calibrate_test_data(srcpath, numfiles, calib_file):
    name_list = list(os.path.join(srcpath, name) for name in os.listdir(srcpath))
    random_name_list = list(random.sample(name_list, numfiles))
    with open(calib_file,'w') as f:    #设置文件对象
        for calib in random_name_list:
            f.write(calib.split('/')[-1])
            f.write('\n')                 #将字符串写入文件中
        f.close()

def infer_process(light_graph=None, calibration_data=None, config=config):
    outputs = lt.run_functional_simulation(light_graph, calibration_data, config)
    feature_map1 = []
    feature_map2 = []
    for inf_out in outputs.batches:
        for named_ten in inf_out.results:
            if named_ten.edge_info.name.startswith("yolov3_tiny/yolov3_tiny_head/feature_map_1"):   #输出节点名0
                feat_map1 = tf.convert_to_tensor(lt_utils.tensor_pb_to_array(named_ten.data,np.float32))
                feature_map1.append(feat_map1)
            if named_ten.edge_info.name.startswith("yolov3_tiny/yolov3_tiny_head/feature_map_2"):   #输出节点名0
                feat_map2 = tf.convert_to_tensor(lt_utils.tensor_pb_to_array(named_ten.data,np.float32))
                feature_map2.append(feat_map2)
    return feature_map1, feature_map2

if __name__ == '__main__':
    graph_path = './yolov3_tiny_savedmodel'
    light_graph_pb = './lt_graph_pb/yolov3_tiny_lgf.pb' 

    image_path = "../data/demo_data/messi.jpg"
    anchor_path = 'data/yolov3_tiny_widerface_anchors.txt'
    class_name_path = 'data/widerface.names'
    input_image = image_path
    img_input, anchors, classes, num_class, height_ori, width_ori, img_ori, color_table = preprocess_input(anchor_path, class_name_path, input_image)
    print("img_input type = ", type(img_input))
    print("img_input shape = ", img_input.shape)
    np.save("test.npy", img_input)

    npy_file_name = 'test.npy'
    
    imported_graph = lt.import_graph(graph_path, config)
    calibration_data = get_calibration_data(calibration_data_file_path=npy_file_name, input_edge_name=input_name)
    light_graph = lt.transform_graph(imported_graph, config, calibration_data=calibration_data)
    outs = infer_process(light_graph=light_graph, calibration_data=calibration_data, config=config)
    feature_map1 = outs[0][0]
    feature_map2 = outs[1][0]
    print("feature_map1 type = ", type(feature_map1))
    print("feature_map1 shape = ", feature_map1.shape)
    print("feature_map2 shape = ", feature_map2.shape)
    # feature_map1_tensor = tf.convert_to_tensor(feature_map1)
    # feature_map2_tensor = tf.convert_to_tensor(feature_map2)
    # print("feature_map1_tensor type = ", type(feature_map1_tensor))
    pred_feature_maps = tuple([feature_map1, feature_map2])
    yolo_model = yolov3_tiny(num_class, anchors)
    pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)
    pred_scores = pred_confs * pred_probs
    boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, num_class, max_boxes=200, score_thresh=0.4, iou_thresh=0.5)

    boxes_, scores_, labels_ = boxes.numpy(), scores.numpy(), labels.numpy()
    # rescale the coordinates to the original image
    boxes_[:, 0] *= (width_ori/float(input_h_w))
    boxes_[:, 2] *= (width_ori/float(input_h_w))
    boxes_[:, 1] *= (height_ori/float(input_h_w))
    boxes_[:, 3] *= (height_ori/float(input_h_w))

    print("box coords:")
    print(boxes_)
    print('*' * 30)
    print("scores:")
    print(scores_)
    print('*' * 30)
    print("labels:")
    print(labels_)

    for i in range(len(boxes_)):
        x0, y0, x1, y1 = boxes_[i]
        plot_one_box(img_ori, [x0, y0, x1, y1], label=classes[labels_[i]], color=color_table[labels_[i]])
    # cv2.imshow('Detection result', img_ori)
    cv2.imwrite('detection_result.jpg', img_ori)
    # cv2.waitKey(0)



