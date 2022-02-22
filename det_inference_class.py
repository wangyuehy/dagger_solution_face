from __future__ import division, print_function
import os
import math
import random
import numpy as np
import shutil
import cv2
import tensorflow as tf
tf.enable_eager_execution()
slim = tf.contrib.slim

import lt_sdk as lt
import pdb
from lt_sdk.proto import hardware_configs_pb2, graph_types_pb2
from lt_sdk.graph.transform_graph import utils as lt_utils
from utils.misc_utils import parse_anchors, read_class_names
from utils.plot_utils import get_color_table, plot_one_box
from utils.nms_utils import gpu_nms, cpu_nms
from model.yolov3_tiny import yolov3_tiny

class Detection_Inference(object):
    def __init__(self, input_name,
        input_h_w, 
        classes, 
        batch_size,
        anchors,
        lt_graph_path,
        config):
        self.input_name = input_name
        self.input_h_w = input_h_w
        self.classes = classes
        self.num_class = len(classes)
        self.batch_size = batch_size
        self.anchors = anchors
        self.config = config
        self.lt_graph = self.init_lt_model(lt_graph_path)

    def init_lt_model(self, lt_model_path):
        lt_graph = lt.import_graph(lt_model_path, self.config)
        return lt_graph

    def preprocess_input(self, input_image):
        img_ori = cv2.imread(input_image)
        height_ori, width_ori = img_ori.shape[:2]
        img = cv2.resize(img_ori, tuple([self.input_h_w, self.input_h_w]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.
        return img, height_ori, width_ori, img_ori

    def det_lgt_infer_process(self, x):
        img, height_ori, width_ori, img_ori = self.preprocess_input(x)
        named_tensor = lt.data.named_tensor.NamedTensorSet([self.input_name], [img])
        batch_input = lt.data.batch.batch_inputs(named_tensor, self.batch_size)
        outputs = lt.run_functional_simulation(self.lt_graph, batch_input, self.config)
        feature_map1 = []
        feature_map2 = []
        for inf_out in outputs.batches:
            for named_ten in inf_out.results:
                if named_ten.edge_info.name.startswith("yolov3_tiny/yolov3_tiny_head/feature_map_1"):   #输出节点名0
                    feat_map1 = tf.convert_to_tensor(lt_utils.tensor_pb_to_array(named_ten.data,np.float32))
                    feature_map1.append(feat_map1)
                if named_ten.edge_info.name.startswith("yolov3_tiny/yolov3_tiny_head/feature_map_2"):   #输出节点名1
                    feat_map2 = tf.convert_to_tensor(lt_utils.tensor_pb_to_array(named_ten.data,np.float32))
                    feature_map2.append(feat_map2)
        return feature_map1, feature_map2, height_ori, width_ori, img_ori

    def run(self, x):
        lgt_res = self.det_lgt_infer_process(x)
        feature_map1 = lgt_res[0][0]
        feature_map2 = lgt_res[1][0]
        height_ori = lgt_res[2]
        width_ori = lgt_res[3]
        img_ori = lgt_res[4]
        
        # anchors = parse_anchors(self.anchor_path)
        # classes = read_class_names(self.class_name_path)
        # num_class = len(classes)
        # color_table = get_color_table(num_class)

        print("feature_map1 type = ", type(feature_map1))
        print("feature_map1 shape = ", feature_map1.shape)
        print("feature_map2 shape = ", feature_map2.shape)
        pred_feature_maps = tuple([feature_map1, feature_map2])
        yolo_model = yolov3_tiny(self.num_class, self.anchors)
        pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)
        pred_scores = pred_confs * pred_probs
        boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, self.num_class, max_boxes=200, score_thresh=0.4, iou_thresh=0.5)

        boxes_, scores_, labels_ = boxes.numpy(), scores.numpy(), labels.numpy()
        # rescale the coordinates to the original image
        boxes_[:, 0] *= (width_ori/float(self.input_h_w))
        boxes_[:, 2] *= (width_ori/float(self.input_h_w))
        boxes_[:, 1] *= (height_ori/float(self.input_h_w))
        boxes_[:, 3] *= (height_ori/float(self.input_h_w))

        print("box coords:")
        print(boxes_)
        print('*' * 30)
        print("scores:")
        print(scores_)
        print('*' * 30)
        print("labels:")
        print(labels_)
        return boxes_, scores_, labels_

    