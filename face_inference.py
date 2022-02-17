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

import argparse
import cv2

from utils.plot_utils import plot_one_box
from det_infer import preprocess_input, det_lgt_infer_process, detection_inference
from facenet_infer import facenet_process
from pfld_landmark_infer import pfld_infer_process, read_img_input, Coordinate_landmarks_map, align_face
from common import parse_csv_get_feas_list, face_recognition

config = lt.get_default_config(hw_cfg=hardware_configs_pb2.DAGGER,graph_type=graph_types_pb2.TFSavedModel)

batch_size = 1

######## detection config #########
input_name = "input_data"  # 模型输入名
input_h_w = 224

######## facenet config ########
image_shape = (160, 160)

def init_LGT_graph(det_graph_path, fet_graph_path, ldmk_graph_path):
    yolov3_lt_graph = lt.import_graph(det_graph_path, config)
    facenet_lt_graph = lt.import_graph(fet_graph_path, config)
    pfld_lt_graph = lt.import_graph(ldmk_graph_path, config)
    return yolov3_lt_graph, facenet_lt_graph, pfld_lt_graph

def face_inference_all_process(det_graph_path, fet_graph_path, ldmk_graph_path, anchor_path, class_name_path, features_csv_path, input_image):
    yolov3_lt_graph, facenet_lt_graph, pfld_lt_graph = init_LGT_graph(det_graph_path, fet_graph_path, ldmk_graph_path)
    feas_map_list = parse_csv_get_feas_list(features_csv_path)
    img_input, anchors, classes, num_class, height_ori, width_ori, img_ori, color_table = preprocess_input(anchor_path, class_name_path, input_image, input_h_w)
    outs = det_lgt_infer_process(yolov3_lt_graph, img_input, input_name, config, batch_size)
    boxes_, scores_, labels_ = detection_inference(outs, num_class, anchors, height_ori, width_ori, input_h_w)
    face_features = []
    face_landmarks = []
    names = []
    for i in range(len(boxes_)):
        x0, y0, x1, y1 = boxes_[i]
        # x0, y0, x1, y1 = x0-30, y0-30, x1+30, y1+30
        # boxes_list = [x0, y0, x1, y1]
        # boxes_npy = np.array(boxes_list)
        face_crop = img_ori[int(y0):int(y1), int(x0):int(x1)]
        # cv2.imwrite("src_img.jpg", face_crop)
        ########### face landmark detection start ###########
        input_pfld = read_img_input(face_crop, 112, 112)
        pfld_outs = pfld_infer_process(pfld_lt_graph, input_pfld, config)
        landmarks = pfld_outs[0][0]
        theatas = pfld_outs[1][0]
        ########### face landmark detection end #############
        rotated_img, eye_center, angle, rotated_landmarks = align_face(face_crop, landmarks, img_ori, boxes_[i])  # face alignment
        # cv2.imwrite("rotated_img.jpg", rotated_img)
        rotated_input = cv2.resize(rotated_img, image_shape)
        features, _ = facenet_process(facenet_lt_graph, rotated_input, config, image_shape)
        fea_map = features[0].tolist()[0]
        face_features.append(fea_map)
        name = face_recognition(fea_map, feas_map_list)
        if name == None:
            name = 'Unkown'
        print("name = ", name)
        names.append(name)
        landmark_map_res = Coordinate_landmarks_map(landmarks, boxes_[i])
        face_landmarks.append(landmark_map_res)
        ############ draw landmark ######################
        for p in range(int(landmark_map_res.shape[1]/2)):
            cv2.circle(img_ori, (int(landmark_map_res[0][p*2+0]), int(landmark_map_res[0][p*2+1])), 2, (0, 0, 255), -1)
        ############ draw landmark end ##################
        plot_one_box(img_ori, [x0, y0, x1, y1], label=name+' '+classes[labels_[i]], color=color_table[labels_[i]])
    cv2.imwrite('detection_result.jpg', img_ori)
    return boxes_, scores_, labels_, face_features, face_landmarks, names

def test_face_recognition(input_image):
    features_csv_path = 'face_features_lib.csv'
    yolov3_lt_graph_path = './lt_graph_pb/yolov3_tiny_lgf_end.pb' 
    facenet_lt_graph_path = './lt_graph_pb/facenet_lgf_end.pb'
    pfld_lt_graph_path = './lt_graph_pb/pfld_lgf_end2.pb' 
    anchor_path = 'data/yolov3_tiny_widerface_anchors.txt'
    class_name_path = 'data/widerface.names'
    outs = face_inference_all_process(yolov3_lt_graph_path, facenet_lt_graph_path, pfld_lt_graph_path, anchor_path, class_name_path, features_csv_path, input_image)
    return outs
    
if __name__ == '__main__':
    input_image = "test.jpg"
    outs = test_face_recognition(input_image)
