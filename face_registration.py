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
import csv
import codecs
from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms, cpu_nms
from utils.plot_utils import get_color_table, plot_one_box
from model.yolov3 import yolov3
from model.yolov3_tiny import yolov3_tiny
from facenet_infer import facenet_process
from face_landmark_lt_infer import pfld_infer_process, read_img_input, Coordinate_landmarks_map, align_face
from common import parse_csv_get_feas_list, face_recognition
from common import numpy_extractor_info

config = lt.get_default_config(hw_cfg=hardware_configs_pb2.DAGGER,graph_type=graph_types_pb2.TFSavedModel)

batch_size = 1
input_name = "input_data"  # 模型输入名
input_h_w = 224
image_shape = (160, 160)

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

def infer_process(light_graph, input_data, input_name, config):
    named_tensor = lt.data.named_tensor.NamedTensorSet([input_name], [input_data])
    batch_input = lt.data.batch.batch_inputs(named_tensor, batch_size)
    outputs = lt.run_functional_simulation(light_graph, batch_input, config)
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
    face_regist_path  = './facedata_test'
    yolov3_lt_graph_path = './lt_graph_pb/yolov3_tiny_lgf_end.pb' 
    facenet_lt_graph_path = './lt_graph_pb/facenet_lgf_end.pb'
    pfld_lt_graph_path = './lt_graph_pb/pfld_lgf_end2.pb'

    yolov3_lt_graph = lt.import_graph(yolov3_lt_graph_path, config)
    facenet_lt_graph = lt.import_graph(facenet_lt_graph_path, config)
    pfld_lt_graph = lt.import_graph(pfld_lt_graph_path, config)

    anchor_path = 'data/yolov3_tiny_widerface_anchors.txt'
    class_name_path = 'data/widerface.names'
    face_features_list = []
    all_name_list = os.listdir(face_regist_path)
    
    for name in all_name_list:
        all_images_list = os.listdir(os.path.join(face_regist_path, name))
        for j in all_images_list:
            os.path.join(face_regist_path, name)
            image_path = os.path.join(os.path.join(face_regist_path, name), j)
            input_image = image_path
            img_input, anchors, classes, num_class, height_ori, width_ori, img_ori, color_table = preprocess_input(anchor_path, class_name_path, input_image)
            
            outs= infer_process(yolov3_lt_graph, img_input, input_name, config)
            feature_map1 = outs[0][0]
            feature_map2 = outs[1][0]
            print("feature_map1 type = ", type(feature_map1))
            print("feature_map1 shape = ", feature_map1.shape)
            print("feature_map2 shape = ", feature_map2.shape)
            pred_feature_maps = tuple([feature_map1, feature_map2])
            yolo_model = yolov3_tiny(num_class, anchors)
            pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)
            pred_scores = pred_confs * pred_probs
            boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, num_class, max_boxes=200, score_thresh=0.6, iou_thresh=0.6)

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
                # x0, y0, x1, y1 = x0-30, y0-30, x1+30, y1+30
                boxes_list = [x0, y0, x1, y1]
                boxes_npy = np.array(boxes_list)
                face_crop = img_ori[int(y0):int(y1), int(x0):int(x1)]
                face_crop = cv2.resize(face_crop, image_shape)
                ########### face landmark detection start ###########
                input_pfld = read_img_input(face_crop, 112, 112)
                pfld_outs = pfld_infer_process(pfld_lt_graph, input_pfld, config)
                landmarks = pfld_outs[0][0]
                theatas = pfld_outs[1][0]
                ########### face landmark detection end #############
                rotated_img, eye_center, angle, rotated_landmarks = align_face(face_crop, landmarks, img_ori, boxes_npy)  # face align
                # cv2.imwrite("rotated_img.jpg", rotated_img)
                rotated_input = cv2.resize(rotated_img, image_shape)
                features, _ = facenet_process(facenet_lt_graph, rotated_input, config, image_shape)
                
                face_name = name
                fea_map = numpy_extractor_info(features[0])
                face_features_list.append(tuple([face_name, fea_map]))
                plot_one_box(img_ori, [x0, y0, x1, y1], label=classes[labels_[i]], color=color_table[labels_[i]])
            cv2.imwrite('detection_result.jpg', img_ori)
            # print("face_features_list = ", face_features_list)

    f = codecs.open('face_features_lib.csv','w','utf-8')
    writer = csv.writer(f)
    for i in face_features_list:
        writer.writerow(i)
    f.close()


    


