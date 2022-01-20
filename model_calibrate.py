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
from facenet_infer import facenet_process, facenet_preprocess

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
    return feature_map1, feature_map2, light_graph

if __name__ == '__main__':
    out_yolov3_pb = 'output_lt/yolov3_tiny_lgf_end.pb'
    out_facenet_pb = 'output_lt/facenet_lgf_end.pb'
    graph_path = './yolov3_tiny_savedmodel'
    light_graph_pb = './lt_graph_pb/yolov3_tiny_lgf.pb' 

    facenet_savedmodel = './facenet_savedmodel'
    facenet_lt_graph_path = './lt_graph_pb/facenet_lgf.pb'
    imported_graph = lt.import_graph(graph_path, config)
    facenet_graph = lt.import_graph(facenet_savedmodel, config)
    
    all_image_path = '/home/ubuntu/huangyang/t4_spacework/lt_code_sdk/solution/object_detection/YOLOv3_tiny_TensorFlow/data/widerface_voc/val/JPEGImages'
    all_images_list = os.listdir(all_image_path)
    count = 0
    all_image_number = len(all_images_list)
    for j in all_images_list:
        # image_path = "../data/demo_data/messi.jpg"
        image_path = os.path.join(all_image_path, j)
        anchor_path = 'data/yolov3_tiny_widerface_anchors.txt'
        class_name_path = 'data/widerface.names'
        input_image = image_path
        img_input, anchors, classes, num_class, height_ori, width_ori, img_ori, color_table = preprocess_input(anchor_path, class_name_path, input_image)
        np.save("test.npy", img_input)
        npy_file_name = 'test.npy'
        
        calibration_data = get_calibration_data(calibration_data_file_path=npy_file_name, input_edge_name=input_name)
        light_graph = lt.transform_graph(imported_graph, config, calibration_data=calibration_data)
        outs = infer_process(light_graph=light_graph, calibration_data=calibration_data, config=config)
        feature_map1 = outs[0][0]
        feature_map2 = outs[1][0]
        light_graph_yolov3 = outs[2]
        # print("feature_map1 type = ", type(feature_map1))
        # print("feature_map1 shape = ", feature_map1.shape)
        # print("feature_map2 shape = ", feature_map2.shape)
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

        # print("box coords:")
        # print(boxes_)
        # print('*' * 30)
        # print("scores:")
        # print(scores_)
        # print('*' * 30)
        # print("labels:")
        # print(labels_)

        for i in range(len(boxes_)):
            x0, y0, x1, y1 = boxes_[i]
            face_crop = img_ori[int(y0):int(y1), int(x0):int(x1)]
            face_crop = cv2.resize(face_crop, image_shape)
            cv2.imwrite('face'+str(i)+'.jpg', face_crop)
            image_face_path = 'face'+str(i)+'.jpg'
            npy_image = facenet_preprocess(image_face_path)
            np.save('face'+str(i)+'.npy', npy_image)
            npy_file_name = 'face'+str(i)+'.npy'
            calibration_data = get_calibration_data(calibration_data_file_path=npy_file_name, input_edge_name="image_batch")
            light_graph_facenet = lt.transform_graph(facenet_graph, config, calibration_data=calibration_data)
            features, light_graph_facenet = facenet_process(light_graph_facenet, image_face_path, config, image_shape)
            plot_one_box(img_ori, [x0, y0, x1, y1], label=classes[labels_[i]], color=color_table[labels_[i]])
        cv2.imwrite('detection_result.jpg', img_ori)
        count += 1
        print("************* finished ************ : {}/{}".format(count, all_image_number))    
        if count % 50 == 0:
            lt.export_graph(light_graph_yolov3, out_yolov3_pb, config)
            lt.export_graph(light_graph_facenet, out_facenet_pb, config)
