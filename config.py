# coding=utf-8
import os

INPUT_SIZE = 224
BATCH_SIZE = 1

MODEL_NAME = ['face_detection', 'face_feature_extr'] # 所有模型名

MODEL_INPUT_NAME = {'face_detection': 'input_data', 'face_feature_extr': 'input'}  # 每一个模型对应其模型的输入名

MODEL_OUTPUT_NAME = {'face_detection': ['yolov3_tiny/yolov3_tiny_head/feature_map_1', 'yolov3_tiny/yolov3_tiny_head/feature_map_2'],
                     'face_feature_extr': ['InceptionV2/Predictions/Reshape_1']
                     } # 每一个模型对应其模型的输出名

SAVED_MODEL_PATH = {'inceptionv1_224': 'model/inception_v1_224', 
                    'inceptionv2_224': 'model/inception_v2_224', 
                    'inceptionv3_299': 'model/inception_v3_299', 
                    'inceptionv4_299': 'model/inception_v4_299',
                    'mobilenet_v1_1.0_224': 'model/mobilenet_v1_1.0_224',
                    'mobilenet_v2_1.0_224': 'model/mobilenet_v2_1.0_224',
                    'mobilenet_v2_1.4_224': 'model/mobilenet_v2_1.4_224',
                    'resnet50_v1_224': 'model/resnet50_v1_224',
                    'resnet50_v2_224': 'model/resnet50_v2_224',
                    'resnet18_224': 'model/resnet18_224'
                    } # 每一个模型对应其保存的saved_model的路径

LT_TRANSFORM_GRAPH_NAME = {'inceptionv1_224': './output/inception_v1_224_lgf_end.pb',
                            'inceptionv2_224': './output/inception_v2_224_lgf_end.pb', 
                            'inceptionv3_299': './output/inception_v3_224_lgf_end.pb',
                            'inceptionv4_299': './output/inception_v4_224_lgf_end.pb',
                            'mobilenet_v1_1.0_224': 'output/mobilenet_v1_1.0_224_lgf_end.pb',
                            'mobilenet_v2_1.0_224': 'output/mobilenet_v2_1.0_224_lgf_end.pb', 
                            'mobilenet_v2_1.4_224': 'output/mobilenet_v2_1.4_224_lgf_end.pb',
                            'resnet50_v1_224': 'output/resnet50_v1_224_lgf_end.pb',
                            'resnet50_v2_224': 'output/resnet50_v2_224_lgf_end.pb', 
                            'resnet18_224': 'output/resnet18_224_lgf_end.pb'                   
                            }  # 每一个模型对应其在Dagger SDK转换后的模型名字和路径
