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
config = lt.get_default_config(hw_cfg=hardware_configs_pb2.DAGGER,graph_type=graph_types_pb2.TFSavedModel)

batch_size = 1
input_name = "image_batch"  # 模型输入名
input_h_w = 160

def infer_process(light_graph=None, calibration_data=None, config=None):
    outputs = lt.run_functional_simulation(light_graph, calibration_data, config)
    results = []
    for inf_out in outputs.batches:
        for named_ten in inf_out.results:
            if named_ten.edge_info.name.startswith("embeddings"):   #输出节点名0
                logits = tf.convert_to_tensor(lt_utils.tensor_pb_to_array(named_ten.data,np.float32))
                results.append(logits.numpy())
    # outs = tf.nn.softmax(results[0], axis=1)
    return results, light_graph


def facenet_preprocess(image_data, image_shape):
    # try:
    #     raw_image = tf.io.read_file(image_data, 'r') # for tf 1.15
    # except Exception as e:
    #     raw_image = tf.read_file(image_data, 'r') # for tf 1.7, there is no tf.io
    # image_data = tf.image.decode_jpeg(raw_image, channels=3)
    # image_data = tf.image.resize_image_with_crop_or_pad(image_data, image_shape[0], image_shape[1])
    image_data = tf.image.resize_images(image_data, (image_shape[0], image_shape[1]))
    image_data = tf.cast(image_data, dtype=tf.float32) # for tf 1.15, tf.image.per_image_standardization will not cast the input to float32 automatically, so we do the cast in advance
    image_data = tf.image.per_image_standardization(image_data)
    image_data = tf.expand_dims(image_data, 0)
    image_data = image_data.numpy()
    return image_data

def facenet_process(graph, image, config, image_shape):
    image_data = facenet_preprocess(image, image_shape)
    print("image_data shape = ", image_data.shape)
    batch_size = 1
    input_name = 'image_batch'
    named_tensor = lt.data.named_tensor.NamedTensorSet([input_name], [image_data])
    batch_input = lt.data.batch.batch_inputs(named_tensor, batch_size)
    # embed_res = lt.run_functional_simulation(graph, batch_input, config)
    embed_res, light_graph = infer_process(graph, batch_input, config)
    # print(f'embeddings:\n{embed_res}')
    return embed_res, light_graph

if __name__ == '__main__':
    graph_path = './facenet_savedmodel'
    light_graph_pb = './lt_graph_pb/facenet_lgf.pb' 
    image_path = "../data/demo_data/messi.jpg"
    func_infererence(light_graph_pb, image_path)
 