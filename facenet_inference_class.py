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

class Facenet_Inference(object):
    def __init__(self, input_name, lt_graph_path, width, height, batch_size, config):
        self.input_name = input_name
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.config = config
        self.lt_graph = self.init_lt_model(lt_graph_path)

    def init_lt_model(self, lt_model_path):
        lt_graph = lt.import_graph(lt_model_path, self.config)
        return lt_graph

    def facenet_preprocess(self, image_data):
        image_data = tf.image.resize_images(image_data, (self.width, self.height))
        image_data = tf.cast(image_data, dtype=tf.float32) # for tf 1.15, tf.image.per_image_standardization will not cast the input to float32 automatically, so we do the cast in advance
        image_data = tf.image.per_image_standardization(image_data)
        image_data = tf.expand_dims(image_data, 0)
        image_data = image_data.numpy()
        return image_data

    def inference(self, x):
        outputs = lt.run_functional_simulation(self.lt_graph, x, self.config)
        results = []
        for inf_out in outputs.batches:
            for named_ten in inf_out.results:
                if named_ten.edge_info.name.startswith("embeddings"):   #输出节点名0
                    logits = tf.convert_to_tensor(lt_utils.tensor_pb_to_array(named_ten.data,np.float32))
                    results.append(logits.numpy())
        return results

    def run(self, x):
        image_data = self.facenet_preprocess(x)
        print("image_data shape = ", image_data.shape)
        named_tensor = lt.data.named_tensor.NamedTensorSet([self.input_name], [image_data])
        batch_input = lt.data.batch.batch_inputs(named_tensor, self.batch_size)
        embed_res = self.inference(batch_input)
        # print(f'embeddings:\n{embed_res}')
        return embed_res