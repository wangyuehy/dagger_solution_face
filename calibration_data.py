import argparse
import json
import logging
import os
import shutil
import tarfile

import numpy as np
# from azure.storage import blob
from lt_sdk.data import batch, named_tensor
from lt_sdk.graph import full_graph_pipeline
from lt_sdk.graph.transform_graph import utils as lt_utils
from lt_sdk.proto import dtypes_pb2
from lt_sdk.visuals import sim_result_to_trace


def load_np_array(path, preprocess_fn=None):
    array = np.load(path)
    if preprocess_fn is not None:
        array = preprocess_fn(array)

    return array


def get_input_data(filenames,
                   batch_size,
                   preprocess_fn=None,
                   num_samples=None,
                   dtype=dtypes_pb2.DT_FLOAT):
    names = []
    tensors = []
    for path, name in filenames.items():
        print("path = ", path)
        print("name = ", name)
        names.append(name)
        tensors.append(load_np_array(path, preprocess_fn=preprocess_fn))

    named_tensors = named_tensor.NamedTensorSet(names, tensors, dtype=dtype)

    if num_samples is not None:

        def get_subset(array):
            return array[:num_samples]

        named_tensors.apply_all(get_subset)

        if num_samples < batch_size:
            logging.warning(
                "Num samples {} is less than batch size {}. Consider reducing batch size"
                .format(num_samples,
                        batch_size))

    batched_inputs = batch.batch_inputs(named_tensors, batch_size=batch_size)

    return batched_inputs

batch_size = 1
preprocess_input_data_fn = None
num_samples = 1
def get_calibration_data(calibration_data_file_path=None, input_edge_name=None):
    
    # list_npy = os.listdir(calibration_data_file_path)
    # calib_dict = {}
    # for i in range(len(list_npy)):
    #     key = os.path.join(calibration_data_file_path, list_npy[i])
    #     value = input_edge_name
    #     temp_dict = {key : value}
    #     calib_dict.update(temp_dict)
    key = calibration_data_file_path
    value = input_edge_name
    calib_dict = {key : value}
    return get_input_data(calib_dict,
                            batch_size,
                            preprocess_fn=preprocess_input_data_fn,
                            num_samples=num_samples)

def get_calibration_data_batch(calibration_data_batch=None):
    
    # list_npy = os.listdir(calibration_data_file_path)
    # calib_dict = {}
    # for i in range(len(list_npy)):
    #     key = os.path.join(calibration_data_file_path, list_npy[i])
    #     value = input_edge_name
    #     temp_dict = {key : value}
    #     calib_dict.update(temp_dict)
    calib_dict = calibration_data_batch
    return get_input_data(calib_dict,
                            batch_size=2,
                            preprocess_fn=preprocess_input_data_fn,
                            num_samples=4952)
if __name__ == '__main__':
    calibration_data_file_path="/home/yhuang/tensorRT_work/ssd_resnet18_tf2trt/datanpy/000001.npy"
    # list_npy = os.listdir(calibration_data_file)
    # calib_dict = {}
    # for i in range(len(list_npy)):
    #     key = os.path.join(calibration_data_file, list_npy[i])
    #     value = input_edge_name
    #     temp_dict = {key : value}
    #     calib_dict.update(temp_dict)
    
    res =  get_calibration_data(calibration_data_file_path=calibration_data_file_path)
    # print(res)
