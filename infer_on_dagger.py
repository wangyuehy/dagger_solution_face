from numpy.lib.arraysetops import isin
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import lt_sdk as lt
from lt_sdk.proto import hardware_configs_pb2, graph_types_pb2
from lt_sdk.graph.transform_graph import utils as lt_utils

def my_test_TBDR():
    # config = lt.get_default_config()
    config = lt.get_default_config(hw_cfg=hardware_configs_pb2.DAGGER,graph_type=graph_types_pb2.TFSavedModel)
    # config.sw_config.compiler_params.compiler_restrictions.no_odd_image_dims_conv2d=False
    print(config)
    print('-'*50)
    #srcpath = "/home/yhuang/lt_code_sdk/SDKExamples"
    graph_path = './model/nasnet-A_mobile_224'
    outpb = './output/nasnet-A_mobile_224_lgf.pb'
    outtrace = "./output/nasnet-A_mobile_224.trace"
    igraph = lt.import_graph(graph_path, config)
    lgraph = lt.transform_graph(igraph, config)
    lt.export_graph(lgraph, outpb, config)
    FUNC_SIM=True
    import time
    if FUNC_SIM:
        execstat = lt.run_performance_simulation(lgraph, config)
        from lt_sdk.visuals import sim_result_to_trace
        print(execstat)
        sim_result_to_trace.instruction_trace(outtrace, execstat, config.hw_specs, config.sim_params)
    elif not FUNC_SIM:
        batchsize = 1
        inputname = 'input_1'
        #   tensor = np.random.random((1,224,224,3))
        tensor = np.load('cov_valid.npy')
        print(tensor.shape)
        namedtensors = lt.data.named_tensor.NamedTensorSet([inputname],[tensor])
        inputbatch = lt.data.batch.batch_inputs(namedtensors, batchsize)
        # pdb.set_trace()
        for i in range(100):
            t1 = time.time()
            outputs = lt.run_functional_simulation(lgraph, inputbatch, config)
            t2 = time.time()
            print("times: ", (t2-t1)*1000000)
            #   print("outputs type = ", type(outputs))
            #   print("outputs = ", outputs)

def convert_to_lt_graph(input_saved_model_path, output_lt_graph_path):
    config = lt.get_default_config(hw_cfg=hardware_configs_pb2.DAGGER,graph_type=graph_types_pb2.TFSavedModel)
    input_graph = lt.import_graph(input_saved_model_path, config)
    trans_graph = lt.transform_graph(input_graph, config)
    lt.export_graph(trans_graph, output_lt_graph_path, config)
    return

def visual(lt_graph_path):
    port = 8032
    cmdstr = 'python lt_sdk/visuals/plot_lgf_graph.py --pb_graph_path ' + lt_graph_path + ' --port ' + str(port)
    print(cmdstr)


def infer_process(light_graph=None, calibration_data=None, config=None):
    outputs = lt.run_functional_simulation(light_graph, calibration_data, config)
    results = []
    for inf_out in outputs.batches:
        for named_ten in inf_out.results:
            if named_ten.edge_info.name.startswith("embeddings"):   #输出节点名0
                logits = tf.convert_to_tensor(lt_utils.tensor_pb_to_array(named_ten.data,np.float32))
                results.append(logits.numpy())
    # outs = tf.nn.softmax(results[0], axis=1)
    return results

def func_infererence(lt_graph_path, image_path):
    config = lt.get_default_config(hw_cfg=hardware_configs_pb2.DAGGER,graph_type=graph_types_pb2.TFSavedModel)
    graph = lt.import_graph(lt_graph_path, config)
    image_shape = (160, 160) # must same with graph input dimension

    try:
        raw_image = tf.io.read_file(image_path, 'r') # for tf 1.15
    except Exception as e:
        raw_image = tf.read_file(image_path, 'r') # for tf 1.7, there is no tf.io
    image_data = tf.image.decode_jpeg(raw_image, channels=3)
    image_data = tf.image.resize_image_with_crop_or_pad(image_data, image_shape[0], image_shape[1])
    image_data = tf.cast(image_data, dtype=tf.float32) # for tf 1.15, tf.image.per_image_standardization will not cast the input to float32 automatically, so we do the cast in advance
    image_data = tf.image.per_image_standardization(image_data)
    image_data = tf.expand_dims(image_data, 0)
    image_data = image_data.numpy()
    # with tf.Session() as sess:
    #     image_data = sess.run(image_data) # tensor to list
    # image_data = np.random.random((1, 160, 160, 3))
    batch_size = 1
    input_name = 'image_batch'
    named_tensor = lt.data.named_tensor.NamedTensorSet([input_name], [image_data])
    batch_input = lt.data.batch.batch_inputs(named_tensor, batch_size)
    # embed_res = lt.run_functional_simulation(graph, batch_input, config)
    embed_res = infer_process(graph, batch_input, config)
    print(f'embeddings:\n{embed_res}')
    return embed_res

    
if __name__ == '__main__':
    # saved_model_path = 'data/facenet_saved_model/20180402-114759'
    # lt_graph_path = 'data/facenet_20180402-114759_lt.pb'
    # saved_model_path = 'data/facenet_saved_model/20220110-144041'
    # lt_graph_path = 'data/facenet_20220110-144041_lt.pb'
    # saved_model_path = 'data/facenet_saved_model/20220111-154935'
    # lt_graph_path = 'data/facenet_20220111-154935_lt.pb'
    # convert_to_lt_graph(saved_model_path, lt_graph_path)
    # visual(lt_graph_path)
    # image_path = 'data/datasets/lfw/lfw-112X96/Abel_Pacheco/Abel_Pacheco_0001.jpg'
    # embed_res = func_infererence(lt_graph_path, image_path)
    # np.testing.assert_almost_equal(embed_res[0][0:1, :27], 
    # [[
    #     6.59179688e-02,  1.48925781e-02, -4.85839844e-02,
    #     -8.25195312e-02, -6.73828125e-02, -9.22851562e-02,
    #     -2.12402344e-02,  8.69140625e-02,  1.54418945e-02,
    #      4.37011719e-02, -8.30078125e-02, -2.68554688e-02,
    #     -4.56542969e-02, -3.73535156e-02,  2.17285156e-02,
    #     -1.01074219e-01, -7.22656250e-02, -1.50756836e-02,
    #     -1.45874023e-02, -2.96020508e-03, -3.95507812e-02,
    #      9.22851562e-02, -3.03955078e-02, -6.17675781e-02,
    #     -1.48315430e-02, -2.79541016e-02, -3.88183594e-02,
    # ]], 
    # decimal=5)




    ###############
    saved_model_path = 'data/facenet_saved_model/20220113-100920'
    lt_graph_path = 'data/facenet_20220113-100920_lt.pb'
    convert_to_lt_graph(saved_model_path, lt_graph_path)
    visual(lt_graph_path)
    image_path = 'data/datasets/lfw/lfw-112X96/Abel_Pacheco/Abel_Pacheco_0001.jpg'
    embed_res = func_infererence(lt_graph_path, image_path)
