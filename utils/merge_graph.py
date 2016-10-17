import tensorflow as tf
import os
import numpy as np
from tensorflow.python.platform import gfile
from tensorflow.python.platform import gfile
import inception_model as inception
from tensorflow.python.framework import graph_util


grap_path = "./classify_image_graph_def.pb"
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'

graph = tf.Graph()
with graph.as_default():
    print("Load Graph:" + grap_path)
    with gfile.FastGFile(grap_path,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def,name ="")
        image_data = graph.get_tensor_by_name(RESIZED_INPUT_TENSOR_NAME)
        logits, endpoints = inception.inference(images=image_data, num_classes=65, for_training=True, restore_logits=True)

with tf.Session(graph=graph) as sess:

    sess.run(tf.initialize_all_variables())

    vars_to_load = []
    for var in tf.all_variables():
        if var.name.startswith(("conv0", "conv1", "conv2", "conv3", "conv4", "mixed_35x35", "mixed_17x17", "mixed_8x8")):
            vars_to_load.append(var)
    restore_inception = tf.train.Saver(vars_to_load)

    restore_inception.restore(sess,'./data/model.ckpt-157585')
    for op in sess.graph.get_operations():
                print op.name
    output_graph_def = graph_util.convert_variables_to_constants(sess,sess.graph_def, ['inception_v3/logits/predictions'])
    with gfile.FastGFile('./inception3_merge_10_16.pb', 'wb') as f:
            f.write(output_graph_def.SerializeToString())

    '''
    print("map variables")

    #print sess.graph.get_operation_by_name('DecodeJpeg')
    #print sess.graph.get_operation_by_name('final_result')
    #print sess.graph.get_variable_by_name('DecodeJpeg:0')
    #print sess.graph.get_variable_by_name('final_result:0')
    #print tf.all_variables()
    #print (sess.graph.get_operations())
    #summary_writer = tf.train.SummaryWriter('./work/logs', graph=sess.graph)
    #test = tf.get_default_graph().get_operation_by_name("19_fc")
    #print test

    #tensorboard --logdir=./work/logs
    '''
