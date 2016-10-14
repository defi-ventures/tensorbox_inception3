#!/usr/bin/env python
import json
import datetime
import random
import time
import string
import argparse
import os
import sys
from scipy import misc
import tensorflow as tf
import numpy as np
try:
    from tensorflow.models.rnn import rnn_cell
except ImportError:
    rnn_cell = tf.nn.rnn_cell
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import gfile
import inception_model as inception
from tensorflow.python.framework import graph_util

random.seed(0)
np.random.seed(0)

from utils import train_utils, googlenet_load, tensor_net_load


def build_overfeat_forward(H, x, phase):
    input_mean = 128.
    input_std = 128.
    x -= input_mean

    k = H['num_classes']
    dense_layer_num_output = [k, 4]
    features_dim = 2048
    with tf.variable_scope('inception_v3') as scope:
      W = [
          tf.get_variable('softmax/weights_{}'.format(i), initializer=tf.truncated_normal([features_dim, dense_layer_num_output[i]], stddev=0.001))
          for i in range(2)
      ]

      B = [
          tf.get_variable('softmax/biases_{}'.format(i), initializer=tf.random_normal([dense_layer_num_output[i]], stddev=0.001))
          for i in range(2)
      ]

    logits, endpoints = inception.inference(images=x, num_classes=2, for_training=True, restore_logits=True)
    tf.get_variable_scope().reuse_variables()
    feature = endpoints['feature']
    mixed5b = endpoints['mixed_8x8x2048b'] # right before avgpool layer
    mixed5b = tf.reshape(mixed5b, [H['batch_size'] * H['grid_width'] * H['grid_height'], features_dim])
    grid_size = H['grid_width'] * H['grid_height']
    pred_logits = tf.reshape(tf.nn.xw_plus_b(mixed5b, W[0], B[0],
                                             name=phase+'/logits_0'),
                             [H['batch_size'] * grid_size, H['num_classes']])
    pred_confidences = tf.nn.softmax(pred_logits)
    pred_boxes = tf.reshape(tf.nn.xw_plus_b(mixed5b, W[1], B[1],
                                            name=phase+'/logits_1'),
                           [H['batch_size'] * grid_size, 1, 4]) * 100
    #pred_boxesout = tf.reshape( pred_boxes , [H['batch_size'] * grid_size, 4])
    out_logit = tf.reshape(pred_confidences,[1,H['batch_size'] * grid_size* H['num_classes']])
    out_boxes = tf.reshape(pred_boxes,[1,H['batch_size'] * grid_size*4])
    out_feature = tf.reshape(feature,[1,2048])

    output = tf.concat(1,[out_logit,out_boxes,out_feature],name = 'output')
    return pred_boxes, pred_logits, pred_confidences

def build_forward_backward(H, x, inception3, phase, boxes, flags):
    '''
    Call build_forward() and then setup the loss functions
    '''

    grid_size = H['grid_width'] * H['grid_height']
    outer_size = grid_size * H['batch_size']
    reuse = {'train': None, 'test': True}[phase]

        #pred_boxes, pred_logits, pred_confidences = build_forward(H, x,inception3, phase, reuse)
    pred_boxes, pred_logits, pred_confidences= build_overfeat_forward (H,x,phase)
    with tf.variable_scope('decoder', reuse={'train': None, 'test': True}[phase]):
        outer_boxes = tf.reshape(boxes, [outer_size, H['rnn_len'], 4])
        outer_flags = tf.cast(tf.reshape(flags, [outer_size, H['rnn_len']]), 'int32')

        classes = tf.reshape(flags, (outer_size, 1))
        perm_truth = tf.reshape(outer_boxes, (outer_size, 1, 4))
        pred_mask = tf.reshape(tf.cast(tf.greater(classes, 0), 'float32'), (outer_size, 1, 1))
        true_classes = tf.reshape(tf.cast(tf.greater(classes, 0), 'int64'),
                                  [outer_size * H['rnn_len']])
        real_classes = tf.reshape(tf.cast(classes, 'int64'),
                                  [outer_size * H['rnn_len']])
        pred_logit_r = tf.reshape(pred_logits,
                                  [outer_size * H['rnn_len'], H['num_classes']])
        confidences_loss = (tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(pred_logit_r, true_classes))
            ) / outer_size * H['solver']['head_weights'][0]
        residual = tf.reshape(perm_truth - pred_boxes * pred_mask,
                              [outer_size, H['rnn_len'], 4])
        boxes_loss = tf.reduce_sum(tf.abs(residual)) / outer_size * H['solver']['head_weights'][1]

        loss = confidences_loss + boxes_loss

    return pred_boxes, pred_confidences, loss, confidences_loss, boxes_loss

def build(H, q):
    '''
    Build full model for training, including forward / backward passes,
    optimizers, and summary statistics.
    '''
    arch = H
    solver = H["solver"]

    os.environ['CUDA_VISIBLE_DEVICES'] = str(solver.get('gpu', ''))

    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    gpu_options = tf.GPUOptions()
    config = tf.ConfigProto(gpu_options=gpu_options)

    encoder_net = tensor_net_load.init(H, config)

    learning_rate = tf.placeholder(tf.float32)

    opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                        decay=0.9, epsilon=solver['epsilon'])

    loss, accuracy, confidences_loss, boxes_loss = {}, {}, {}, {}
    for phase in ['train', 'test']:
        # generate predictions and losses from forward pass
        x, confidences, boxes = q[phase].dequeue_many(arch['batch_size'])
        flags = tf.argmax(confidences, 3)

        grid_size = H['grid_width'] * H['grid_height']

        (pred_boxes, pred_confidences,
         loss[phase], confidences_loss[phase],
         boxes_loss[phase]) = build_forward_backward(H, x, encoder_net, phase, boxes, flags)
        pred_confidences_r = tf.reshape(pred_confidences, [H['batch_size'], grid_size, H['rnn_len'], arch['num_classes']])
        pred_boxes_r = tf.reshape(pred_boxes, [H['batch_size'], grid_size, H['rnn_len'], 4])


        # Set up summary operations for tensorboard
        a = tf.equal(tf.argmax(confidences[:, :, 0, :], 2), tf.argmax(pred_confidences_r[:, :, 0, :], 2))
        accuracy[phase] = tf.reduce_mean(tf.cast(a, 'float32'), name=phase+'/accuracy')

        if phase == 'train':
            global_step = tf.Variable(0, trainable=False)

            tvars = tf.trainable_variables()
            if H['clip_norm'] <= 0:
                grads = tf.gradients(loss['train'], tvars)
            else:
                grads, norm = tf.clip_by_global_norm(tf.gradients(loss['train'], tvars), H['clip_norm'])
            train_op = opt.apply_gradients(zip(grads, tvars), global_step=global_step)
        elif phase == 'test':
            moving_avg = tf.train.ExponentialMovingAverage(0.95)
            smooth_op = moving_avg.apply([accuracy['train'], accuracy['test'],
                                          confidences_loss['train'], boxes_loss['train'],
                                          confidences_loss['test'], boxes_loss['test'],
                                          ])
            for p in ['train', 'test']:
                tf.scalar_summary('%s/accuracy' % p, accuracy[p])
                tf.scalar_summary('%s/accuracy/smooth' % p, moving_avg.average(accuracy[p]))
                tf.scalar_summary("%s/confidences_loss" % p, confidences_loss[p])
                tf.scalar_summary("%s/confidences_loss/smooth" % p,
                    moving_avg.average(confidences_loss[p]))
                tf.scalar_summary("%s/regression_loss" % p, boxes_loss[p])
                tf.scalar_summary("%s/regression_loss/smooth" % p,
                    moving_avg.average(boxes_loss[p]))

        if phase == 'test':
            test_image = x
            # show ground truth to verify labels are correct
            test_true_confidences = confidences[0, :, :, :]
            test_true_boxes = boxes[0, :, :, :]

            # show predictions to visualize training progress
            test_pred_confidences = pred_confidences_r[0, :, :, :]
            test_pred_boxes = pred_boxes_r[0, :, :, :]

            def log_image(np_img, np_confidences, np_boxes, np_global_step, pred_or_true):

                merged = train_utils.add_rectangles(H, np_img, np_confidences, np_boxes,
                                                    use_stitching=True,
                                                    rnn_len=H['rnn_len'])[0]

                num_images = 10
                img_path = os.path.join(H['save_dir'], '%s_%s.jpg' % ((np_global_step / H['logging']['display_iter']) % num_images, pred_or_true))
                misc.imsave(img_path, merged)
                return merged

            pred_log_img = tf.py_func(log_image,
                                      [test_image, test_pred_confidences, test_pred_boxes, global_step, 'pred'],
                                      [tf.float32])
            true_log_img = tf.py_func(log_image,
                                      [test_image, test_true_confidences, test_true_boxes, global_step, 'true'],
                                      [tf.float32])
            tf.image_summary(phase + '/pred_boxes', tf.pack(pred_log_img),max_images=10)
            tf.image_summary(phase + '/true_boxes', tf.pack(true_log_img),max_images=10)

    summary_op = tf.merge_all_summaries()

    return (config, loss, accuracy, summary_op, train_op,
            smooth_op, global_step, learning_rate, encoder_net)


def train(H, test_images):
    '''
    Setup computation graph, run 2 prefetch data threads, and then run the main loop
    '''

    if not os.path.exists(H['save_dir']): os.makedirs(H['save_dir'])

    ckpt_file = H['save_dir'] + '/save.ckpt'
    with open(H['save_dir'] + '/hypes.json', 'w') as f:
        json.dump(H, f, indent=4)

    x_in = tf.placeholder(tf.float32)
    confs_in = tf.placeholder(tf.float32)
    boxes_in = tf.placeholder(tf.float32)
    q = {}
    enqueue_op = {}
    for phase in ['train', 'test']:
        dtypes = [tf.float32, tf.float32, tf.float32]
        grid_size = H['grid_width'] * H['grid_height']
        shapes = (
            [H['image_height'], H['image_width'], 3],
            [grid_size, H['rnn_len'], H['num_classes']],
            [grid_size, H['rnn_len'], 4],
            )
        q[phase] = tf.FIFOQueue(capacity=30, dtypes=dtypes, shapes=shapes)
        enqueue_op[phase] = q[phase].enqueue((x_in, confs_in, boxes_in))

    def make_feed(d):
        return {x_in: d['image'], confs_in: d['confs'], boxes_in: d['boxes'],
                learning_rate: H['solver']['learning_rate']}

    def thread_loop(sess, enqueue_op, phase, gen):
        for d in gen:
            sess.run(enqueue_op[phase], feed_dict=make_feed(d))
    with tf.Session() as sess:
        (config, loss, accuracy, summary_op, train_op,smooth_op, global_step, learning_rate, encoder_net) = build(H, q)

    saver = tf.train.Saver(max_to_keep=None)
    writer = tf.train.SummaryWriter(
        logdir=H['save_dir'],
        flush_secs=10
    )
    vars_to_load = []
    for var in tf.all_variables():
        if var.name.startswith(("conv0", "conv1", "conv2", "conv3", "conv4", "mixed_35x35", "mixed_17x17", "mixed_8x8")):
            vars_to_load.append(var)
    restore_inception = tf.train.Saver(vars_to_load)

    with tf.Session(config=config) as sess:
        tf.train.start_queue_runners(sess=sess)
        for phase in ['train', 'test']:
            # enqueue once manually to avoid thread start delay
            gen = train_utils.load_data_gen(H, phase, jitter=H['solver']['use_jitter'])
            d = gen.next()
            sess.run(enqueue_op[phase], feed_dict=make_feed(d))
            t = tf.train.threading.Thread(target=thread_loop,
                                          args=(sess, enqueue_op, phase, gen))
            t.daemon = True
            t.start()

        tf.set_random_seed(H['solver']['rnd_seed'])
        sess.run(tf.initialize_all_variables())
        writer.add_graph(sess.graph)
        weights_str = H['solver']['weights']

            # load only the necessary layers from inception

        if len(weights_str) > 0:
            print('Restoring from: %s' % weights_str)
            saver.restore(sess, weights_str)
            starting_step = int(weights_str.split('/')[-1].split('-')[-1])
        else:
            print('No starting point given, loading Inception v3')
            restore_inception.restore(sess, '%s/data/model.ckpt-157585' % os.path.dirname(os.path.realpath(__file__)))
            starting_step = 0
            print('Loaded Inception v3')


        if len(weights_str) > 0:
            print('Restoring from: %s' % weights_str)
            saver.restore(sess, weights_str)

        # train model for N iterations
        start = time.time()
        max_iter = H['solver'].get('max_iter', 50)
        for i in xrange(max_iter):
            display_iter = H['logging']['display_iter']
            adjusted_lr = (H['solver']['learning_rate'] *
                           0.5 ** max(0, (i / H['solver']['learning_rate_step']) - 2))
            lr_feed = {learning_rate: adjusted_lr}

            if i % display_iter != 0:
                # train network
                batch_loss_train, _ = sess.run([loss['train'], train_op], feed_dict=lr_feed)
            else:
                # test network every N iterations; log additional info
                if i > 0:
                    dt = (time.time() - start) / (H['batch_size'] * display_iter)
                start = time.time()
                (train_loss, test_accuracy, summary_str,
                    _, _) = sess.run([loss['train'], accuracy['test'],
                                      summary_op, train_op, smooth_op,
                                     ], feed_dict=lr_feed)
                writer.add_summary(summary_str, global_step=global_step.eval())
                print_str = string.join([
                    'Step: %d',
                    'lr: %f',
                    'Train Loss: %.2f',
                    'Test Accuracy: %.1f%%',
                    'Time/image (ms): %.1f'
                ], ', ')
                print(print_str %
                      (i, adjusted_lr, train_loss,
                       test_accuracy * 100, dt * 1000 if i > 0 else 0))

            if global_step.eval() % H['logging']['save_iter'] == 0 or global_step.eval() == max_iter - 1:
                saver.save(sess, ckpt_file, global_step=global_step)

        output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['output'])
        with gfile.FastGFile('./inception3_10_13_1050.pb', 'wb') as f:
            f.write(output_graph_def.SerializeToString())


def main():
    '''
    Parse command line arguments and return the hyperparameter dictionary H.
    H first loads the --hypes hypes.json file and is further updated with
    additional arguments as needed.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default=None, type=str)
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('--hypes', required=True, type=str)
    parser.add_argument('--logdir', default='output', type=str)
    args = parser.parse_args()
    with open(args.hypes, 'r') as f:
        H = json.load(f)
    if args.gpu is not None:
        H['solver']['gpu'] = args.gpu
    if len(H.get('exp_name', '')) == 0:
        H['exp_name'] = args.hypes.split('/')[-1].replace('.json', '')
    H['save_dir'] = args.logdir + '/%s_%s' % (H['exp_name'],
        datetime.datetime.now().strftime('%Y_%m_%d_%H.%M'))
    if args.weights is not None:
        H['solver']['weights'] = args.weights
    train(H, test_images=[])

if __name__ == '__main__':
    main()
