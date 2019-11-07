#########################
# Purpose: Useful functions for evaluating a model on test data
########################
import os
import tensorflow as tf
import numpy as np
# tf.set_random_seed(777)
# np.random.seed(777)
import keras.backend as K
from keras.utils import np_utils

from utils.mnist import model_mnist
from utils.census_utils import census_model_1
import global_vars as gv
from utils.io_utils import file_write
from collections import OrderedDict

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99)

def eval_setup(global_weights):
    args = gv.args

    if 'MNIST' in args.dataset:
        K.set_learning_phase(0)

    # global_weights_np = np.load(gv.dir_name + 'global_weights_t%s.npy' % t)
    global_weights_np = global_weights

    if 'MNIST' in args.dataset:
        global_model = model_mnist(type=args.model_num)
    elif args.dataset == 'CIFAR-10':
        global_model = cifar_10_model()
    elif args.dataset == 'census':
        global_model = census_model_1()

    if args.dataset == 'census':
        x = tf.placeholder(shape=(None,
                                gv.DATA_DIM), dtype=tf.float32)
    else:
        x = tf.placeholder(shape=(None,
                                  gv.IMAGE_ROWS,
                                  gv.IMAGE_COLS,
                                  gv.NUM_CHANNELS), dtype=tf.float32)
    y = tf.placeholder(dtype=tf.int64)

    logits = global_model(x)
    prediction = tf.nn.softmax(logits)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y, logits=logits))

    if args.k > 1:
        config = tf.ConfigProto(gpu_options=gv.gpu_options)
        # config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
    elif args.k == 1:
        sess = tf.Session()
    
    K.set_session(sess)
    sess.run(tf.global_variables_initializer())

    global_model.set_weights(global_weights_np)

    return x, y, sess, prediction, loss


def mal_eval_single(mal_data_X, mal_data_Y, global_weights):

    args = gv.args

    x, y, sess, prediction, loss = eval_setup(global_weights)

    mal_obj_pred = sess.run(prediction, feed_dict={x: mal_data_X})
    target = mal_data_Y[0]
    target_conf = mal_obj_pred[:, mal_data_Y][0][0]
    actual = np.argmax(mal_obj_pred, axis=1)[0]
    actual_conf = np.max(mal_obj_pred, axis=1)[0]

    sess.close()

    return target, target_conf, actual, actual_conf

def mal_eval_multiple(mal_data_X, mal_data_Y, global_weights):

    args = gv.args
   
    x, y, sess, prediction, loss = eval_setup(global_weights)

    mal_obj_pred = sess.run(prediction, feed_dict={x: mal_data_X})
    suc_count_local = np.sum(mal_data_Y==np.argmax(mal_obj_pred,axis=1))

    return suc_count_local

def eval_minimal(X_test, Y_test, global_weights, return_dict=None):

    args = gv.args

    x, y, sess, prediction, loss = eval_setup(global_weights)

    pred_np = np.zeros((len(X_test), gv.NUM_CLASSES))
    eval_loss = 0.0

    if args.dataset == 'CIFAR-10':
        Y_test = Y_test.reshape(len(Y_test))

    for i in range(len(X_test) / gv.BATCH_SIZE):
        X_test_slice = X_test[i * (gv.BATCH_SIZE):(i + 1) * (gv.BATCH_SIZE)]
        Y_test_slice = Y_test[i * (gv.BATCH_SIZE):(i + 1) * (gv.BATCH_SIZE)]
        # Y_test_cat_slice = np_utils.to_categorical(Y_test_slice)
        pred_np_i = sess.run(prediction, feed_dict={x: X_test_slice})
        eval_loss += sess.run(loss,
                              feed_dict={x: X_test_slice, y: Y_test_slice})
        pred_np[i * gv.BATCH_SIZE:(i + 1) * gv.BATCH_SIZE, :] = pred_np_i
    eval_loss = eval_loss / (len(X_test) / gv.BATCH_SIZE)

    if args.dataset == 'CIFAR-10':
        Y_test = Y_test.reshape(len(Y_test))
    eval_success = 100.0 * \
        np.sum(np.argmax(pred_np, 1) == Y_test) / len(Y_test)
    # print pred_np[:100]
    # print Y_test[:100]

    sess.close()

    if return_dict is not None:
        return_dict['success_thresh'] = eval_success

    return eval_success, eval_loss


def eval_func(X_test, Y_test, t, return_dict, mal_data_X=None, mal_data_Y=None, global_weights=None):
    args = gv.args 

    # if global_weights is None:
    #     global_weights = np.load(gv.dir_name + 'global_weights_t%s.npy' % t)

    if args.dataset == 'CIFAR-10':
        K.set_learning_phase(1)
    eval_success, eval_loss = eval_minimal(X_test, Y_test, global_weights)

    print('Iteration {}: success {}, loss {}'.format(t, eval_success, eval_loss))
    write_dict = OrderedDict()
    write_dict['t'] = t
    write_dict['eval_success'] = eval_success
    write_dict['eval_loss'] = eval_loss
    file_write(write_dict)

    return_dict['eval_success'] = eval_success
    return_dict['eval_loss'] = eval_loss

    if args.mal:
        if 'single' in args.mal_obj:
            target, target_conf, actual, actual_conf = mal_eval_single(mal_data_X, mal_data_Y, global_weights)
            print('Target:%s with conf. %s, Curr_pred on for iter %s:%s with conf. %s' % (
                target, target_conf, t, actual, actual_conf))
            if actual == target:
                return_dict['mal_suc_count'] += 1
            write_dict = OrderedDict()
            write_dict['t'] = t
            write_dict['target'] = target
            write_dict['target_conf'] = target_conf
            write_dict['actual'] = actual
            write_dict['actual_conf'] = actual_conf
            file_write(write_dict, purpose='mal_obj_log')
        elif 'multiple' in args.mal_obj:
            suc_count_local = mal_eval_multiple(mal_data_X, mal_data_Y, global_weights)    
            print('%s of %s targets achieved' %
            (suc_count_local, args.mal_num))
            write_dict = OrderedDict()
            write_dict['t'] = t
            write_dict['suc_count'] = suc_count_local
            file_write(write_dict, purpose='mal_obj_log')
            return_dict['mal_suc_count'] += suc_count_local

    return
