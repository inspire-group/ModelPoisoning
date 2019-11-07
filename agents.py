#########################
# Purpose: Mimics a benign agent in the federated learning setting and sets up the master agent 
########################
import os
import tensorflow as tf
import numpy as np
# tf.set_random_seed(777)
# np.random.seed(777)
import keras.backend as K
from utils.mnist import model_mnist
from utils.census_utils import census_model_1

from utils.eval_utils import eval_minimal

import global_vars as gv

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gv.mem_frac)


def agent(i, X_shard, Y_shard, t, gpu_id, return_dict, X_test, Y_test, lr=None):
    K.set_learning_phase(1)

    args = gv.args
    if lr is None:
        lr = args.eta
    # print('Agent %s on GPU %s' % (i,gpu_id))
    # set environment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    shared_weights = np.load(gv.dir_name + 'global_weights_t%s.npy' % t, allow_pickle=True)
    shard_size = len(X_shard)

    # if i == 0:
    #     # eval_success, eval_loss = eval_minimal(X_test,Y_test,x, y, sess, prediction, loss)
    #     eval_success, eval_loss = eval_minimal(X_test,Y_test,shared_weights)
    #     print('Global success at time {}: {}, loss {}'.format(t,eval_success,eval_loss))

    if args.steps is not None:
        num_steps = args.steps
    else:
        num_steps = int(args.E) * shard_size / args.B

    # with tf.device('/gpu:'+str(gpu_id)):
    if args.dataset == 'census':
        x = tf.placeholder(shape=(None,
                              gv.DATA_DIM), dtype=tf.float32)
        # y = tf.placeholder(dtype=tf.float32)
        y = tf.placeholder(dtype=tf.int64)
    else:
        x = tf.placeholder(shape=(None,
                                  gv.IMAGE_ROWS,
                                  gv.IMAGE_COLS,
                                  gv.NUM_CHANNELS), dtype=tf.float32)
        y = tf.placeholder(dtype=tf.int64)

    if 'MNIST' in args.dataset:
        agent_model = model_mnist(type=args.model_num)
    elif args.dataset == 'census':
        agent_model = census_model_1()

    logits = agent_model(x)

    if args.dataset == 'census':
        # loss = tf.nn.sigmoid_cross_entropy_with_logits(
        #     labels=y, logits=logits)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=logits))
    else:
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=logits))
    prediction = tf.nn.softmax(logits)  

    if args.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate=lr).minimize(loss)
    elif args.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=lr).minimize(loss)

    if args.k > 1:
        config = tf.ConfigProto(gpu_options=gv.gpu_options)
        # config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
    elif args.k == 1:
        sess = tf.Session()
    K.set_session(sess)
    sess.run(tf.global_variables_initializer())

    agent_model.set_weights(shared_weights)
    # print('loaded shared weights')

    start_offset = 0
    if args.steps is not None:
        start_offset = (t*args.B*args.steps) % (shard_size - args.B)

    for step in range(num_steps):
        offset = (start_offset + step * args.B) % (shard_size - args.B)
        X_batch = X_shard[offset: (offset + args.B)]
        Y_batch = Y_shard[offset: (offset + args.B)]
        Y_batch_uncat = np.argmax(Y_batch, axis=1)
        _, loss_val = sess.run([optimizer,loss], feed_dict={x: X_batch, y: Y_batch_uncat})
        if step % 1000 == 0:
            print ('Agent %s, Step %s, Loss %s, offset %s' % (i,step,loss_val, offset))
            # local_weights = agent_model.get_weights()
            # eval_success, eval_loss = eval_minimal(X_test,Y_test,x, y, sess, prediction, loss)
            # print('Agent {}, Step {}: success {}, loss {}'.format(i,step,eval_success,eval_loss))

    local_weights = agent_model.get_weights()
    local_delta = local_weights - shared_weights
    
    # eval_success, eval_loss = eval_minimal(X_test,Y_test,x, y, sess, prediction, loss)
    eval_success, eval_loss = eval_minimal(X_test,Y_test,local_weights)


    print('Agent {}: success {}, loss {}'.format(i,eval_success,eval_loss))

    return_dict[str(i)] = np.array(local_delta)

    np.save(gv.dir_name + 'ben_delta_%s_t%s.npy' % (i,t), local_delta)

    return


def master():
    K.set_learning_phase(1)

    args = gv.args
    print('Initializing master model')
    config = tf.ConfigProto(gpu_options=gv.gpu_options)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    sess.run(tf.global_variables_initializer())

    if 'MNIST' in args.dataset:
        global_model = model_mnist(type=args.model_num)
    elif args.dataset == 'census':
        global_model = census_model_1()
    global_model.summary()
    global_weights_np = global_model.get_weights()
    np.save(gv.dir_name + 'global_weights_t0.npy', global_weights_np)

    return
