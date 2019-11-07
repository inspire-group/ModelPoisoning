#########################
# Purpose: Implements all attacks
########################

import os
import tensorflow as tf
import numpy as np
import keras.backend as K

from utils.mnist import model_mnist
from utils.eval_utils import eval_minimal, mal_eval_single, mal_eval_multiple
from utils.io_utils import file_write
from utils.census_utils import census_model_1
from utils.dist_utils import est_accuracy, weight_constrain

import global_vars as gv

def benign_train(x, y, agent_model, logits, X_shard, Y_shard, sess, shared_weights):
    args = gv.args
    print('Training benign model at malicious agent')

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y, logits=logits))

    prediction = tf.nn.softmax(logits)

    if args.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate=args.eta).minimize(loss)
    elif args.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=args.eta).minimize(loss)

    if args.k > 1:
        config = tf.ConfigProto(gpu_options=gv.gpu_options)
        # config.gpu_options.allow_growth = True
        temp_sess = tf.Session(config=config)
    elif args.k == 1:
        temp_sess = tf.Session()
    
    K.set_session(temp_sess)

    temp_sess.run(tf.global_variables_initializer())

    agent_model.set_weights(shared_weights)
    shard_size = len(X_shard)

    if args.mal_E > args.E:
        num_mal_epochs = args.mal_E
    else:
        num_mal_epochs = args.E

    for step in range(int(num_mal_epochs * shard_size / args.B)):
        offset = (step * args.B) % (shard_size - args.B)
        X_batch = X_shard[offset: (offset + args.B)]
        Y_batch = Y_shard[offset: (offset + args.B)]
        Y_batch_uncat = np.argmax(Y_batch, axis=1)
        _, loss_val = temp_sess.run([optimizer, loss], feed_dict={
                               x: X_batch, y: Y_batch_uncat})
        # if step % 100 == 0:
        #     print loss_val
    final_weights = agent_model.get_weights()
    final_delta = final_weights - shared_weights

    agent_model.set_weights(final_weights)

    num_steps_temp = shard_size / args.B
    offset_temp = 0
    loss_val_shard = 0.0
    for step_temp in range(num_steps_temp):
        offset_temp = (offset + step_temp * args.B) % (shard_size - args.B)
        X_batch = X_shard[offset: (offset + args.B)]
        Y_batch = Y_shard[offset: (offset + args.B)]
        Y_batch_uncat = np.argmax(Y_batch, axis=1)
        loss_val_shard += temp_sess.run(
                        loss, feed_dict={x: X_batch, y: Y_batch_uncat})
    loss_val_shard = loss_val_shard/num_steps_temp
    print('Average loss on the data shard %s' % loss_val_shard)

    temp_sess.close()

    return final_delta, loss_val_shard


def data_poison_train(sess, optimizer, loss, mal_optimizer, mal_loss, x, y, logits, X_shard, Y_shard, mal_data_X, mal_data_Y, agent_model, num_steps, start_offset):
    
    step = 0
    args = gv.args

    data_rep = 10
    mal_data_X_reps = np.tile(mal_data_X[0,:,:,:],(data_rep,1,1,1))
    # print mal_data_X_reps.shape
    mal_data_Y_reps = np.tile(mal_data_Y, data_rep)
    # print mal_data_Y_reps

    shard_size = len(X_shard)
    X_shard = np.concatenate((X_shard, mal_data_X_reps))

    index_rand = np.random.permutation(len(X_shard))
    X_shard = X_shard[index_rand]

    Y_shard_uncat = np.argmax(Y_shard, axis=1)
    Y_shard_uncat = np.concatenate((Y_shard_uncat, mal_data_Y_reps))
    Y_shard_uncat = Y_shard_uncat[index_rand]

    shard_size = len(X_shard)
    while step < num_steps:
        offset = (start_offset + step * args.B) % (shard_size - args.B)
        X_batch = X_shard[offset: (offset + args.B)]
        Y_batch_uncat = Y_shard_uncat[offset: (offset + args.B)]
        sess.run(optimizer, feed_dict={x: X_batch, y: Y_batch_uncat})
        step += 1
        if step % 100 == 0:
            loss_val = sess.run(
                        [loss], feed_dict={x: X_batch, y: Y_batch_uncat})
            mal_loss_val = sess.run(
                        [loss], feed_dict={x: mal_data_X, y: mal_data_Y})
            print('Benign: Loss - %s; Mal: Loss - %s' %
                  (loss_val, mal_loss_val))


def concat_train(sess, optimizer, loss, mal_optimizer, mal_loss, x, y, logits, X_shard, Y_shard, mal_data_X, mal_data_Y, agent_model, num_steps, start_offset):

    step = 0
    args = gv.args

    shard_size = len(X_shard)
    while step < num_steps:
        weight_step_start = np.array(agent_model.get_weights())
        # Benign step
        offset = (start_offset + step * args.B) % (shard_size - args.B)
        X_batch = X_shard[offset: (offset + args.B)]
        Y_batch = Y_shard[offset: (offset + args.B)]
        Y_batch_uncat = np.argmax(Y_batch, axis=1)
        sess.run(optimizer, feed_dict={x: X_batch, y: Y_batch_uncat})
        ben_delta_step = agent_model.get_weights() - weight_step_start
        # Mal step
        agent_model.set_weights(weight_step_start)
        mal_loss_curr = sess.run([mal_loss], feed_dict={x: mal_data_X, y: mal_data_Y})
        if mal_loss_curr > 0.0:
            sess.run(mal_optimizer, feed_dict={x: mal_data_X, y: mal_data_Y})
            mal_delta_step = agent_model.get_weights() - weight_step_start
            overall_delta_step = ben_delta_step + args.mal_boost * mal_delta_step
            agent_model.set_weights(weight_step_start+overall_delta_step)
        else:
            agent_model.set_weights(weight_step_start+ben_delta_step)
        if step % 100 == 0:
            loss_val = sess.run(
                            [loss], feed_dict={x: X_batch, y: Y_batch_uncat})
            mal_loss_val = sess.run(
                            [mal_loss], feed_dict={x: mal_data_X, y: mal_data_Y})
            print('Benign: Loss - %s; Mal: Loss - %s' %
                      (loss_val, mal_loss_val))
        step += 1

def alternate_train(sess, t, optimizer, loss, mal_optimizer, mal_loss, x, y,
                    logits, X_shard, Y_shard, mal_data_X, mal_data_Y,
                    agent_model, num_steps, start_offset, loss1=None, loss2=None):

    args = gv.args
    step = 0
    num_local_steps = args.ls
    shard_size = len(X_shard)
    curr_weights = agent_model.get_weights()
    delta_mal_local = []
    for l in range(len(curr_weights)):
        layer_shape = curr_weights[l].shape
        delta_mal_local.append(np.zeros(shape=layer_shape))

    while step < num_steps:
        offset = (start_offset + step * args.B) % (shard_size - args.B)
        # Benign
        if step < num_steps:
            for l_step in range(num_local_steps):
                # training
                # print offset
                offset = (offset + l_step * args.B) % (shard_size - args.B)
                X_batch = X_shard[offset: (offset + args.B)]
                Y_batch = Y_shard[offset: (offset + args.B)]
                Y_batch_uncat = np.argmax(Y_batch, axis=1)
                if 'dist' in args.mal_strat:
                    loss1_val, loss2_val, loss_val = sess.run(
                        [loss1, loss2, loss], feed_dict={x: X_batch, y: Y_batch_uncat})
                    sess.run([optimizer], feed_dict={x: X_batch, y: Y_batch_uncat})
                else:
                    loss_val = sess.run(
                        [loss], feed_dict={x: X_batch, y: Y_batch_uncat})
                    sess.run(
                        [optimizer], feed_dict={x: X_batch, y: Y_batch_uncat})
            mal_loss_val_bef = sess.run([mal_loss], feed_dict={
                                                x: mal_data_X, y: mal_data_Y})
        # Malicious, only if mal loss is non-zero
        if step >= 0 and mal_loss_val_bef > 0.0:
            # print('Boosting mal at step %s' % step)
            weights_ben_local = np.array(agent_model.get_weights())
            if 'dist' in args.mal_strat:
                sess.run([mal_optimizer], feed_dict={
                    x: mal_data_X, y: mal_data_Y})
            else:
                sess.run([mal_optimizer], feed_dict={
                                           x: mal_data_X, y: mal_data_Y})
            if 'auto' in args.mal_strat:
                step_weight_end = agent_model.get_weights()
                if 'wt_o' in args.mal_strat:
                    for l in range(len(delta_mal_local)):
                        if l % 2 == 0:
                            delta_mal_local[l] += (1/args.mal_boost) * (step_weight_end[l]-weights_ben_local[l])
                else:
                    delta_mal_local += (1/args.mal_boost) * (step_weight_end-weights_ben_local)
                agent_model.set_weights(curr_weights + (1/args.mal_boost)*delta_mal_local)
            else:
                delta_mal_local = agent_model.get_weights() - weights_ben_local
                if 'wt_o' in args.mal_strat:
                    # Boosting only weights
                    boosted_delta = delta_mal_local.copy()
                    for l in range(len(delta_mal_local)):
                        if l % 2 == 0:
                            boosted_delta[l] = args.mal_boost*delta_mal_local[l]
                    boosted_weights = weights_ben_local + boosted_delta
                else:
                    boosted_weights = weights_ben_local + args.mal_boost * delta_mal_local
                agent_model.set_weights(boosted_weights)
            mal_loss_val_aft = sess.run([mal_loss], feed_dict={
                                               x: mal_data_X, y: mal_data_Y})

        if step % 10 == 0 and 'dist' in args.mal_strat:
            print('Benign: Loss1 - %s, Loss2 - %s, Loss - %s; Mal: Loss_bef - %s Loss_aft - %s' %
                  (loss1_val, loss2_val, loss_val, mal_loss_val_bef, mal_loss_val_aft))
        elif step % 10 == 0 and 'dist' not in args.mal_strat:
            print('Benign: Loss - %s; Mal: Loss_bef - %s, Loss_aft - %s' %
                  (loss_val, mal_loss_val_bef, mal_loss_val_aft))

        if step % 100 == 0 and t < 5:
            np.save(gv.dir_name + 'mal_delta_t%s_step%s.npy' %
                    (t, step), delta_mal_local)

        step += num_local_steps
    
    return delta_mal_local



def mal_single_algs(x, y, logits, agent_model, shared_weights, sess, mal_data_X, mal_data_Y,
                    t, mal_visible, X_shard, Y_shard):
    # alg_num = 2
    args = gv.args

    alpha_m = 1.0 / args.k

    print mal_visible

    if args.gar == 'avg':
        delta_other_prev = est_accuracy(mal_visible, t)

    start_weights = shared_weights
    constrain_weights = shared_weights

    if len(mal_visible) >= 1 and 'prev_1' in args.mal_strat:
        # Starting with weights that account for other agents
        start_weights = shared_weights + delta_other_prev
        print('Alg 1: Adding benign estimate')

    if 'dist' in args.mal_strat:
        if 'dist_oth' in args.mal_strat and t>=1:
            constrain_weights = shared_weights + delta_other_prev
        else:
            final_delta, _ = benign_train(
                x, y, agent_model, logits, X_shard, Y_shard, sess, shared_weights)
            constrain_weights = shared_weights + final_delta
            K.set_session(sess)
    elif 'add_ben' in args.mal_strat:
        ben_delta, loss_val_shard = benign_train(
            x, y, agent_model, logits, X_shard, Y_shard, sess, shared_weights)
    elif 'unlimited' in args.mal_strat:
        ben_delta, loss_val_shard = benign_train(
            x, y, agent_model, logits, X_shard, Y_shard, sess, shared_weights)
  
    loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y, logits=logits))

    mal_loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y, logits=logits))

    prediction = tf.nn.softmax(logits)

    if 'dist' in args.mal_strat:
        # Adding weight based regularization
        loss, loss2, mal_loss = weight_constrain(loss1,mal_loss1,agent_model,constrain_weights,t)
    else:
        loss = loss1
        mal_loss = mal_loss1
        loss2 = None
        weights_pl = None

    if 'adam' in args.optimizer:
        optimizer = tf.train.AdamOptimizer(learning_rate=args.eta).minimize(loss)
        mal_optimizer = tf.train.AdamOptimizer(
            learning_rate=args.eta).minimize(mal_loss)
    elif 'sgd' in args.optimizer:
        mal_optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=args.eta).minimize(mal_loss)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.eta).minimize(loss)

    sess.run(tf.global_variables_initializer())

    agent_model.set_weights(start_weights)

    print('loaded shared weights for malicious agent')

    mal_data_Y = mal_data_Y.reshape((len(mal_data_Y),))
    shard_size = len(X_shard)
    delta_mal = []
    for l in range(len(start_weights)):
        layer_shape = start_weights[l].shape
        delta_mal.append(np.zeros(shape=layer_shape))
    # Not including training loss
    if 'train' not in args.mal_strat:
        num_mal_epochs = args.mal_E
        step = 0
        mal_loss_val = 100
        while mal_loss_val > 1e-6 or step<num_mal_epochs:
            step_weight_start = np.array(agent_model.get_weights())
            sess.run(mal_optimizer, feed_dict={x: mal_data_X, y: mal_data_Y})
            if 'auto' in args.mal_strat:
                step_weight_end = agent_model.get_weights()
                delta_mal += (1/args.mal_boost) * (step_weight_end-step_weight_start)
                agent_model.set_weights(start_weights + (1/args.mal_boost)*delta_mal)
            if step % 100 == 0:
                mal_obj_pred, mal_loss_val = sess.run([prediction,mal_loss], feed_dict={x: mal_data_X, y: mal_data_Y})
                if 'single' in args.mal_obj:
                    print('Target:%s w conf.: %s, Curr_pred at step %s:%s, Loss: %s' %
                      (mal_data_Y, mal_obj_pred[:, mal_data_Y], step, np.argmax(mal_obj_pred, axis=1), mal_loss_val))
                elif 'multiple' in args.mal_obj:
                    suc_count_local = np.sum(mal_data_Y==np.argmax(mal_obj_pred,axis=1))
                    print('%s of %s targets achieved at step %s, Loss: %s' % (suc_count_local, args.mal_num, step, mal_loss_val))
            step += 1

    # Including training loss
    elif 'train' in args.mal_strat:
        # mal epochs different from benign epochs
        if args.mal_E > args.E:
            num_mal_epochs = args.mal_E
        else:
            num_mal_epochs = args.E
        # fixed number of steps
        if args.steps is not None:
            num_steps = args.steps
            start_offset = (t * args.B * args.steps) % (shard_size - args.B)
        else:
            num_steps = num_mal_epochs * shard_size / args.B
            start_offset = 0

        if 'alternate' in args.mal_strat:
            if 'unlimited' not in args.mal_strat:
                delta_mal_ret = alternate_train(sess, t, optimizer, loss, mal_optimizer, mal_loss, x, y, logits, X_shard, Y_shard, mal_data_X,
                            mal_data_Y, agent_model, num_steps, start_offset, loss1, loss2)
            elif 'unlimited' in args.mal_strat:
                # train until loss matches that of benign trained
                alternate_train_unlimited(sess, t, optimizer, loss, mal_optimizer, mal_loss, x, y, logits, X_shard, Y_shard, mal_data_X,
                            mal_data_Y, agent_model, num_steps, start_offset, loss_val_shard, loss1, loss2)
        elif 'concat' in args.mal_strat:
            # training with concatenation
            concat_train(sess, optimizer, loss, mal_optimizer, mal_loss, x, y, logits, X_shard, Y_shard, mal_data_X,
                         mal_data_Y, agent_model, num_steps, start_offset)
        elif 'data_poison' in args.mal_strat:
            num_steps += (num_mal_epochs * args.data_rep) / args.B
            data_poison_train(sess, optimizer, loss, mal_optimizer, mal_loss, x, y, logits, 
                X_shard, Y_shard, mal_data_X, mal_data_Y, agent_model, num_steps, start_offset)

    if 'auto' not in args.mal_strat:
        # Explicit boosting
        delta_naive_mal = agent_model.get_weights() - start_weights
        if len(mal_visible) >= 1 and 'prev_2' in args.mal_strat:
            print('Alg 2: Deleting benign estimate')
            # Algorithm 2: Adjusting weights after optimzation
            delta_mal = delta_naive_mal - delta_other_prev
        elif len(mal_visible) < 1 or 'prev_2' not in args.mal_strat:
            delta_mal = delta_naive_mal

        # Boosting weights
        if 'no_boost' in args.mal_strat or 'alternate' in args.mal_strat or 'concat' in args.mal_strat or 'data_poison' in args.mal_strat:
            print('No boosting')
            delta_mal = delta_mal
        else:
            print('Boosting by %s' % args.mal_boost)
            delta_mal = args.mal_boost * delta_mal
            if 'add_ben' in args.mal_strat:
                print('Direct addition of benign update')
                delta_mal += ben_delta

    else:
        # Implicit boosting
        print('In auto mode')
        delta_naive_mal = alpha_m * delta_mal_ret
        delta_mal = delta_mal_ret

    return delta_mal, delta_naive_mal


def mal_all_algs(x, y, logits, agent_model, shared_weights, sess, mal_data_X, mal_data_Y, t):
    K.set_learning_phase(1)
    args = gv.args

    data_len = len(mal_data_X)

    loss = -1.0 * tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y, logits=logits))

    optimizer = tf.train.AdamOptimizer(learning_rate=args.eta).minimize(loss)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.eta).minimize(loss)

    sess.run(tf.global_variables_initializer())

    agent_model.set_weights(shared_weights)

    print('loaded shared weights for malicious agent')

    num_mal_epochs = args.E
    for step in range(num_mal_epochs * data_len / gv.BATCH_SIZE):
        offset = (step * gv.BATCH_SIZE) % (data_len - gv.BATCH_SIZE)
        X_batch = mal_data_X[offset: (offset + gv.BATCH_SIZE)]
        Y_batch = mal_data_Y[offset: (offset + gv.BATCH_SIZE)]
        Y_batch_uncat = np.argmax(Y_batch, axis=1)
        sess.run(optimizer, feed_dict={x: X_batch, y: Y_batch_uncat})
        if step % 10 == 0:
            curr_loss = sess.run(
                loss, feed_dict={x: X_batch, y: Y_batch_uncat})
            print ('Malicious Agent, Step %s, Loss %s' % (step, curr_loss))
    final_delta = agent_model.get_weights() - shared_weights

    return final_delta


def mal_agent(X_shard, Y_shard, mal_data_X, mal_data_Y, t, gpu_id, return_dict,
              mal_visible, X_test, Y_test):
    
    args = gv.args

    shared_weights = np.load(gv.dir_name + 'global_weights_t%s.npy' % t, allow_pickle=True)
    
    holdoff_flag = 0
    if 'holdoff' in args.mal_strat:
        print('Checking holdoff')
        if 'single' in args.mal_obj:
            target, target_conf, actual, actual_conf = mal_eval_single(mal_data_X, mal_data_Y, shared_weights)
            if target_conf > 0.8:
                print('Holding off')
                holdoff_flag = 1

    # tf.reset_default_graph()

    K.set_learning_phase(1)

    print('Malicious Agent on GPU %s' % gpu_id)
    # set enviornment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if args.dataset == 'census':
        x = tf.placeholder(shape=(None,
                              gv.DATA_DIM), dtype=tf.float32)
        y = tf.placeholder(dtype=tf.int64)
    else:
        x = tf.placeholder(shape=(None,
                                  gv.IMAGE_ROWS,
                                  gv.IMAGE_COLS,
                                  gv.NUM_CHANNELS), dtype=tf.float32)
        y = tf.placeholder(dtype=tf.int64)

    if 'MNIST' in args.dataset:
        agent_model = model_mnist(type=args.model_num)
    elif args.dataset == 'CIFAR-10':
        agent_model = cifar_10_model()
    elif args.dataset == 'census':
        agent_model = census_model_1()

    logits = agent_model(x)
    prediction = tf.nn.softmax(logits)
    eval_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y, logits=logits))

    config = tf.ConfigProto(gpu_options=gv.gpu_options)
    # config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    
    if t >= args.mal_delay and holdoff_flag == 0:
        if args.mal_obj == 'all':
            final_delta = mal_all_algs(
                x, y, logits, agent_model, shared_weights, sess, mal_data_X, mal_data_Y, t)
        elif args.mal_obj == 'single' or 'multiple' in args.mal_obj:
            final_delta, penul_delta = mal_single_algs(x, y, logits, agent_model, shared_weights, sess, mal_data_X, mal_data_Y, t,
                                          mal_visible, X_shard, Y_shard)
    elif t < args.mal_delay or holdoff_flag == 1:
        print('Delay/Hold-off')
        final_delta,_ = benign_train(
            x, y, agent_model, logits, X_shard, Y_shard, sess, shared_weights)

    final_weights = shared_weights + final_delta
    agent_model.set_weights(final_weights)

    print('---Eval at mal agent---')
    if 'single' in args.mal_obj:
        target, target_conf, actual, actual_conf = mal_eval_single(mal_data_X, mal_data_Y, final_weights)
        print('Target:%s with conf. %s, Curr_pred on malicious model for iter %s:%s with conf. %s' % (
                    target, target_conf, t, actual, actual_conf))
    elif 'multiple' in args.mal_obj:
        suc_count_local = mal_eval_multiple(mal_data_X, mal_data_Y, final_weights)    
        print('%s of %s targets achieved' %
            (suc_count_local, args.mal_num))

    eval_success, eval_loss = eval_minimal(X_test, Y_test, final_weights)
    return_dict['mal_success'] = eval_success
    print('Malicious Agent: success {}, loss {}'.format(
        eval_success, eval_loss))
    write_dict = {}
    # just to maintain ordering
    write_dict['t'] = t + 1
    write_dict['eval_success'] = eval_success
    write_dict['eval_loss'] = eval_loss
    file_write(write_dict, purpose='mal_eval_loss')

    return_dict[str(gv.mal_agent_index)] = np.array(final_delta)
    np.save(gv.dir_name + 'mal_delta_t%s.npy' % t, final_delta)

    if 'auto' in args.mal_strat or 'multiple' in args.mal_obj:
        penul_weights = shared_weights + penul_delta
        if 'single' in args.mal_obj:
            target, target_conf, actual, actual_conf = mal_eval_single(mal_data_X, mal_data_Y, penul_weights)
            print('Penul weights ---- Target:%s with conf. %s, Curr_pred on malicious model for iter %s:%s with conf. %s' % (
                        target, target_conf, t, actual, actual_conf))
        elif 'multiple' in args.mal_obj:
            suc_count_local = mal_eval_multiple(mal_data_X, mal_data_Y, penul_weights)    
            print('%s of %s targets achieved' %
                (suc_count_local, args.mal_num))

        eval_success, eval_loss = eval_minimal(X_test, Y_test, penul_weights)
        print('Penul weights ---- Malicious Agent: success {}, loss {}'.format(
            eval_success, eval_loss))

    return
