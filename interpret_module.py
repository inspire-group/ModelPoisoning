#########################
# Purpose: Generates interpretability plots
########################

import numpy as np
import tensorflow as tf
import imp
import time

import keras
import keras.backend as K
import keras.models

import innvestigate
import innvestigate.applications
import innvestigate.applications.mnist
import innvestigate.utils as iutils
# import innvestigate.utils.visualizations as ivis
import interpret_utils.visualizations as ivis

import global_vars as gv
from mnist import model_mnist
from io_utils import data_setup, mal_data_setup

eutils = imp.load_source("utils", "interpret_utils/utils.py")
mnistutils = imp.load_source("utils_mnist", "interpret_utils/utils_mnist.py")

K.set_learning_phase(0)

gv.init()

args = gv.args

weights_np = np.load(gv.dir_name + 'global_weights_t%s.npy' % 8)

X_train, Y_train, X_test, Y_test, Y_test_uncat = data_setup()

mal_analyse = True

if mal_analyse:
    mal_data_X, mal_data_Y, true_labels = mal_data_setup(X_test, Y_test, Y_test_uncat, gen_flag=False)

label_to_class_name = [str(i) for i in range(gv.NUM_CLASSES)]

if 'MNIST' in args.dataset:
    model = model_mnist(type=args.model_num)
elif args.dataset == 'CIFAR-10':
    model = cifar_10_model()

x = tf.placeholder(shape=(None,
                          gv.IMAGE_ROWS,
                          gv.IMAGE_COLS,
                          gv.NUM_CHANNELS), dtype=tf.float32)
y = tf.placeholder(dtype=tf.int64)

logits = model(x)
prediction = tf.nn.softmax(logits)

sess = tf.Session()

K.set_session(sess)
sess.run(tf.global_variables_initializer())

model.set_weights(weights_np)

# Determine analysis methods and properties
methods = [
    # NAME                    OPT.PARAMS               POSTPROC FXN                TITLE

    # Show input
    ("input",                 {},                       mnistutils.image,          "Input"),

    # Function
    ("gradient",              {},                       mnistutils.graymap,        "Gradient"),
    ("smoothgrad",            {"noise_scale": 50},      mnistutils.graymap,        "SmoothGrad"),
    ("integrated_gradients",  {},                       mnistutils.graymap,        "Integrated Gradients"),

    # Signal
    ("deconvnet",             {},                       mnistutils.bk_proj,        "Deconvnet"),
    ("guided_backprop",       {},                       mnistutils.bk_proj,        "Guided Backprop",),
    # ("pattern.net",           {},                       mnistutils.bk_proj,        "PatternNet"),

    # Interaction
    ("lrp.z",                 {},                       mnistutils.heatmap,         "LRP-Z"),
    ("lrp.epsilon",           {"epsilon": 1},           mnistutils.heatmap,         "LRP-Epsilon"),
    ]

# Create analyzers.
analyzers = []
print('Creating analyzer instances. ')
for method in methods:
    analyzer = innvestigate.create_analyzer(method[0],   # analysis method identifier
                                            model,       # model without softmax output
                                            **method[1]) # optional analysis parameters
    # some analyzers require additional training. For those
    analyzer.fit(X_train,
                 pattern_type='relu',
                 batch_size=256, verbose=1)
    analyzers.append(analyzer)

print('Running analyses.')

if mal_analyse:
	num_to_analyse = len(mal_data_X)
	Xa = mal_data_X
	Ya = true_labels
else:
	num_to_analyse = 10
	Xa = X_test
	Ya = Y_test_uncat
# Apply analyzers to trained model.
analysis = np.zeros([num_to_analyse, len(analyzers), 28, 28, 3])
text = []
for i in range(num_to_analyse):
    print('Image {}: '.format(i))
    t_start = time.time()
    image = Xa[i:i+1]
    
    # Predict label.
    presm = sess.run(logits, feed_dict={x:image})
    prob = sess.run(prediction, feed_dict={x:image})
    y_hat = prob.argmax()
    
    # Save prediction info:
    text.append(("%s" %label_to_class_name[Ya[i]],    # ground truth label
                 "%.2f" %presm.max(),             # pre-softmax logits
                 "%.2f" %prob.max(),              # probabilistic softmax output  
                 "%s" %label_to_class_name[y_hat] # predicted label
                ))
    
    for aidx, analyzer in enumerate(analyzers):
        
        is_input_analyzer = methods[aidx][0] == "input"
        # Analyze.
        a = analyzer.analyze(image)
        
        # Postprocess.
        if not is_input_analyzer:
            a = mnistutils.postprocess(a)
        a = methods[aidx][2](a)
        analysis[i, aidx] = a[0]
    t_elapsed = time.time() - t_start
    print('{:.4f}s'.format(t_elapsed))


# Plot the analysis.
grid = [[analysis[i, j] for j in range(analysis.shape[1])]
        for i in range(analysis.shape[0])]
label, presm, prob, pred = zip(*text)
row_labels_left = [('label: {}'.format(label[i]),'pred: {}'.format(pred[i])) for i in range(len(label))]
row_labels_right = [('logit: {}'.format(presm[i]),'prob: {}'.format(prob[i])) for i in range(len(label))]
col_labels = [''.join(method[3]) for method in methods]

if mal_analyse and args.mal:
    interpret_fig_name = 'mal_data_%s_%s' % (args.mal_obj, args.mal_strat)
elif mal_analyse:
    interpret_fig_name = 'mal_data_%s_ben' % (args.mal_obj)
elif not mal_analyse and args.mal:
    interpret_fig_name = 'ben_data_%s' % (args.mal_strat)
else:
    interpret_fig_name = 'ben_data_ben'

interpret_fig_name += '.pdf'

dir_name = gv.interpret_figs_dir_name

eutils.plot_image_grid(grid, row_labels_left, row_labels_right, col_labels, file_name=dir_name+interpret_fig_name)

