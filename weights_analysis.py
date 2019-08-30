#########################
# Purpose: Generates weight distribution plots
########################

import numpy as np
import os
import argparse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import collections
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.stats import wasserstein_distance

import global_vars as gv

top_k = 10


def two_d_convert(indices, values, cutoffs, shape_list):
    two_d_indices = collections.OrderedDict()
    for i, item in enumerate(indices):
        layer_nums = np.where(item > cutoffs)[0]
        if list(layer_nums):
            layer_num = layer_nums[-1]
            curr_layer_index = item - cutoffs[layer_num]
            layer_num = layer_nums[-1] + 1
        else:
            layer_num = 0
            curr_layer_index = item
        curr_shape = shape_list[layer_num]
        row_index = curr_layer_index / curr_shape[1]
        column_index = curr_layer_index % curr_shape[1]
        two_d_indices[str(top_k - i)] = [item, (layer_num, row_index,
                                                column_index), values[i]]
    return two_d_indices


def one_d_convert(indices, cutoffs, shape_list):
    one_d_indices = {}
    for item in indices:
        layer_num = np.where(item > cutoffs)[0]
        curr_layer_index = item - cutoffs[layer_num]
        one_d_indices[str(layer_num)] = curr_layer_index
    return one_d_indices


def model_shape_size(delta_curr):
    shape_w = []
    shape_b = []
    size_w = []
    size_b = []
    for l in range(len(delta_curr)):
        layer_shape = delta_curr[l].shape
        size = 1
        for item in layer_shape:
            size *= item
        if l % 2 == 0:
            size_w.append(size)
            shape_w.append(layer_shape)
        elif (l + 1) % 2 == 0:
            size_b.append(size)
            shape_b.append(layer_shape)
    return shape_w, shape_b, size_w, size_b


def collate_weights(delta_curr):
    for l in range(len(delta_curr)):
        flat_layer = delta_curr[l].flatten()
        if l == 0:
            delta_curr_w = flat_layer
        elif l == 1:
            delta_curr_b = flat_layer
        elif l % 2 == 0:
            delta_curr_w = np.concatenate(
                (delta_curr_w, flat_layer))
        elif (l + 1) % 2 == 0:
            delta_curr_b = np.concatenate(
                (delta_curr_b, flat_layer))
    return delta_curr_w, delta_curr_b


def top_k_finder(delta_curr, t, mode=None, step=None):
    delta_curr_w, delta_curr_b = collate_weights(delta_curr)

    delta_curr_w_nz = delta_curr_w[np.where(np.abs(delta_curr_w) > 1e-7)]

    manual_bins = np.linspace(-0.2,0.2,500)
    # print manual_bins

    hist, bins = np.histogram(delta_curr_w_nz, bins=manual_bins)
    # hist, bins = np.histogram(delta_curr_w_nz, bins=500)

    x = (bins[:-1] + bins[1:])/2

    width = 2*abs(x[0]-x[1])

    if mode == 'Malicious':
        # ax_mal.bar(x, hist, zs=t, zdir='y', width=width)
        mal_bars = ax_ben.bar(x, hist, width=width, color='red',alpha=0.5)
        mal_max.append(np.amax(delta_curr_w))
        mal_min.append(np.amin(delta_curr_w))
    elif mode == 'Benign':
        ben_bars = ax_ben.bar(x, hist, width=width, alpha=0.3)
        ben_max.append(np.amax(delta_curr_w))
        ben_min.append(np.amin(delta_curr_w))

    min_signed_weight = np.amin(delta_curr_w)
    max_signed_weight = np.amax(delta_curr_w)

    print('Range of weights is from %s to %s' %
          (min_signed_weight, max_signed_weight))
    abs_weights = np.abs(delta_curr_w)

    min_abs_weight = np.amin(abs_weights)
    max_abs_weight = np.amax(abs_weights)

    print('Range of absolute weights is from %s to %s' %
          (min_abs_weight, max_abs_weight))
    print('No. of large weights is %s' %
          (len(np.where(abs_weights > 1e-6)[0])))

    ind_weights = np.argpartition(abs_weights, -top_k)[-top_k:]
    top_k_weights = abs_weights[ind_weights]

    ind_weights_sorted = ind_weights[np.argsort(top_k_weights)]
    top_k_weights_sorted = top_k_weights[np.argsort(top_k_weights)]

    ind_proper = two_d_convert(
        ind_weights_sorted, top_k_weights_sorted, cutoffs_w, shape_w)
    # print ind_proper

    if mode == 'Malicious':
        return hist, mal_bars
    elif mode == 'Benign':
        return hist, ben_bars


gv.init()
args = gv.args
print(gv.figures_dir_name)
global_weights_0 = np.load(
    gv.dir_name + 'global_weights_t%s.npy' % 0)

shape_w, shape_b, size_w, size_b = model_shape_size(global_weights_0)

cutoffs_w = np.cumsum(np.asarray(size_w))
cutoffs_b = np.cumsum(np.asarray(size_b))

mal_max = []
mal_min = []
ben_max = []
ben_min = []



fig = plt.figure()
ax_ben = fig.add_subplot(111)


ax_ben.set_xlabel('Weight values')


ax_ben.grid(False)

final_t = 0
for t in range(4,5):
    print('Time Step %s' % t)
    if not args.mal:
        global_weights_curr = np.load(
            gv.dir_name + 'global_weights_t%s.npy' % t)
        global_weights_next = np.load(
            gv.dir_name + 'global_weights_t%s.npy' % (t + 1))
        global_delta = global_weights_next - global_weights_curr

        top_k_finder(global_delta, t)
    else:
        mode = 'Benign'
        print (mode)
        ben_flag = 0
        
        if os.path.exists(gv.dir_name + 'ben_delta_sample%s.npy' % t):
            ben_delta_curr = np.load(gv.dir_name + 'ben_delta_sample%s.npy' % t)
            weights_curr, bias_curr = collate_weights(ben_delta_curr)
            ben_delta_hist_1d, ben_bars = top_k_finder(ben_delta_curr, t, mode)
            ben_flag = 1
            final_t = t

        if os.path.exists(gv.dir_name + 'mal_delta_t%s.npy' % t):
            mode = 'Malicious'
            print (mode)
            mal_delta = np.load(gv.dir_name + 'mal_delta_t%s.npy' % t)
            print('Directory found for iteration %s' % t)
            mal_weights_curr, mal_bias_curr = collate_weights(mal_delta)
            mal_delta_hist_1d, mal_bars = top_k_finder(mal_delta, t, mode)
            mal_flag = 1

ax_ben.legend((ben_bars[0],mal_bars[0]),('Benign','Malicious'))


plt.savefig(gv.figures_dir_name + 'hist_delta_2d_%s_%s.pdf' %
            (args.mal_obj, args.mal_strat),format='pdf')
plt.savefig(gv.figures_dir_name + 'hist_delta_2d_%s_%s.png' %
            (args.mal_obj, args.mal_strat),format='png')        

plt.clf()
