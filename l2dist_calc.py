#########################
# Purpose: Calculates the distances between different weight vectors
########################
import numpy as np
import os
import argparse

import global_vars as gv

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


gv.init()
args = gv.args
print(gv.figures_dir_name)

for t in range(args.T):
    ben_weights = []
    ben_bias = []
    exist_flag = 0

    for i in range(args.k):
        if os.path.exists(gv.dir_name + 'ben_delta_%s_t%s.npy' % (i,t)):
            ben_delta_curr = np.load(gv.dir_name + 'ben_delta_%s_t%s.npy' % (i,t))
            weights_curr, bias_curr = collate_weights(ben_delta_curr)
            ben_weights.append(weights_curr)
            ben_bias.append(bias_curr)
            exist_flag = 1 
    ben_weights_max = 0.0
    ben_weights_min = 0.0
    mal_weights_max = 0.0
    mal_weights_min = 0.0
    if exist_flag == 1:
        count = 0
        avg_dist = 0.0
        for i in range(args.k-1):
            for j in range(i+1,args.k-1):
                # print('Distance[%s,%s]: %s' % (i,j,np.linalg.norm(ben_weights[i]-ben_weights[j])))
                curr_dist = np.linalg.norm(ben_weights[i]-ben_weights[j]) 
                count +=1
                avg_dist += np.linalg.norm(ben_weights[i]-ben_weights[j])
                if i == 0 and j == 1:
                    ben_weights_max = curr_dist
                    ben_weights_min = curr_dist
                else:
                    if curr_dist > ben_weights_max:
                        ben_weights_max = curr_dist
                    if curr_dist < ben_weights_min:
                        ben_weights_min = curr_dist
    avg_mal_dist = 0.0
    mal_count = 0
    if os.path.exists(gv.dir_name + 'mal_delta_t%s.npy' % t):
        mode = 'Malicious'
        # print (mode)
        mal_delta = np.load(gv.dir_name + 'mal_delta_t%s.npy' % t)
        # print('Directory found for iteration %s' % t)
        mal_weights_curr, mal_bias_curr = collate_weights(mal_delta)
        for i in range(args.k-1):
            curr_mal_dist = np.linalg.norm(ben_weights[i]-mal_weights_curr)
            if i == 0:
                mal_weights_max = curr_mal_dist
                mal_weights_min = curr_mal_dist
            else:
                if curr_mal_dist > mal_weights_max:
                    mal_weights_max = curr_mal_dist
                if curr_mal_dist < mal_weights_min:
                    mal_weights_min = curr_mal_dist
            avg_mal_dist += curr_mal_dist
            mal_count += 1
        print t, ben_weights_min, ben_weights_max, mal_weights_min, mal_weights_max
