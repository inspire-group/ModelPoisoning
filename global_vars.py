#########################
# Purpose: Sets up global variables to be used throughout
########################

import argparse
import os
import tensorflow as tf


def dir_name_fn(args):

    # Setting directory name to store computed weights
    dir_name = 'weights/%s/model_%s/%s/k%s_E%s_B%s_C%1.0e_lr%.1e' % (
        args.dataset, args.model_num, args.optimizer, args.k, args.E, args.B, args.C, args.eta)
    # dir_name = 'weights/k{}_E{}_B{}_C{%e}_lr{}'
    output_file_name = 'output'

    output_dir_name = 'output_files/%s/model_%s/%s/k%s_E%s_B%s_C%1.0e_lr%.1e' % (
        args.dataset, args.model_num, args.optimizer, args.k, args.E, args.B, args.C, args.eta)

    figures_dir_name = 'figures/%s/model_%s/%s/k%s_E%s_B%s_C%1.0e_lr%.1e' % (
        args.dataset, args.model_num, args.optimizer, args.k, args.E, args.B, args.C, args.eta)

    interpret_figs_dir_name = 'interpret_figs/%s/model_%s/%s/k%s_E%s_B%s_C%1.0e_lr%.1e' % (
        args.dataset, args.model_num, args.optimizer, args.k, args.E, args.B, args.C, args.eta)

    if args.gar != 'avg':
        dir_name = dir_name + '_' + args.gar
        output_file_name = output_file_name + '_' + args.gar
        output_dir_name = output_dir_name + '_' + args.gar
        figures_dir_name = figures_dir_name + '_' + args.gar
        interpret_figs_dir_name = interpret_figs_dir_name + '_' + args.gar

    if args.lr_reduce:
        dir_name += '_lrr'
        output_dir_name += '_lrr'
        figures_dir_name += '_lrr'

    if args.steps is not None:
        dir_name += '_steps' + str(args.steps)
        output_dir_name += '_steps' + str(args.steps)
        figures_dir_name += '_steps' + str(args.steps)

    if args.mal:
        if 'multiple' in args.mal_obj:
            args.mal_obj = args.mal_obj + str(args.mal_num)
        if 'dist' in args.mal_strat:
            args.mal_strat += '_rho' + '{:.2E}'.format(args.rho)
        if args.E != args.mal_E:
            args.mal_strat += '_ext' + str(args.mal_E)
        if args.mal_delay > 0:
            args.mal_strat += '_del' + str(args.mal_delay)
        if args.ls != 1:
            args.mal_strat += '_ls' + str(args.ls)
        if 'data_poison' in args.mal_strat:
            args.mal_strat += '_reps' + str(args.data_rep)
        if 'no_boost' in args.mal_strat or 'data_poison' in args.mal_strat:
            args.mal_strat = args.mal_strat
        else:
            # if 'auto' not in args.mal_strat:
            args.mal_strat += '_boost'+ str(args.mal_boost)
        output_file_name += '_mal_' + args.mal_obj + '_' + args.mal_strat
        dir_name += '_mal_' + args.mal_obj + '_' + args.mal_strat

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    if not os.path.exists(output_dir_name):
        os.makedirs(output_dir_name)

    if not os.path.exists(figures_dir_name):
        os.makedirs(figures_dir_name)

    if not os.path.exists(interpret_figs_dir_name):
        os.makedirs(interpret_figs_dir_name)

    dir_name += '/'
    output_dir_name += '/'
    figures_dir_name += '/'
    interpret_figs_dir_name += '/'

    print dir_name
    print output_file_name

    return dir_name, output_dir_name, output_file_name, figures_dir_name, interpret_figs_dir_name


def init():
    # Reading in arguments for the run
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='fMNIST',
                        help="dataset to be used")
    parser.add_argument("--model_num", type=int,
                        default=0, help="model to be used")
    parser.add_argument("--optimizer", default='adam',
                        help="optimizer to be used")
    parser.add_argument("--eta", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--k", type=int, default=10, help="number of agents")
    parser.add_argument("--C", type=float, default=1.0,
                        help="fraction of agents per time step")
    parser.add_argument("--E", type=int, default=5,
                        help="epochs for each agent")
    parser.add_argument("--steps", type=int, default=None,
                        help="GD steps per agent")
    parser.add_argument("--T", type=int, default=40, help="max time_steps")
    parser.add_argument("--B", type=int, default=50, help="agent batch size")
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--lr_reduce", action='store_true')
    parser.add_argument("--mal", action='store_true')
    parser.add_argument("--mal_obj", default='single',
                        help='Objective for malicious agent')
    parser.add_argument("--mal_strat", default='converge',
                        help='Strategy for malicious agent')
    parser.add_argument("--mal_num", type=int, default=1,
                        help='Objective for simultaneous targeting')
    parser.add_argument("--mal_delay", type=int, default=0,
                        help='Delay for wait till converge')
    parser.add_argument("--mal_boost", type=float, default=10.0,
                        help='Boosting factor for alternating minimization attack')
    parser.add_argument("--mal_E", type=float, default=5,
                        help='Benign training epochs for malicious agent')
    parser.add_argument("--ls", type=int, default=1,
                        help='Training steps for each malicious step')
    parser.add_argument("--gar", type=str, default='avg',
                        help='Gradient Aggregation Rule', choices=['avg', 'krum', 'coomed'])
    parser.add_argument("--rho", type=float, default=1e-4,
                        help='Weighting factor for distance constraints')
    parser.add_argument("--data_rep", type=float, default=10,
                        help='Data repetitions for data poisoning')
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=None,
                        help='GPUs to run on')


    global args
    args = parser.parse_args()
    print args

    if args.mal:
        global mal_agent_index
        mal_agent_index = args.k - 1

    global gpu_ids
    if args.gpu_ids is not None:
        gpu_ids = args.gpu_ids
    else:
        gpu_ids = [3,4]
    global num_gpus
    num_gpus = len(gpu_ids)

    global max_agents_per_gpu

    global IMAGE_ROWS, IMAGE_COLS, NUM_CHANNELS, NUM_CLASSES, BATCH_SIZE

    global max_acc

    if 'MNIST' in args.dataset:
        IMAGE_ROWS = 28
        IMAGE_COLS = 28
        NUM_CHANNELS = 1
        NUM_CLASSES = 10
        BATCH_SIZE = 100
        if args.dataset == 'MNIST':
            max_acc = 99.0
        elif args.dataset == 'fMNIST':
            max_acc = 91.0
        max_agents_per_gpu = 8
        mem_frac = 0.05
    elif args.dataset == 'census':
        global DATA_DIM
        DATA_DIM = 104
        BATCH_SIZE = 50
        NUM_CLASSES = 2
        max_acc = 85.0
        max_agents_per_gpu = 8
        mem_frac = 0.05

    if max_agents_per_gpu < 1:
        max_agents_per_gpu = 1

    global gpu_options
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_frac)

    global dir_name, output_dir_name, output_file_name, figures_dir_name, interpret_figs_dir_name

    dir_name, output_dir_name, output_file_name, figures_dir_name, interpret_figs_dir_name = dir_name_fn(
        args)
