#########################
# Purpose: Help with file input/output
########################
import os
import global_vars as gv
import numpy as np

from mnist import data_mnist
from keras.datasets import cifar10
from keras.utils import np_utils
from fmnist import load_fmnist
from census_utils import data_census


def file_write(write_dict, purpose='global_eval_loss'):
	f = open(gv.output_dir_name + gv.output_file_name +
	         '_' + purpose + '.txt', 'a')
	if write_dict['t'] == 1:
		d_count = 1
		for k, v in write_dict.iteritems():
			if d_count < len(write_dict):
				f.write(k + ',')
			else:
				f.write(k + '\n')
			d_count += 1
		d_count = 1
		for k, v in write_dict.iteritems():
			if d_count < len(write_dict):
				f.write(str(v) + ',')
			else:
				f.write(str(v) + '\n')
			d_count += 1
	elif write_dict['t'] != 1:
		d_count = 1
		for k, v in write_dict.iteritems():
			if d_count < len(write_dict):
				f.write(str(v) + ',')
			else:
				f.write(str(v) + '\n')
			d_count += 1
	f.close()


def data_setup():
	args = gv.args
	if 'MNIST' in args.dataset:
		X_train, Y_train, X_test, Y_test = data_mnist()
		Y_test_uncat = np.argmax(Y_test, axis=1)
		print('Loaded f/MNIST data')
	elif args.dataset == 'CIFAR-10':
		(X_train, Y_train_uncat), (X_test, Y_test_uncat) = cifar10.load_data()

		# Convert class vectors to binary class matrices.
		Y_train = np_utils.to_categorical(Y_train_uncat, gv.NUM_CLASSES)
		Y_test = np_utils.to_categorical(Y_test_uncat, gv.NUM_CLASSES)

		X_train = X_train.astype('float32')
		X_test = X_test.astype('float32')

		# subtract mean and normalize
		mean_image = np.mean(X_train, axis=0)
		X_train -= mean_image
		X_test -= mean_image
		X_train /= 128.
		X_test /= 128.
		print('Loaded CIFAR-10 data')
	elif args.dataset == 'census':
		X_train, Y_train, X_test, Y_test = data_census()
		Y_test_uncat = np.argmax(Y_test, axis=1)
		print Y_test
		print Y_test_uncat
		print('Loaded Census data')

	return X_train, Y_train, X_test, Y_test, Y_test_uncat


def mal_data_create(X_test, Y_test, Y_test_uncat):
	args = gv.args

	if args.mal_obj == 'all':
		mal_data_X = X_test
		mal_data_Y = Y_test
		true_labels = Y_test
	elif args.mal_obj == 'single':
		r = np.random.choice(len(X_test))
		print r
		mal_data_X = X_test[r:r + 1]
		allowed_targets = list(range(gv.NUM_CLASSES))
		print("Initial class: %s" % Y_test_uncat[r])
		true_labels = Y_test_uncat[r:r+1]
		allowed_targets.remove(Y_test_uncat[r])
		mal_data_Y = np.random.choice(allowed_targets)
		mal_data_Y = mal_data_Y.reshape(1,)
		print("Target class: %s" % mal_data_Y[0])
	elif 'multiple' in args.mal_obj:
		target_indices = np.random.choice(len(X_test), args.mal_num)
		mal_data_X = X_test[target_indices]
		print("Initial classes: %s" % Y_test_uncat[target_indices])
		true_labels = Y_test_uncat[target_indices]
		mal_data_Y = []
		for i in range(args.mal_num):
		    allowed_targets = list(range(gv.NUM_CLASSES))
		    allowed_targets.remove(Y_test_uncat[target_indices[i]])
		    mal_data_Y.append(np.random.choice(allowed_targets))
		mal_data_Y = np.array(mal_data_Y)
	return mal_data_X, mal_data_Y, true_labels

def mal_data_setup(X_test, Y_test, Y_test_uncat, gen_flag=True):
	args = gv.args

	data_path = 'data/mal_X_%s_%s.npy' % (args.dataset, args.mal_obj)

	print data_path
	
	if not os.path.exists('data/mal_X_%s_%s.npy' % (args.dataset, args.mal_obj)):
		if gen_flag:
			mal_data_X, mal_data_Y, true_labels = mal_data_create(X_test, Y_test, Y_test_uncat)
		else:
			raise ValueError('Tried to generate mal data when disallowed')
	else:
		mal_data_X = np.load('data/mal_X_%s_%s.npy' % (args.dataset, args.mal_obj))
		mal_data_Y = np.load('data/mal_Y_%s_%s.npy' % (args.dataset, args.mal_obj))
		true_labels = np.load('data/true_labels_%s_%s.npy' % (args.dataset, args.mal_obj))
		print("Initial classes: %s" % true_labels)
		print("Target classes: %s" % mal_data_Y)
   	
	return mal_data_X, mal_data_Y, true_labels

