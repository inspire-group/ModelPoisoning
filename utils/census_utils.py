#########################
# Purpose: Utility functions for attacks on the census data
########################

import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout
import scipy.io as sio
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from keras.utils import np_utils

import global_vars as gv

def data_census():
	data_dir = '/home/data/census/'
	traindata = sio.loadmat(data_dir+'CensusTrain.mat')['traindata']
	testdata = sio.loadmat(data_dir+'CensusTest.mat')['testdata']

	mask = ~np.any(np.isnan(traindata),axis=1)
	Xtr = traindata[mask,:-1]
	ytr = traindata[mask,-1]
	mask = ~np.any(np.isnan(testdata),axis=1)
	Xte = testdata[mask,:-1]
	yte = testdata[mask,-1]

	scaler = MinMaxScaler()
	Xtr = scaler.fit_transform(Xtr)
	Xte = scaler.transform(Xte)

	Xtr = Xtr[:30160]
	ytr = ytr[:30160]

	print(ytr.shape)
	ytr = np_utils.to_categorical(ytr, gv.NUM_CLASSES)
	yte = np_utils.to_categorical(yte, gv.NUM_CLASSES)
	# lb = LabelBinarizer()
	# ytr = lb.fit_transform(ytr)
	# yte = lb.transform(yte)
	print(ytr.shape)

	return Xtr, ytr, Xte, yte

def census_model_1():
	main_input = Input(shape=(gv.DATA_DIM,), name='main_input')
	x = Dense(256, use_bias=True, activation='relu')(main_input)
	x = Dropout(0.5)(x)
	x = Dense(256, use_bias=True, activation='relu')(main_input)
	x = Dropout(0.5)(x)
	# main_output = Dense(1)(x)
	main_output = Dense(gv.NUM_CLASSES)(x)

	model = Model(inputs=main_input, outputs=main_output)

	return model
