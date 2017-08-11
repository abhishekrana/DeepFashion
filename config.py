#!/usr/bin/python

### IMPORTS
from __future__ import print_function

import os
import numpy as np
import glob
import fnmatch
from random import randint
import shutil

import keras.optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.layers import Dropout, Flatten, Dense, Input
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping
from keras.optimizers import RMSprop, Adagrad
from keras.wrappers.scikit_learn import KerasClassifier
from keras import applications

from sklearn.utils import class_weight
from sklearn.model_selection import GridSearchCV

import PIL
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import skimage.data
import selectivesearch


import logging
import colored_traceback
colored_traceback.add_hook(always=True)
FORMAT = "[%(lineno)4s : %(funcName)-30s ] %(message)s"
#logging.basicConfig(level=logging.INFO, format=FORMAT)
logging.basicConfig(level=logging.DEBUG, format=FORMAT)


### GLOBALS
seed = 7
np.random.seed(seed)

dataset_path='dataset'
dataset_train_path=os.path.join(dataset_path, 'train')
dataset_val_path=os.path.join(dataset_path, 'validation')
dataset_test_path=os.path.join(dataset_path, 'test')
#dataset_src_path='fashion_data'
fashion_dataset_path='../Deep_Learning/DeepFashionV2/fashion_data/'
dataset_train_info=os.path.join(dataset_train_path, 'train_info.txt')
dataset_val_info=os.path.join(dataset_val_path, 'val_info.txt')

top_model_weights_path_save = 'output/bottleneck_fc_model.h5'
#top_model_weights_path_load = 'output/bottleneck_fc_model.h5'
top_model_weights_path_load = 'output/best-weights-011-1.3547.hdf5'

btl_path = 'bottleneck/'
btl_train_path = 'bottleneck/train/'
btl_val_path = 'bottleneck/validation/'

logs_path_name='logs'
output_path_name='output/'

prediction_dataset_path='dataset_prediction/images/'

img_width = 224             # For VGG16
img_height = 224            # For VGG16
img_channel = 3

epochs = 100
batch_size_train = 32
batch_size_predict = 32

predictions_class_weight=0.5
predictions_iou_weight=0.5
prediction_class_prob_threshold = 0.80
prediction_iou_threshold = 0.70
# prediction_class_prob_threshold = 0.50
# prediction_iou_threshold = 0.50


early_stopping_patience=500


input_image_width_threshold=500
input_image_height_threshold=500


# optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
# learn_rates = [0.001, 0.01, 0.1, 0.2, 0.3]
# decays = [[0.0]]
# momentums = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
# activations = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
# dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# weight_constraint = [1, 2, 3, 4, 5]
# batch_sizes = [5, 10, 20]
# epochs_grid = [50, 100, 150]
# init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
# neurons = [1, 5, 10, 15, 20, 25, 30]

#optimizer='Adagrad'    # 70% Inception
#optimizer='Adam'       # 44% Inception
#optimizer='RMSprop'    # 70% Inception
#optimizer='Adadelta'   # 65% Inception
#optimizer='Adamax'     # 45% Inception

#optimizer='RMSprop'    # 45% with 0.000001 lr
#optimizer='Adagrad'    # Starts from 23%
#optimizer='Adadelta'   # Straight line
#optimizer='Adam'       # 30%
#optimizer='Adamax'     # 31%
optimizer='SGD'         # 42%; might go above 50

learn_rate=0.001
decay=0.0
momentum=0.0
activation='relu'
dropout_rate=0.5



