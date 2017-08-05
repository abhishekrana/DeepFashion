#!/usr/bin/python

### IMPORTS
from __future__ import print_function

import numpy as np
import fnmatch
import os

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

import glob
from PIL import Image
from random import randint

import logging
FORMAT = "[%(lineno)4s : %(funcName)-30s ] %(message)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT)



### GLOBALS
# TODO: Test with 224 for VGG16
img_width = 224
img_height = 224
class_names=[]
#batch_size = 32
#epochs = 50

top_model_weights_path = 'output/bottleneck_fc_model.h5'

dataset_path = 'dataset'
dataset_train_path=os.path.join(dataset_path, 'train')
dataset_val_path=os.path.join(dataset_path, 'validation')
dataset_test_path=os.path.join(dataset_path, 'test')


### FUNCTIONS ###

# Sorted subdirectories list
def get_subdir_list(path):
    names=[]
    for name in sorted(os.listdir(path)):
        if os.path.isdir(os.path.join(path, name)):
            names.append(name)
    logging.info('names {}'.format(names))
    return names


def get_optimizer(optimizer='Adagrad', lr=None, decay=0.0, momentum=0.0):

    if optimizer == 'SGD':
        if lr is None:
            lr = 0.01
        optimizer_mod = keras.optimizers.SGD(lr=lr, momentum=momentum, decay=decay, nesterov=False)

    elif optimizer == 'RMSprop':
        if lr is None:
            lr = 0.001
        optimizer_mod = keras.optimizers.RMSprop(lr=lr, rho=0.9, epsilon=1e-08, decay=decay)

    elif optimizer == 'Adagrad':
        if lr is None:
            lr = 0.01
        optimizer_mod = keras.optimizers.Adagrad(lr=lr, epsilon=1e-08, decay=decay)

    elif optimizer == 'Adadelta':
        if lr is None:
            lr = 1.0
        optimizer_mod = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)

    elif optimizer == 'Adam':
        if lr is None:
            lr = 0.001
        optimizer_mod = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    elif optimizer == 'Adamax':
        if lr is None:
            lr = 0.002
        optimizer_mod = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    elif optimizer == 'Nadam':
        if lr is None:
            lr = 0.002
        optimizer_mod = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

    else:
        logging.error('Unknown optimizer {}'.format(optimizer))
        exit(1)

    # logging.debug('lr {}'.format(lr))
    # logging.debug('momentum {}'.format(momentum))
    # logging.debug('decay {}'.format(decay))
    # logging.debug('optimizer_mod {}'.format(optimizer_mod))

    return optimizer_mod, lr


# INPUT:
#           VGG16 - block5_pool (MaxPooling2D) (None, 7, 7, 512)
# OUTPUT:
#           Branch1 - Class Prediction
#           Branch2 - IOU Prediction

# NOTE: Both models in create_model_train() and  create_model_predict() should be exaclty same
def create_model_train(input_shape, optimizer='Adagrad', learn_rate=None, decay=0.0, momentum=0.0, activation='relu', dropout_rate=0.5):
    logging.debug('input_shape {}'.format(input_shape))
    logging.debug('input_shape {}'.format(type(input_shape)))

    # Optimizer
    optimizer, learn_rate = get_optimizer(optimizer, learn_rate, decay, momentum)

    # input_shape = (7, 7, 512)                                                                     # VGG bottleneck layer - block5_pool (MaxPooling2D)

    inputs = Input(shape=(input_shape))
    # x_common = Dense(256, activation='relu')(inputs)

    ## Model Classification
    #x = Flatten()(x_common)
    x = Flatten()(inputs)
    x = Dense(256, activation='tanh')(x)
    x = Dropout(dropout_rate)(x)
    predictions_class = Dense(len(class_names), activation='softmax', name='predictions_class')(x)


    ## Model (Regression) IOU score
    #x = Flatten()(x_common)
    x = Flatten()(inputs)
    x = Dense(256, activation='tanh')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(256, activation='tanh')(x)
    x = Dropout(dropout_rate)(x)
    predictions_iou = Dense(1, activation='sigmoid', name='predictions_iou')(x)


    # This creates a model that includes the Input layer and three Dense layers
    model = Model(inputs=inputs, outputs=[predictions_class, predictions_iou])


    logging.debug('model summary {}'.format(model.summary()))


    # Compile
    model.compile(optimizer=optimizer,
                  loss={'predictions_class': 'sparse_categorical_crossentropy', 'predictions_iou': 'mean_squared_error'}, metrics=['accuracy'],
                  loss_weights={'predictions_class': 0.5, 'predictions_iou': 0.5})
                  #loss_weights={'predictions_class': 0.5, 'predictions_iou': 0.5})
                  #loss={'predictions_class': 'sparse_categorical_crossentropy', 'predictions_iou': 'logcosh'}, metrics=['accuracy'],

    logging.info('optimizer:{}  learn_rate:{}  decay:{}  momentum:{}  activation:{}  dropout_rate:{}'.format(
        optimizer, learn_rate, decay, momentum, activation, dropout_rate))

    return model


def save_bottleneck():

    ## Build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet', input_shape=(img_width, img_height, 3))                               # exclude 3 FC layers on top of network


    ## Variables
    batch_size = 32

    # Hardcoding
    input_shape = (img_width, img_height,3)
    logging.debug('input_shape {}'.format(input_shape))

    for train_val in ['train', 'validation']:

        with open('bottleneck/btl_' + train_val + '.txt', 'w') as f_image:
            for class_name in class_names:
                dataset_train_class_path = os.path.join(dataset_path, train_val, class_name)
                logging.debug('dataset_train_class_path {}'.format(dataset_train_class_path))

                images_list = []
                images_name_list = []

                images_path_name = sorted(glob.glob(dataset_train_class_path + '/*.jpg'))
                logging.debug('images_path_name {}'.format(len(images_path_name)))
                # logging.debug('images_path_name {}'.format(images_path_name))

                for index, image in enumerate(images_path_name):
                    # logging.debug('images_path_name {}'.format(images_path_name))
                    # logging.debug('image {}'.format(image))

                    img = Image.open(image)

                    img = img.resize((img_width, img_height))
                    # logging.debug('img {}'.format(img))

                    img=np.array(img).astype(np.float32)

                    # TODO: no -ve sign in training; check
                    img[:,:,0] -= 103.939
                    img[:,:,1] -= 116.779
                    img[:,:,2] -= 123.68

                    # img = np.expand_dims(img, 0)
                    # images_list = img

                    current_batch_size = len(images_list)
                    # logging.debug('current_batch_size {}'.format(current_batch_size))

                    images_list.append(img)
                    image_name = image.split('/')[-1].split('.jpg')[0]
                    images_name_list.append(image)

                    # TODOD: Skipping images which do not form a batch at the end of the class
                    if (current_batch_size < batch_size-1):
                        continue


                    images_list_arr = np.array(images_list)

                    X = images_list_arr

                    bottleneck_features_train_class = model.predict(X, batch_size)
                    # bottleneck_features_train_class = model.predict(X, nb_train_class_samples // batch_size)

                    btl_save_file_name = 'bottleneck/'+train_val+'/btl_'+train_val+'_' + class_name + '.' + str(index).zfill(7) + '.npy'
                    logging.debug('btl_save_file_name {}'.format(btl_save_file_name))
                    np.save(open(btl_save_file_name, 'w'), bottleneck_features_train_class)

                    for name in images_name_list:
                        f_image.write(str(name) + '\n')

                    images_list = []
                    images_name_list = []



def train_model():

    ## Variables
    btl_train_path = 'bottleneck/train/'
    btl_val_path = 'bottleneck/validation/'
    batch_size = 32
    epochs = 100
    input_shape = (img_width, img_height, 3)

    ## Build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)                               # exclude 3 FC layers on top of network


    # logging.debug('Loading bottleneck_features_train...')
    # input_shape = train_data.shape[1:]


    # Bottleneck files list
    btl_train_names = sorted(glob.glob(btl_train_path + '/*.npy'))
    # logging.debug('btl_train_names {}'.format(btl_train_names))
    btl_val_names = sorted(glob.glob(btl_val_path + '/*.npy'))
    # logging.debug('btl_val_names {}'.format(btl_val_names))


    ## Train
    btl_train_list = []
    train_labels_class = []
    train_labels_iou = []
    with open('bottleneck/btl_train.txt') as f_btl_train:
        btl_train_list = f_btl_train.readlines()
        # logging.debug('btl_train_list {}'.format(btl_train_list))

    for btl_train_image in btl_train_list:
        train_labels_class.append(btl_train_image.split('/')[2])
        iou_value = np.round(np.float( btl_train_image.split('_')[-1].split('.jpg')[0] ), 2)
        train_labels_iou.append(iou_value)
        # logging.debug('val {}'.format(val))

    # logging.debug('class_names {}'.format(class_names))
    # logging.debug('train_labels_class {}'.format(train_labels_class))
    train_labels_class_int = []
    for index, class_name in enumerate(train_labels_class):
        train_labels_class_int.append(class_names.index(class_name))
    train_labels_class = train_labels_class_int
    logging.debug('train_labels_class {}'.format(train_labels_class))

    train_labels_class = np.array(train_labels_class)
    train_labels_iou = np.array(train_labels_iou)
    logging.debug('train_labels_iou {}'.format(train_labels_iou))
    logging.debug('train_labels_iou {}'.format(type(train_labels_iou)))
    logging.debug('train_labels_class {}'.format(type(train_labels_class)))
    logging.debug('train_labels_class {}'.format((train_labels_class.shape)))


    # Create train set
    train_data = np.load(open(btl_train_names[0]))
    for index, btl_name in enumerate(btl_train_names[1:]):
        # logging.debug('btl_name {}'.format(btl_name))
        temp = np.load(open(btl_name))
        train_data = np.concatenate((train_data, temp), axis=0)

    train_data = np.array(train_data)
    logging.debug('train_data {}'.format(train_data.shape))


    # Validation
    btl_val_list = []
    val_labels_class = []
    val_labels_iou = []
    with open('bottleneck/btl_validation.txt') as f_btl_val:
        btl_val_list = f_btl_val.readlines()
        # logging.debug('btl_val_list {}'.format(btl_val_list))

    for btl_val_image in btl_val_list:
        val_labels_class.append(btl_val_image.split('/')[2])
        val = np.round(np.float( btl_val_image.split('_')[-1].split('.jpg')[0] ), 2)
        val_labels_iou.append(val)
        # logging.debug('val {}'.format(val))

    logging.debug('val_labels_class {}'.format(val_labels_class))
    val_labels_class_int = []
    for index, class_name in enumerate(val_labels_class):
        val_labels_class_int.append(class_names.index(class_name))
    val_labels_class = val_labels_class_int
    logging.debug('val_labels_class {}'.format(val_labels_class))

    val_labels_class = np.array(val_labels_class)
    logging.debug('val_labels_class {}'.format(val_labels_class))
    val_labels_iou = np.array(val_labels_iou)
    logging.debug('val_labels_iou {}'.format(val_labels_iou))
    logging.debug('val_labels_iou {}'.format(type(val_labels_iou)))
    logging.debug('val_labels_class {}'.format(type(val_labels_class)))
    logging.debug('val_labels_class {}'.format(val_labels_class.shape))

    # Create validation set
    val_data = np.load(open(btl_val_names[0]))
    for index, btl_name in enumerate(btl_val_names[1:]):
        temp = np.load(open(btl_name))
        val_data = np.concatenate((val_data, temp), axis=0)

    val_data = np.array(val_data)
    logging.debug('val_data {}'.format(val_data.shape))






    ## Callbacks
    filename = 'output/model_train.csv'
    csv_log = CSVLogger(filename, separator=' ', append=False)

    early_stopping = EarlyStopping(
        monitor='loss', patience=500, verbose=1, mode='min')                                         # Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,dense_4_loss,dense_2_acc,dense_2_loss,dense_4_acc

    #filepath = "output/best-weights-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"
    filepath = "output/best-weights-{epoch:03d}-{val_loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
        save_best_only=True, save_weights_only=False, mode='min', period=1)                         # min because we are monitoring val_loss that should decrease

    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

    callbacks_list = [csv_log, early_stopping, checkpoint, tensorboard]
    logging.debug('callbacks_list {}'.format(callbacks_list))


    # Generate weights based on images count for each class
    class_weight_val = class_weight.compute_class_weight('balanced', np.unique(train_labels_class), train_labels_class)
    logging.info('class_weight_val {}'.format(class_weight_val))


    optimizer='Adagrad'
    learn_rate=0.001
    decay=0.0
    momentum=0.0
    activation='relu'
    dropout_rate=0.5

    input_shape = train_data.shape[1:]
    logging.debug('input_shape {}'.format(input_shape))
    model = create_model_train(input_shape, optimizer, learn_rate, decay, momentum, activation, dropout_rate)

    logging.debug('train_labels_iou {}'.format(train_labels_iou.shape))
    logging.debug('train_labels_class {}'.format(train_labels_class.shape))
    logging.debug('train_data {}'.format(train_data.shape))

    # TODO: class_weight_val wrong
    model.fit(train_data, [train_labels_class, train_labels_iou],
            class_weight=[class_weight_val, class_weight_val],                                      # dictionary mapping classes to a weight value, used for scaling the loss function (during training only).
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(val_data, [val_labels_class, val_labels_iou]),
            callbacks=callbacks_list)

    # TODO: These are not the best weights
    model.save_weights(top_model_weights_path)





### MAIN ###
seed = 7
np.random.seed(seed)

class_names = get_subdir_list(dataset_train_path)
logging.debug('class_names {}'.format(class_names))

save_bottleneck()
train_model()                                                                                       # Save weights at output/bottleneck_fc_model.h5


