'''
1. Instantiate the convolutional part of the model, everything up to the fully-connected layers.
2. Run this model on our training and validation data once, recording the output (the "bottleneck features"
   from th VGG16 model: the last activation maps before the fully-connected layers) in two numpy arrays.
3. Train a small fully-connected model on top of the stored features.

Storing the features offline rather than adding our fully-connected model directly on top of a frozen
convolutional base and running the whole thing, is computational effiency.
Running VGG16 is expensive and we want to only do it once.

* Note that this prevents us from using data augmentation.

'''

# IMPORTS
from __future__ import print_function

import os
import cv2                  # load and preprocess dataset
import numpy as np              # reshape images, ...
import matplotlib.pyplot as plt
import fnmatch

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

from keras import backend as K
K.set_image_dim_ordering('tf')
# tensorflow (N, height or rows , width or cols     , channels)
# theano     (N, channels       , height or rows    , width or cols)

from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.models import model_from_json
from keras.models import load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

from keras import applications
from keras.datasets import cifar10


import logging
FORMAT = "[%(lineno)4s : %(funcName)-30s ] %(message)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT)


import keras
import tensorflow as tf
logging.debug('keras {}'.format(keras.__version__))
logging.debug('tensorflow {}'.format(tf.__version__))


# GLOBALS

# VGG input size (3,224,224)
# DeepFahion dataset: The long side of images are resized to 300;
img_height = 150         # rows
img_width = 150          # cols
img_channels=3

#dataset_path = 'dataset_dogs_cats'
dataset_path = 'dataset'
model_saved_path = 'output/model.h5'
dataset_train_path=os.path.join(dataset_path, 'train')
dataset_val_path=os.path.join(dataset_path, 'validation')
dataset_test_path=os.path.join(dataset_path, 'test')

data_augmentation = True
num_classes = 2
batch_size = 32
epochs = 2

# FUNCTIONS

if not os.path.exists('output'):
    os.makedirs('output')

if not os.path.exists('logs'):
    os.makedirs('logs')

def dataset_images_count(dataset_split_name):
    # Only matches .jpg files
    matches = []
    path = ''

    if dataset_split_name == 'train':
        path=os.path.join(dataset_path, 'train')
    elif dataset_split_name == 'val':
        path=os.path.join(dataset_path, 'validation')
    elif dataset_split_name == 'test':
        path=os.path.join(dataset_path, 'test')
    else:
        logging.error('Unknown dataset_split_name')

    logging.debug('path {}'.format(path))
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*.jpg'):
            matches.append(os.path.join(root, filename))

    # logging.debug('matches {}'.format(matches))

    images_count = len(matches)
    return images_count


def load_and_preprocess_data_generator():

        # Issue: Predict(correct) and predict generator(wrong) resulting in very different results using
        # keras.applications.vgg16.
        # Solution: Must not use rescale=1./255 in ImageDataGenerator. VGG was trained on demeaned but not rescaled data.
        # Initiate the train and test generators with data Augumentation
        # train_datagen = ImageDataGenerator(rescale = 1./255, shear_range=0.2, horizontal_flip = True, fill_mode = "nearest",
        #                                    zoom_range = 0.3, width_shift_range = 0.3, height_shift_range=0.3,
        #                                    rotation_range=30)

        # test_datagen = ImageDataGenerator(rescale = 1./255, shear_range=0.2, horizontal_flip = True, fill_mode = "nearest",
        #                                   zoom_range = 0.3, width_shift_range = 0.3, height_shift_range=0.3,
        #                                   rotation_range=30)


        # VGG image prepreocessing
        train_datagen = ImageDataGenerator(rescale=1., featurewise_center=True) #(rescale=1./255)
        train_datagen.mean=np.array([103.939, 116.779, 123.68],dtype=np.float32).reshape(3,1,1)
        train_generator = train_datagen.flow_from_directory(dataset_train_path,
                                                            target_size = (img_height, img_width),
                                                            batch_size = batch_size,
                                                            class_mode = "binary")                  # None means yield batches of data, no labels
                                                            # TODO
                                                            #class_mode = "categorical")

        validation_datagen = ImageDataGenerator(rescale=1., featurewise_center=True) #(rescale=1./255)
        validation_datagen.mean=np.array([103.939, 116.779, 123.68],dtype=np.float32).reshape(3,1,1)
        validation_generator = validation_datagen.flow_from_directory(dataset_val_path,
                                                                target_size = (img_height, img_width),
                                                                batch_size = batch_size,
                                                                class_mode = "binary")
                                                                # TODO
                                                                #class_mode = "categorical")

        # TODO: HARDCODING
        input_shape = (img_width, img_height, img_channels)
        logging.debug('input_shape {}'.format(input_shape))

        return train_generator, validation_generator, input_shape


def model_create(input_shape, num_classes):
        logging.debug('input_shape {}'.format(input_shape))

        model = Sequential()

        model.add(Conv2D(32, (3, 3), border_mode='same', input_shape=input_shape))
        model.add(Activation('relu'))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        # use binary_crossentropy if has just 2 prediction yes or no
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

        return model

def model_vgg_create(input_shape, num_classes):

    logging.debug('input_shape {}'.format(input_shape))
    #model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (256, 256, 3))
    model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (input_shape))        #  input_shape (128, 128, 1)
                                                                                                            #  input_shape (128, 128, 3)

    # Freeze the layers which you don't want to train. Freezing the first 5 layers.
    for layer in model.layers[:5]:
        layer.trainable = False

    # Adding custom Layers
    x = model.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    # Creating the final model
    model_final = Model(inputs = model.input, outputs = predictions)

    # Compile the model
    # opt = RMSprop(lr=0.0001, decay=1e-6)
    opt = SGD(lr=0.0001, momentum=0.9)
    model_final.compile(loss = "categorical_crossentropy", optimizer = opt, metrics=["accuracy"])

    return model_final


def model_train(model, train_generator, validation_generator):

        # Callbacks
        filename = 'output/model_train.csv'
        csv_log = CSVLogger(filename, separator=' ', append=False)

        early_stopping = EarlyStopping(
            monitor='val_loss', patience=50, verbose=1, mode='min')

        filepath = "output/best-weights-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
            save_best_only=True, save_weights_only=False, mode='min', period=1)                     # min because we are monitoring val_loss that should decrease

        tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

        callbacks_list = [csv_log, early_stopping, checkpoint, tensorboard]
        logging.debug('callbacks_list {}'.format(callbacks_list))

        images_count_train = dataset_images_count('train')
        logging.debug('images_count_train {}'.format(images_count_train))
        images_count_val = dataset_images_count('val')
        logging.debug('images_count_val {}'.format(images_count_val))

        model.fit_generator(
                    train_generator,
                    steps_per_epoch=images_count_train// batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=images_count_val// batch_size)



def model_evaluate(model, X_train, y_train, X_test, y_test):

        score = model.evaluate_generator(X_test, y_test, verbose=1)

        logging.debug('Test Loss: {}'.format(score[0]))
        logging.debug('Test Accuracy: {}'.format(score[1]))

        img_test = X_test[0:1]
        logging.debug('X_test.shape {}'.format(X_test.shape))                                           # (162, 128, 128, 1)
        logging.debug('img_test.shape {}'.format(img_test.shape))                                       # (1, 128, 128, 1)

        logging.debug('Prediction: {}'.format(model.predict(img_test)))
        logging.debug('Prediction: {}'.format(model.predict_classes(img_test)))
        logging.debug('Actual: {}'.format(y_test[0:1]))


def test_image():

        image_path_name = 'dataset/Humans/rider-8.jpg'

        # Testing a new image
        img_test = cv2.imread(image_path_name)
        img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
        img_test = cv2.resize(img_test, (img_height, img_width))
        img_test = np.array(img_test)
        img_test = img_test.astype('float32')
        # Solution: Must not use rescale=1./255 in ImageDataGenerator. VGG was trained on demeaned but not rescaled data.
        #img_test /= 255
        logging.debug('img_test.shape {}'.format(img_test.shape))                                       # (128, 128)

        if img_channels is 1:
                if K.image_dim_ordering() is 'th':
                        img_test = np.expand_dims(img_test, axis=0)
                        img_test = np.expand_dims(img_test, axis=0)
                        logging.debug('img_test.shape {}'.format(img_test.shape))
                else:
                        img_test = np.expand_dims(img_test, axis=3)
                        img_test = np.expand_dims(img_test, axis=0)
                        logging.debug('img_test.shape {}'.format(img_test.shape)
                                      )                               # (1, 128, 128, 1)

        else:
                if K.image_dim_ordering() == 'th':
                        img_test = np.rollaxis(img_test, 2, 0)
                        img_test = np.expand_dims(img_test, axis=0)
                        logging.debug('img_test.shape {}'.format(img_test.shape))
                else:
                        img_test = np.expand_dims(img_test, axis=0)
                        logging.debug('img_test.shape {}'.format(img_test.shape))

        # Predicting the test image
        logging.debug('Prediction: {}'.format(model.predict(img_test)))
        logging.debug('Predictied class: {}'.format(model.predict_classes(img_test)))


def model_save(model):

        model.save(model_saved_path)


def model_load(model_saved_path):

        model = load_model(model_saved_path)

        return model


def model_summary(model):
        model.summary()
        model.get_config()
        model.layers[0].get_config()
        model.layers[0].input_shape
        model.layers[0].output_shape
        model.layers[0].get_weights()
        model.layers[0].trainable

        # Each layer in Keras is a callable object
        logging.debug('Model Input Tensors: {}'.format(model.input))
        for layer in model.layers:
            logging.debug('layer.name {} layer.trainable {}'.format(layer.name, layer.trainable))
            logging.debug('layer config {}'.format(layer.get_config()))
        logging.debug('Model Output Tensors: {}'.format(model.output))


### MAIN ###

train_generator, validation_generator, input_shape = load_and_preprocess_data_generator()

logging.debug('1')
if os.path.exists(model_saved_path):
    logging.debug('Loading saved model...')
    model = model_load(model_saved_path)
else:
    logging.debug('Creating model...')
    # model = model_create(input_shape, num_classes)
    model = model_vgg_create(input_shape, num_classes)

logging.debug('2')
model_train(model, train_generator, validation_generator)

logging.debug('3')
model_save(model)

# model_evaluate(model, X_train, y_train, X_test, y_test)


# model_summary(model)
# test_image()






