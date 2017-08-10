# Dataset:
# wget
# https://github.com/anujshah1003/own_data_cnn_implementation_keras/blob/master/data.zip?raw=true
# -O data.zip


# IMPORTS
from __future__ import print_function

import os
import cv2                  # load and preprocess dataset
import numpy as np              # reshape images, ...
import matplotlib.pyplot as plt

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
# 32 for cifar
img_height = 150         # rows
img_width = 150          # cols
# img_channels = 1
img_channels=3

# 408 samples from 4 classes [cat, dog, horse, human]
dataset_path = 'dataset'
model_saved_path = 'output/model.h5'

data_augmentation = True
num_classes = 2
#batch_size = 32
batch_size = 16
epochs = 2

# FUNCTIONS

if not os.path.exists('output'):
    os.makedirs('output')

if not os.path.exists('logs'):
    os.makedirs('logs')


def load_and_preprocess_data_1():

    img_dataset_list = []

    classes_list = os.listdir(dataset_path)
    logging.debug('classes_list {}'.format(classes_list))

    for class_name in classes_list:

        img_list = os.listdir(os.path.join(dataset_path, class_name))

        for img_name in img_list:
            img_path_name = os.path.join(dataset_path, class_name, img_name)
            # logging.debug('img_path {}'.format(img_path_name))
            # # (379, 499, 3

            img_input = cv2.imread(img_path_name)
            # logging.debug('img_input.shape {}'.format(img_input.shape))

            img_input_gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)

            # cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]]) -> dst
            # where fx is scale factor along the horizontal axis
            img_input_resize = cv2.resize(img_input_gray, (img_height, img_width))
            # logging.debug('img_input_resize.shape
            # {}'.format(img_input_resize.shape))             # (128, 128)

            # Add all images in the dataset to img_dataset_list
            # img_dataset_list is a list of 808 elements{images} each of which is a
            # 128x128{i.e 2D list}
            img_dataset_list.append(img_input_resize)

    img_dataset_arr = np.array(img_dataset_list)
    logging.debug('img_dataset_arr {}'.format(img_dataset_arr.shape)
                    )                               # (808, 128, 128)
    # (808, 128, 128, 3) for RGB

    img_dataset_arr = img_dataset_arr.astype('float32')

    # Method 1
    # Normalize (between 0 and 1)
    img_dataset_arr /= 255

    # Subtract mean

    # Method 2
    # 0 mean and unit variance
    img_dataset_arr_scaled = preprocessing.scale(img_dataset_arr)

    # Dimension ordering

    if img_channels is 1:
        if K.image_dim_ordering() is 'th':
            img_dataset_arr = np.expand_dims(img_dataset_arr, axis=1)
            logging.debug('img_dataset_arr.shape {}'.format(
                img_dataset_arr.shape))                 # (808, 1, 128, 128)
        else:
            img_dataset_arr = np.expand_dims(img_dataset_arr, axis=3)
            logging.debug('img_dataset_arr.shape {}'.format(
                img_dataset_arr.shape))  # (808, 128, 128, 1)

    else:
        # RGB
        if K.image_dim_ordering is 'th':
            img_dataset_arr = np.rollaxis(img_dataset_arr, 3, 1)
            logging.debug('img_dataset_arr.shape {}'.format(
                img_dataset_arr.shape))  # (808, 3, 128, 128)

    # TODO

    return img_dataset_arr


def load_and_preprocess_data_2():

    img_dataset_list = []

    classes_list = os.listdir(dataset_path)
    logging.debug('classes_list {}'.format(classes_list))

    for class_name in classes_list:

        img_list = os.listdir(os.path.join(dataset_path, class_name))

        for img_name in img_list:
            img_path_name = os.path.join(dataset_path, class_name, img_name)
            # logging.debug('img_path {}'.format(img_path_name))
            # # (379, 499, 3

            img_input = cv2.imread(img_path_name)
            # logging.debug('img_input.shape {}'.format(img_input.shape))

            img_input_gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
            img_input_gray = img_input

            # cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]]) -> dst
            # where fx is scale factor along the horizontal axis
            # img_input_resize = cv2.resize(img_input_gray, (img_height, img_width))
            # logging.debug('img_input_resize.shape
            # {}'.format(img_input_resize.shape))             # (128, 128)

            img_input_flatten = cv2.resize(
                img_input_gray, (img_height, img_width)).flatten()
            # logging.debug('img_input_flatten.shape
            # {}'.format(img_input_flatten.shape))           # (16384,)

            # Add all images in the dataset to img_dataset_list
            # img_dataset_list is a list of 808 elements{images} each of which is a
            # 16384
            img_dataset_list.append(img_input_flatten)

    img_dataset_arr = np.array(img_dataset_list)
    # (808, 128x128) i.e. (808, 16384)
    logging.debug('img_dataset_arr {}'.format(img_dataset_arr.shape))
    # (808, 128x128x3) for RGB

    img_dataset_arr_scaled = preprocessing.scale(img_dataset_arr)
    logging.debug('img_dataset_arr_scaled.shape {}'.format(
        img_dataset_arr_scaled.shape))           # (808, 16384)

    logging.debug('img_dataset_arr_scaled mean {}'.format(
        np.mean(img_dataset_arr_scaled)))         # -3.69616328743e-17
    logging.debug('img_dataset_arr_scaled std {}'.format(
        np.std(img_dataset_arr_scaled)))           # 1.0

    # [  1.50869416e-16   1.42487782e-16  -1.27785571e-16 ...,   1.42899993e-16 1.69006723e-16   5.66103819e-17]
    logging.debug('img_dataset_arr_scaled mean {}'.format(img_dataset_arr_scaled.mean(axis=0)))
    logging.debug(
        'img_dataset_arr_scaled std {}'.format(
            img_dataset_arr_scaled.std(axis=0)))       # [ 1.  1.  1. ...,  1.  1.  1.]

    # Dimension ordering

    if K.image_dim_ordering() is 'th':
        img_data = img_dataset_arr_scaled.reshape(img_dataset_arr.shape[0], img_channels, img_height, img_width)
        logging.debug('img_data.shape {}'.format(img_data.shape))
    else:
        img_data = img_dataset_arr_scaled.reshape(img_dataset_arr.shape[0], img_height, img_width, img_channels)
        logging.debug('img_data.shape {}'.format(img_data.shape)
                            )                                   # (808, 128, 128, c)

    input_shape = img_data[0].shape
    logging.debug('input_shape {}'.format(input_shape))

    # Labels to dataset
    num_of_samples = img_data.shape[0]
    logging.debug('num_of_samples {}'.format(num_of_samples))

    labels = np.ones((num_of_samples), dtype='int64')
    labels[0:102] = 0
    labels[102:204] = 1
    labels[204:606] = 2
    labels[606:] = 3

    class_names = ['cats', 'dogs', 'horses', 'humans']

    # One hot encoding
    Y = keras.utils.to_categorical(labels, num_classes)
    #Y = np_utils.to_categorical(labels, num_classes)

    # Shuffle and split dataset
    x, y = shuffle(img_data, Y, random_state=2)
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=4)  # Same split every time as random state is set

    return X_train, X_test, y_train, y_test, input_shape

def load_and_preprocess_data_3():
    # The data, shuffled and split between train and test sets:
    (X_train, y_train), (x_test, y_test) = cifar10.load_data()
    logging.debug('X_train shape: {}'.format(X_train.shape))
    logging.debug('train samples: {}'.format(X_train.shape[0]))
    logging.debug('test samples: {}'.format(x_test.shape[0]))

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    X_train = X_train.astype('float32')
    x_test = x_test.astype('float32')
    X_train /= 255
    x_test /= 255

    input_shape = X_train[0].shape
    logging.debug('input_shape {}'.format(input_shape))
    input_shape = X_train.shape[1:]
    logging.debug('input_shape {}'.format(input_shape))

    return X_train, x_test, y_train, y_test, input_shape

def load_and_preprocess_data_generator():

        # TBD
        train_data_dir = "dataset2/train"
        validation_data_dir = "dataset2/validation"

        # Initiate the train and test generators with data Augumentation
        train_datagen = ImageDataGenerator(rescale = 1./255, shear_range=0.2, horizontal_flip = True, fill_mode = "nearest",
                                           zoom_range = 0.3, width_shift_range = 0.3, height_shift_range=0.3,
                                           rotation_range=30)

        test_datagen = ImageDataGenerator(rescale = 1./255, shear_range=0.2, horizontal_flip = True, fill_mode = "nearest",
                                          zoom_range = 0.3, width_shift_range = 0.3, height_shift_range=0.3,
                                          rotation_range=30)

        train_generator = train_datagen.flow_from_directory(train_data_dir, target_size = (img_height, img_width),
                                                            batch_size = batch_size, class_mode = "categorical")

        validation_generator = test_datagen.flow_from_directory(validation_data_dir, target_size = (img_height, img_width),
                                                                class_mode = "categorical")

        # HARDCODING
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


def model_train(model, X_train, y_train, X_test, y_test, train_generator=None, validation_generator=None):

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

        if train_generator is None:

            # Train
            if not data_augmentation:
                logging.debug('Not using data augmentation.')
                model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        verbose=1,
                        shuffle=True,
                        callbacks=callbacks_list)
                #hist = model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=1, validation_split=0.2, callbacks=callbacks_list)
            else:
                logging.debug('Using real-time data augmentation.')
                # This will do preprocessing and realtime data augmentation:
                datagen = ImageDataGenerator(
                    featurewise_center=False,  # set input mean to 0 over the dataset
                    samplewise_center=False,  # set each sample mean to 0
                    featurewise_std_normalization=False,  # divide inputs by std of the dataset
                    samplewise_std_normalization=False,  # divide each input by its std
                    zca_whitening=False,  # apply ZCA whitening
                    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                    horizontal_flip=True,  # randomly flip images
                    vertical_flip=False,  # randomly flip images
                    rescale=0)

                logging.debug('X_train.shape {}'.format(X_train.shape))

                # Compute quantities required for feature-wise normalization
                # (std, mean, and principal components if ZCA whitening is applied).
                datagen.fit(X_train)

                # Fit the model on the batches generated by datagen.flow().
                model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                                    steps_per_epoch=X_train.shape[0] // batch_size,
                                    epochs=epochs,
                                    validation_data=(X_test, y_test),
                                    callbacks = callbacks_list)


        else:

            model.fit_generator(
                        train_generator,
                        steps_per_epoch=2000// batch_size,
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=800// batch_size)



def model_evaluate(model, X_train, y_train, X_test, y_test):

        score = model.evaluate(X_test, y_test, verbose=1)

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
        img_test /= 255
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

        # Save Model and Weights
        # # Serialize model to JSON
        # model_json = model.to_json()

        # if not os.path.exists('output'):
        #     os.makedirs('output')

        # with open('output/model.json', 'w') as json_file:
        #     json_file.write(model_json)

        # model.save_weights('output/model.h5')
        # logging.debug('Model saved to disk')


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

# X_train, X_test, y_train, y_test, input_shape = load_and_preprocess_data_3()

train_generator, validation_generator, input_shape = load_and_preprocess_data_generator()

if os.path.exists(model_saved_path):
    logging.debug('Loading saved model...')
    model = model_load(model_saved_path)
else:
    logging.debug('Creating model...')
    # model = model_create(input_shape, num_classes)
    model = model_vgg_create(input_shape, num_classes)

# model_train(model, X_train, y_train, X_test, y_test, None, None)
model_train(model, None, None, None, None, train_generator, validation_generator)

model_save(model)

# model_evaluate(model, X_train, y_train, X_test, y_test)


# model_summary(model)
# test_image()













# intermediate_layer_summary()
# plot_graphs()
# visualize_intermediate_cnn_layers()
# plot_confusion_matrix()








