'''

wget https://doc-0g-4s-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/327iep5nmg492g7968am9g08prba2usg/1500897600000/13951467387256278872/*/0Bz7KyqmuGsilT0J5dmRCM0ROVHc?e=download -O vgg16_weights.h5


This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:

data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
'''

### IMPORTS
from __future__ import print_function

import os
import fnmatch
import numpy as np

from sklearn.utils import class_weight

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.optimizers import RMSprop, Adagrad
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping

import os
import fnmatch
import numpy as np
import skimage.data
import cv2
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from selective_search import selective_search_bbox

import logging
FORMAT = "[%(lineno)4s : %(funcName)-30s ] %(message)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT)


### GLOBALS
# dimensions of our images.
# img_width = 150
# img_height = 150
img_width = 224
img_height = 224

# dataset_path = 'dataset_dogs_cats'
dataset_path = 'dataset'
dataset_train_path=os.path.join(dataset_path, 'train')
dataset_val_path=os.path.join(dataset_path, 'validation')
dataset_test_path=os.path.join(dataset_path, 'test')


# path to the model weights files.
weights_path = 'weights/vgg16_weights.h5'

#top_model_weights_path = 'output/bottleneck_fc_model.h5'
#top_model_weights_path = 'output_6_categ/best-weights-015-0.5636-0.7923.hdf5'

# Cropped 6 categ
#top_model_weights_path = 'output/best-weights-015-0.3270-0.8752.hdf5'
#top_model_weights_path = 'output/best-weights-033-1.0725-0.6497.hdf5'
top_model_weights_path = 'output/best-weights-008-0.1017.hdf5'

finetune_model_weights_path = 'output/finetune_bottleneck_fc_model.h5'


#epochs = 50
epochs = 200
#batch_size = 16
batch_size = 1
#batch_size = 64
#batch_size = 32

class_names=[]


# Sorted subdirectories list
def get_subdir_list(path):
    names=[]
    for name in sorted(os.listdir(path)):
        if os.path.isdir(os.path.join(path, name)):
            names.append(name)
    logging.info('names {}'.format(names))
    return names


# Count no. of images(.jpg) in a directory
def get_images_count_recursive(path):
    matches = []
    logging.debug('path {}'.format(path))
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*.jpg'):
            matches.append(os.path.join(root, filename))

    # logging.debug('matches {}'.format(matches))

    images_count = len(matches)
    return images_count



def predict_image_name(model, image_path_name):

    logging.debug('image_path_name {}'.format(image_path_name))

    candidates = selective_search_bbox(image_path_name)
    logging.debug('candidates {}'.format(candidates))

    image_name = image_path_name.split('/')[-1].split('.')[0]
    logging.debug('image_name {}'.format(image_name))

    img_read = Image.open(image_path_name)
    logging.debug('{} {} {}'.format(img_read.format, img_read.size, img_read.mode))
    # img_read.show()
    i=0
    for x, y, w, h in (candidates):
        #  left, upper, right, and lower pixel; The cropped section includes the left column and
        #  the upper row of pixels and goes up to (but doesn't include) the right column and bottom row of pixels

        img_crop = img_read.crop((y, x, y+w, x+h))
        img_crop.save('temp/test/'+ image_name + '_' + str(i) + '_cropped_' + '.jpg')
        logging.debug('img_crop {} {} {}'.format(img_crop.format, img_crop.size, img_crop.mode))

        img_crop_resize = img_crop.resize((img_width, img_height))
        img_crop_resize.save('temp/test/'+ image_name + '_' + str(i) + '_cropped_resize' + '.jpg')
        logging.debug('img_crop_resize {} {} {}'.format(img_crop_resize.format, img_crop_resize.size, img_crop_resize.mode))

        i=i+1

        img=np.array(img_crop_resize).astype(np.float32)
        img[:,:,0] -= 103.939
        img[:,:,1] -= 116.779
        img[:,:,2] -= 123.68
        #img = img.transpose((2,0,1))
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img, batch_size, verbose=1)
        logging.debug('\n\nprediction \n{}'.format(prediction))

        for index,preds in enumerate(prediction):
            for pred in preds:
                logging.debug('pred {0:6f}'.format(float(pred)))







# nb_train_samples = 2000
# nb_validation_samples = 800
nb_train_samples = get_images_count_recursive(dataset_train_path)
logging.debug('nb_train_samples {}'.format(nb_train_samples))
nb_validation_samples = get_images_count_recursive(dataset_val_path)
logging.debug('nb_validation_samples {}'.format(nb_validation_samples))

if not os.path.exists('output'):
    os.makedirs('output')

if not os.path.exists('logs'):
    os.makedirs('logs')


# TODO: HARDCODING - Should be same as used during training VGG; Else error (None, None, 512)
input_shape = (img_width, img_height, 3)

# Sorted subdirectories list
def get_subdir_list(path):
    names=[]
    for name in sorted(os.listdir(path)):
        if os.path.isdir(os.path.join(path, name)):
            names.append(name)
    logging.debug('names {}'.format(names))
    return names

class_names = get_subdir_list(dataset_train_path)
logging.debug('class_names {}'.format(class_names))

# build the VGG16 network
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
print('Model loaded.')
print(base_model.output_shape)                                                                           # (None, None, None, 512) if input_shape not given in applications.VGG16
print(base_model.output_shape[1:])                                                                       # (None, None, 512)


### MODEL1
# # build a classifier model to put on top of the convolutional model
# top_model = Sequential()
# top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
# top_model.add(Dense(256, activation='relu'))
# top_model.add(Dropout(0.5))
# top_model.add(Dense(len(class_names), activation='softmax'))                                        # Binary to Multi classification changes
# #top_model.add(Dense(1, activation='sigmoid'))
# logging.debug('class_names len {}'.format(len(class_names)))

# # note that it is necessary to start with a fully-trained
# # classifier, including the top classifier,
# # in order to successfully do fine-tuning
# top_model.load_weights(top_model_weights_path)

# # add the model on top of the convolutional base
# # base_model.add(top_model)                                                                                # Not working; AttributeError: 'Model' object has no attribute 'add'
# model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
# print(model.summary())







### MODEL2
inputs = Input(shape=(base_model.output_shape[1:]))
x_common = Dense(256, activation='relu')(inputs)

## Model Classification
x = Flatten()(x_common)
#x = Dropout(dropout_rate)(x)
predictions_class = Dense(len(class_names), activation='softmax', name='predictions_class')(x)

## Model (Regression) IOU score
x = Flatten()(x_common)
# x = Dense(256, activation='relu')(x)
# x = Dropout(dropout_rate)(x)
predictions_iou = Dense(1, activation='sigmoid', name='predictions_iou')(x)


predictions_class.load_weights(top_model_weights_path)
predictions_iou.load_weights(top_model_weights_path)

logging.debug('final_model ')

# This creates a model that includes the Input layer and three Dense layers
top_model = Model(inputs=inputs, outputs=[predictions_class(base_model.output), predictions_iou(base_model.output)])


logging.debug('model summary {}'.format(top_model.summary()))


# def predict_image_name(model, image_path_name):



# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:15]:                                                                     # Should be 15 instead of 25 I guess
    layer.trainable = False

# compile the model with a SGD/momentum optimizer and a very slow learning rate.
# rmsprop = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-8, decay=0.0)
# rmsprop = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-8, decay=0.0)
# adagrad = Adagrad(lr=1e-4, epsilon=1e-08, decay=0.0)

### MODEL 1
# model.compile(loss='sparse_categorical_crossentropy',
#               optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
#               metrics=['accuracy'])

### MODEL 2
#model.compile(loss={'predictions_class': 'sparse_categorical_crossentropy', 'predictions_iou': 'mean_squared_error'},
model.compile(loss={'model_1': 'sparse_categorical_crossentropy', 'model_1': 'mean_squared_error'},
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])


### TODO
model.save_weights(finetune_model_weights_path)

# # with pre-precessing version
# # Use keras 1.2.2 for preprocessing_function
# # x = 3D tensor version
# def preprocess_input(x):
#     # 'RGB'->'BGR'
#     x = x[:, :, ::-1]
#     # Zero-center by mean pixel
#     x[:, :, 0] -= 103.939
#     x[:, :, 1] -= 116.779
#     x[:, :, 2] -= 123.68
#     return x
# train_datagen = ImageDataGenerator(
#     preprocessing_function=preprocess_input,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True)

# test_datagen = ImageDataGenerator(
#     preprocessing_function=preprocess_input)



# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    dataset_train_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse')                                                                       # Binary to Multi classification changes
#    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    dataset_val_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse')                                                                       # Binary to Multi classification changes
#    class_mode='binary')

# Callbacks
filename = 'output/model_train_finetune.csv'
csv_log = CSVLogger(filename, separator=' ', append=False)

early_stopping = EarlyStopping(
    monitor='val_loss', patience=50, verbose=1, mode='min')

filepath = "output/best-weights-finetune-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
    save_best_only=True, save_weights_only=False, mode='min', period=1)                     # min because we are monitoring val_loss that should decrease

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

callbacks_list = [csv_log, early_stopping, checkpoint, tensorboard]
logging.debug('callbacks_list {}'.format(callbacks_list))



# train labels should be calculated using train_generator.class_indices but presently using
# recursive folder iteration method as generator reads folder i.e. class labels alphabetically
class_names = get_subdir_list(dataset_train_path)
logging.debug('class_names {}'.format(class_names))

train_labels = []
for index, class_name in enumerate(class_names):
    images_count  = get_images_count_recursive(os.path.join(dataset_train_path, class_name))
    logging.debug('images_count {}'.format(images_count))
    # class_weight_dict[class_name] = images_count
    for _ in range(images_count):
        train_labels.append(index)
train_labels = np.array(train_labels)
logging.debug('train_labels {}'.format(train_labels))
logging.debug('train_labels len {}'.format(len(train_labels)))
logging.debug('train_labels type {}'.format(type(train_labels)))

# Generate weights based on images count for each class
class_weight_val = class_weight.compute_class_weight('balanced', np.unique(train_labels), train_labels)
logging.info('class_weight_val {}'.format(class_weight_val))
logging.info('train_labels {}'.format(train_labels))
logging.info('train_labels len {}'.format(len(train_labels)))



# MODEL 1
# # fine-tune the model
# model.fit_generator(
#     train_generator,
#     class_weight=class_weight_val,                                                            # dictionary mapping classes to a weight value, used for scaling the loss function (during training only).
#     samples_per_epoch=nb_train_samples // batch_size,
#     epochs=epochs,
#     validation_data=validation_generator,
#     nb_val_samples=nb_validation_samples,
#     callbacks=callbacks_list)

# MODEL 2
# fine-tune the model
model.fit_generator(
    train_generator,
    class_weight=[class_weight_val, class_weight_val],                                              # dictionary mapping classes to a weight value, used for scaling the loss function (during training only).
    samples_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=(validation_generator, [validation_labels, validation_labels_iou]),
    nb_val_samples=nb_validation_samples,
    callbacks=callbacks_list)





# TODO: These are not the best weights
model.save_weights(finetune_model_weights_path)





