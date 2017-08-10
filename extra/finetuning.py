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

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.optimizers import RMSprop, Adagrad
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping

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
top_model_weights_path = 'output/bottleneck_fc_model.h5'
top_model_weights_path = 'output/best-weights-015-0.5636-0.7923.hdf5'
finetune_model_weights_path = 'output/finetune_bottleneck_fc_model.h5'

#epochs = 50
epochs = 5
#batch_size = 16
batch_size = 1

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

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))
#top_model.add(Dense(len(class_names), activation='softmax'))                                        # Binary to Multi classification changes

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
# base_model.add(top_model)                                                                                # Not working; AttributeError: 'Model' object has no attribute 'add'
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
print(model.summary())

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:15]:                                                                     # Should be 15 instead of 25 I guess
    layer.trainable = False

# compile the model with a SGD/momentum optimizer and a very slow learning rate.
# rmsprop = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-8, decay=0.0)
# rmsprop = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-8, decay=0.0)
# adagrad = Adagrad(lr=1e-4, epsilon=1e-08, decay=0.0)
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])



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

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    dataset_train_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')
#    class_mode='sparse')                                                                       # Binary to Multi classification changes
#    class_mode='categorical')                                                                       # Binary to Multi classification changes

validation_generator = test_datagen.flow_from_directory(
    dataset_val_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')
#    class_mode='sparse')                                                                       # Binary to Multi classification changes
#    class_mode='categorical')                                                                       # Binary to Multi classification changes

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



# fine-tune the model
model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples,
    callbacks=callbacks_list)

# TODO: These are not the best weights
model.save_weights(finetune_model_weights_path)





