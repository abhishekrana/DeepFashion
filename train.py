'''
VGG16

    Conv2D              |
    Conv2D              |   ConvBlock1 (64 filters)
    MaxPooling2D        |

    Conv2D              |
    Conv2D              |   ConvBlock2 (128)
    MaxPooling2D        |

    Conv2D              |
    Conv2D              |   ConvBlock3 (256)
    Conv2D              |
    MaxPooling2D        |

    Conv2D              |
    Conv2D              |   ConvBlock4 (512)
    Conv2D              |
    MaxPooling2D        |

    Conv2D              |
    Conv2D              |   ConvBlock5 (512)
    Conv2D              |
    MaxPooling2D        |<- BottleneckFeatures


    Flatten             |
    Dense               |   FullyConnectedClassifier
    Dense               |
    Dense               |


1. Instantiate the convolutional part of the model, everything up to the fully-connected layers.
2. Run this model on our training and validation data once, recording the output (the "bottleneck features"
   from th VGG16 model: the last activation maps before the fully-connected layers) in two numpy arrays.
3. Train a small fully-connected model on top of the stored features.

Storing the features offline rather than adding our fully-connected model directly on top of a frozen
convolutional base and running the whole thing, is computational effiency.
Running VGG16 is expensive and we want to only do it once.

* Note that this prevents us from using data augmentation.

'''

### IMPORTS
from __future__ import print_function

import numpy as np
import fnmatch
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping
from keras import applications

import logging
FORMAT = "[%(lineno)4s : %(funcName)-30s ] %(message)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT)



### GLOBALS
# dimensions of our images.
img_width = 150
img_height = 150

top_model_weights_path = 'output/bottleneck_fc_model.h5'

# dataset_path = 'dataset_dogs_cats'
dataset_path = 'dataset'
dataset_train_path=os.path.join(dataset_path, 'train')
dataset_val_path=os.path.join(dataset_path, 'validation')
dataset_test_path=os.path.join(dataset_path, 'test')

# epochs = 50
epochs = 50
#batch_size = 16    # TODO: Issue at generator = datagen.flow_from_directory()
batch_size = 1

class_names=[]


# Count no. of images(.jpg) in a directory
def images_count(path):
    matches = []

    logging.debug('path {}'.format(path))
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*.jpg'):
            matches.append(os.path.join(root, filename))

    # logging.debug('matches {}'.format(matches))

    images_count = len(matches)
    logging.debug('images_count {}'.format(images_count))
    return images_count

# nb_train_samples = 2000
# nb_validation_samples = 800
nb_train_samples = images_count(dataset_train_path)
logging.debug('nb_train_samples {}'.format(nb_train_samples))
nb_validation_samples = images_count(dataset_val_path)
logging.debug('nb_validation_samples {}'.format(nb_validation_samples))


if not os.path.exists('output'):
    os.makedirs('output')

if not os.path.exists('logs'):
    os.makedirs('logs')

### FUNCTIONS
def dataset_class_images_count(dataset_class_path_name):
    matches = []
    for root, dirnames, filenames in os.walk(dataset_class_path_name):
        for filename in fnmatch.filter(filenames, '*.jpg'):
            matches.append(os.path.join(root, filename))
    images_count = len(matches)
    return images_count



def get_dataset_class_names(dataset_split_name):
    # Only matches .jpg files
    path = ''

    if dataset_split_name == 'train':
        path=os.path.join(dataset_path, 'train')
    elif dataset_split_name == 'val':
        path=os.path.join(dataset_path, 'validation')
    elif dataset_split_name == 'test':
        path=os.path.join(dataset_path, 'test')
    else:
        logging.error('Unknown dataset_split_name')
        exit(1)

    return [name for name in os.listdir(path)
                        if os.path.isdir(os.path.join(path, name))]



def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)                                                 # Epoch 50/50  2000/2000 [==============================] - 0s - loss: 0.0240 - acc: 0.9950 - val_loss: 1.0460 - val_acc: 0.8900

    # TODO
    # # Ideally we want use original VGG image prepossessing mean and no scale
    # datagen = ImageDataGenerator(rescale=1., featurewise_center=True) #(rescale=1./255)
    # datagen.mean=np.array([103.939, 116.779, 123.68],dtype=np.float32).reshape(3,1,1)


    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')                               # exclude 3 FC layers on top of network

    ## Train
    generator = datagen.flow_from_directory(
        dataset_train_path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,                                                                            # yield batches of data, no labels
        shuffle=False)                                                                              # first 1000 images will be cats, then 1000 dogs
    logging.debug('dataset_train_path {}'.format(dataset_train_path))

    # the predict_generator method returns the output of a model, given
    # a generator that yields batches of numpy data
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)

    # save the output as a Numpy array
    np.save(open('output/bottleneck_features_train.npy', 'w'),
            bottleneck_features_train)


    ## Validation
    generator = datagen.flow_from_directory(
        dataset_val_path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)

    np.save(open('output/bottleneck_features_validation.npy', 'w'),
            bottleneck_features_validation)


def train_top_model():

    train_data = np.load(open('output/bottleneck_features_train.npy'))
    validation_data = np.load(open('output/bottleneck_features_validation.npy'))



    # train_labels = np.array(
    #     [0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))                                # TODO  [0,0,0,1,1,1] if nb_train_samples=6; {{0 for dogs and 1 for cats}}
    # validation_labels = np.array(
    #     [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))                      # TODO: Change to Categorical for mutlilabel classification


    # TODO: Mapping may be wrong because datagen.flow_from_directory encodes class index alphabetically
    nb_train_samples = []
    for class_name in class_names:
        samples  = images_count(os.path.join(dataset_train_path, class_name))
        nb_train_samples.append(samples)
    train_labels = np.array(
        [0] * (nb_train_samples[0]) + [1] * (nb_train_samples[1]))                                # TODO  [0,0,0,1,1,1] if nb_train_samples=6; {{0 for dogs and 1 for cats}}

    logging.debug('train_labels {}'.format(train_labels))
    logging.debug('train_labels len {}'.format(len(train_labels)))
    logging.debug('train_labels type {}'.format(type(train_labels)))

    nb_validation_samples = []
    for class_name in class_names:
        samples  = images_count(os.path.join(dataset_val_path, class_name))
        nb_validation_samples.append(samples)
    validation_labels = np.array(
        [0] * (nb_validation_samples[0]) + [1] * (nb_validation_samples[1]))




    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    logging.debug('model summary {}'.format(model.summary()))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])                                 # Modify for multilabel classification

    logging.debug('train_labels len {}'.format(len(train_labels)))
    logging.debug('train_data len {}'.format(len(train_data)))
    logging.debug('validation_labels len {}'.format(len(validation_labels)))
    logging.debug('validation_data len {}'.format(len(validation_data)))


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



    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels),
              callbacks=callbacks_list)

    # TODO: These are not the best weights
    model.save_weights(top_model_weights_path)


### MAIN ###
class_names = get_dataset_class_names('train')
logging.debug('class_names {}'.format(class_names))

save_bottlebeck_features()
train_top_model()


