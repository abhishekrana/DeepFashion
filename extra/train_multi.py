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



VGG16 for input image of size (150, 150, 3)

Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 150, 150, 3)       0


block1_conv1 (Conv2D)        (None, 150, 150, 64)      1792
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 150, 150, 64)      36928
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 75, 75, 64)        0
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 75, 75, 128)       73856
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 75, 75, 128)       147584
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 37, 37, 128)       0
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 37, 37, 256)       295168
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 37, 37, 256)       590080
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 37, 37, 256)       590080
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 18, 18, 256)       0
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 18, 18, 512)       1180160
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 18, 18, 512)       2359808
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 18, 18, 512)       2359808
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 9, 9, 512)         0
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 9, 9, 512)         2359808
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 9, 9, 512)         2359808
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 9, 9, 512)         2359808
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 4, 4, 512)         0            <--- bottleneck_features_train
_________________________________________________________________
sequential_1 (Sequential)    (None, 1)                 2097665
=================================================================



Bottleneck features are saved alphabetically(class names). So we are generating train_labels using this assumption
and hence cannot shuffle data

[ 180 : get_subdir_list                ] names ['Blazer', 'Jeans', 'Joggers', 'Jumpsuit', 'Leggings', 'Romper']
[ 206 : save_bottlebeck_features       ] generator.class_indices {'Blazer': 0, 'Jeans': 1, 'Leggings': 4, 'Romper': 5, 'Joggers': 2, 'Jumpsuit': 3}

Blazer      5408
Jeans       5126
Joggers     3260
Jumpsuit    4464
Leggings    3571
Romper      5425

[ 352 : train_top_model                ] class_weight_val [ 0.8399285   0.88613604  1.39335378  1.01754779  1.27200597  0.83729647]
[ 353 : train_top_model                ] train_labels [0 0 0 ..., 5 5 5]
[ 354 : train_top_model                ] train_labels len 27254



[ 370 : train_top_model                ] class_weight_val [ 30.72420307   0.68743132   0.20942027  16.2341859   15.29888301
0.38701109   9.94018335   2.41561311  10.35551134   3.15856293
16.59655612   7.13556348   1.27753559   0.49253161   0.72524943
8.39193808   6.96185126   1.14037686   0.8328021   37.93498542
2.27100096   1.04106093   7.57154495   6.42077473  34.74419226
0.68527716   0.26189705   0.34441621   0.3906303    1.67159558
4.7600878    0.33181262   0.51136569  12.95340966  37.55180375]

Classes         Images      Weights(less no. of images in dataset are weighted more to compensate for their less count)
Anorak          121         30.72420307
Blazer          5408        0.68743132
Blouse          17752       0.20942027
Bomber          229         16.2341859
Button-Down     243         15.29888301
Cardigan        9606        0.38701109
Chinos          374         9.94018335
Coat            1539        2.41561311
Culottes        359         10.35551134
Cutoffs         1177        3.15856293
Flannel         224         16.59655612
Henley          521         7.13556348
Hoodie          2910        1.27753559
Jacket          7548        0.49253161
Jeans           5126        0.72524943
Jeggings        443         8.39193808
Jersey          534         6.96185126
Joggers         3260        1.14037686
Jumpsuit        4464        0.8328021
Kaftan          98          37.93498542
Kimono          1637        2.27100096
Leggings        3571        1.04106093
Parka           491         7.57154495
Poncho          579         6.42077473
Robe            107         34.74419226
Romper          5425        0.68527716
Shorts          14195       0.26189705
Skirt           10794       0.34441621
Sweater         9517        0.3906303
Sweatpants      2224        1.67159558
Sweatshorts     781         4.7600878
Tank            11204       0.33181262
Top             7270        0.51136569
Trunks          287         12.95340966
Turtleneck      99          37.55180375

train_labels [ 0  0  0 ..., 34 34 34]


'''

### IMPORTS
from __future__ import print_function

import numpy as np
import fnmatch
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.layers import Dropout, Flatten, Dense, Input
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping
from keras import applications
from keras.optimizers import RMSprop, Adagrad
import keras.optimizers

from sklearn.utils import class_weight
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

# from utils import get_dataset_class_image_count_sorted

import logging
FORMAT = "[%(lineno)4s : %(funcName)-30s ] %(message)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT)



### GLOBALS
# dimensions of our images.
# TODO: Test with 224 for VGG16
# img_width = 150
# img_height = 150
img_width = 224
img_height = 224

top_model_weights_path = 'output/bottleneck_fc_model.h5'

# dataset_path = 'dataset_dogs_cats'
dataset_path = 'dataset'
dataset_train_path=os.path.join(dataset_path, 'train')
dataset_val_path=os.path.join(dataset_path, 'validation')
dataset_test_path=os.path.join(dataset_path, 'test')

#epochs = 2000
epochs = 50
#epochs = 1000
#batch_size = 16    # TODO: Issue at generator = datagen.flow_from_directory()

batch_size = 32
#batch_size = 1
# batch_size = 512
# batch_size = 1024
# batch_size = 2048
# batch_size = 4096

class_names=[]
# class_weight_dict={}


# Count no. of images(.jpg) in a directory (sorted)
def get_images_count_recursive(path):
    matches = []
    score_iou = []
    # logging.debug('path {}'.format(path))
    for root, dirnames, filenames in sorted(os.walk(path)):
        for filename in sorted(fnmatch.filter(filenames, '*.jpg')):
            # logging.debug('filename {}'.format(filename))
            matches.append(os.path.join(root, filename))
            score_iou.append(filename.split('_')[-1].replace('.jpg',''))

    # logging.debug('matches {}'.format(matches))

    images_count = len(matches)
    return score_iou, images_count

# nb_train_samples = 2000
# nb_validation_samples = 800
score_iou_train_g, nb_train_samples = get_images_count_recursive(dataset_train_path)
logging.debug('nb_train_samples {}'.format(nb_train_samples))
score_iou_validation_g, nb_validation_samples = get_images_count_recursive(dataset_val_path)
logging.debug('nb_validation_samples {}'.format(nb_validation_samples))


if not os.path.exists('output'):
    os.makedirs('output')

if not os.path.exists('logs'):
    os.makedirs('logs')

### FUNCTIONS

# Sorted subdirectories list
def get_subdir_list(path):
    names=[]
    for name in sorted(os.listdir(path)):
        if os.path.isdir(os.path.join(path, name)):
            names.append(name)
    logging.info('names {}'.format(names))
    return names


def save_bottlebeck_features_btl():

    dataset_btl_path = 'dataset_btl/train'
    batch_size = 1

    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')                               # exclude 3 FC layers on top of network

    score_iou_btl_g, nb_btl_samples = get_images_count_recursive(dataset_btl_path)
    logging.debug('score_iou_btl_g {}'.format(score_iou_btl_g))
    logging.debug('nb_btl_samples {}'.format(nb_btl_samples))


    ## Train
    generator = datagen.flow_from_directory(
        dataset_btl_path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        classes=None,                                                                               #  the order of the classes, which will map to the label indices, will be alphanumeric
        class_mode=None,                                                                            # "categorical": 2D one-hot encoded labels; "None": yield batches of data, no labels; "sparse" will be 1D integer labels.
        save_to_dir='temp',
        shuffle=False)                                                                              # Don't shuffle else [class index = alphabetical folder order] logic used below might become wrong; first 1000 images will be cats, then 1000 dogs
    logging.info('generator.class_indices {}'.format(generator.class_indices))
                                                                                                    # classes: If not given, the order of the classes, which will map to the label indices, will be alphanumeric
    bottleneck_features_btl = model.predict_generator(
        generator, nb_btl_samples // batch_size)
    logging.debug('bottleneck_features_btl {}'.format(bottleneck_features_btl.shape))           # bottleneck_features_train (10534, 4, 4, 512) where train images i.e Blazer+Jeans=5408+5126=10532 images;

    # save the output as a Numpy array
    logging.debug('Saving bottleneck_features_btl...')
    np.save(open('output/bottleneck_features_btl.npy', 'w'),
            bottleneck_features_btl)






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
        classes=None,                                                                               #  the order of the classes, which will map to the label indices, will be alphanumeric
        class_mode=None,                                                                            # "categorical": 2D one-hot encoded labels; "None": yield batches of data, no labels; "sparse" will be 1D integer labels.
        save_to_dir=None,
        shuffle=False)                                                                              # Don't shuffle else [class index = alphabetical folder order] logic used below might become wrong; first 1000 images will be cats, then 1000 dogs
    logging.debug('dataset_train_path {}'.format(dataset_train_path))
    logging.info('generator.class_indices {}'.format(generator.class_indices))
                                                                                                    # classes: If not given, the order of the classes, which will map to the label indices, will be alphanumeric
    #logging.debug('generator {}'.format(generator))
    # the predict_generator method returns the output of a model, given
    # a generator that yields batches of numpy data
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    # logging.debug('bottleneck_features_train {}'.format(len(bottleneck_features_train)))          # bottleneck_features_train is 'block5_pool (MaxPooling2D)' in VGG16
    logging.debug('bottleneck_features_train {}'.format(bottleneck_features_train.shape))           # bottleneck_features_train (10534, 4, 4, 512) where train images i.e Blazer+Jeans=5408+5126=10532 images;

    # save the output as a Numpy array
    logging.debug('Saving bottleneck_features_train...')
    np.save(open('output/bottleneck_features_train.npy', 'w'),
            bottleneck_features_train)


    ## Validation
    generator = datagen.flow_from_directory(
        dataset_val_path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        classes=None,
        class_mode=None,
        save_to_dir=None,
        shuffle=False)

    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)

    logging.debug('Saving bottleneck_features_validation...')
    np.save(open('output/bottleneck_features_validation.npy', 'w'),
            bottleneck_features_validation)




# def create_model(optimizer='rmsprop', init='glorot_uniform')
# param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init)

# def create_model(init_mode='uniform'):
#  ((Dense(1, kernel_initializer=init_mode, activation='sigmoid'))build_fn=create_model, epochs=100, batch_size=10, verbose=0)

# def create_model(dropout_rate=0.0, weight_constraint=0):j
#     model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='linear', kernel_constraint=maxnorm(weight_constraint)))
#         model.add(Dropout(dropout_rate))

# def create_model(neurons=1):
#     model.add(Dense(neurons, input_dim=8, kernel_initializer='uniform', activation='linear', kernel_constraint=maxnorm(4)))


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


def create_model(input_shape, optimizer='Adagrad', learn_rate=None, decay=0.0, momentum=0.0, activation='relu', dropout_rate=0.5):
    logging.debug('input_shape {}'.format(input_shape))
    logging.debug('input_shape {}'.format(type(input_shape)))

    # input_shape = (7, 7, 512)

    # Optimizer
    optimizer, learn_rate = get_optimizer(optimizer, learn_rate, decay, momentum)


    # Model
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))

    model.add(Dense(256, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(len(class_names), activation='softmax'))                                        # Binary to Multi classification changes
    # model.add(Dense(1, activation='sigmoid'))

    logging.debug('model summary {}'.format(model.summary()))


    # Compile
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])                     # Binary to Multi classification changes

    logging.info('optimizer:{}  learn_rate:{}  decay:{}  momentum:{}  activation:{}  dropout_rate:{}'.format(
        optimizer, learn_rate, decay, momentum, activation, dropout_rate))

    return model


def create_model_func(input_shape, optimizer='Adagrad', learn_rate=None, decay=0.0, momentum=0.0, activation='relu', dropout_rate=0.5):
    logging.debug('input_shape {}'.format(input_shape))
    logging.debug('input_shape {}'.format(type(input_shape)))

    # Optimizer
    optimizer, learn_rate = get_optimizer(optimizer, learn_rate, decay, momentum)

    # input_shape = (7, 7, 512)                                                                     # VGG bottleneck layer - block5_pool (MaxPooling2D)

    inputs = Input(shape=(input_shape))                                                             # This returns a tensor

    ## Model Classification
    # a layer instance is callable on a tensor, and returns a tensor
    x = Flatten()(inputs)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    predictions_class = Dense(len(class_names), activation='softmax', name='predictions_class')(x)


    ## Model (Regression) IOU score
    # a layer instance is callable on a tensor, and returns a tensor
    x = Flatten()(inputs)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    predictions_iou = Dense(1, activation='sigmoid', name='predictions_iou')(x)
    #predictions_iou = Dense(1, activation='linear', name='predictions_iou')(x)


    # This creates a model that includes the Input layer and three Dense layers
    model = Model(inputs=inputs, outputs=[predictions_class, predictions_iou])


    logging.debug('model summary {}'.format(model.summary()))


    # Compile
    model.compile(optimizer=optimizer,
                  loss={'predictions_class': 'sparse_categorical_crossentropy', 'predictions_iou': 'mean_squared_error'}, metrics=['accuracy'])
                  #loss_weights={'main_output': 1., 'aux_output': 0.2})
                  #loss='sparse_categorical_crossentropy', metrics=['accuracy'])                     # Binary to Multi classification changes

    logging.info('optimizer:{}  learn_rate:{}  decay:{}  momentum:{}  activation:{}  dropout_rate:{}'.format(
        optimizer, learn_rate, decay, momentum, activation, dropout_rate))

    return model

def create_model_func2(input_shape, optimizer='Adagrad', learn_rate=None, decay=0.0, momentum=0.0, activation='relu', dropout_rate=0.5):
    logging.debug('input_shape {}'.format(input_shape))
    logging.debug('input_shape {}'.format(type(input_shape)))

    # Optimizer
    optimizer, learn_rate = get_optimizer(optimizer, learn_rate, decay, momentum)

    # input_shape = (7, 7, 512)                                                                     # VGG bottleneck layer - block5_pool (MaxPooling2D)

    inputs = Input(shape=(input_shape))
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


    # This creates a model that includes the Input layer and three Dense layers
    model = Model(inputs=inputs, outputs=[predictions_class, predictions_iou])


    logging.debug('model summary {}'.format(model.summary()))


    # Compile
    model.compile(optimizer=optimizer,
                  loss={'predictions_class': 'sparse_categorical_crossentropy', 'predictions_iou': 'mean_squared_error'}, metrics=['accuracy'],
                  loss_weights={'predictions_class': 0.5, 'predictions_iou': 0.5})
                  #loss={'predictions_class': 'sparse_categorical_crossentropy', 'predictions_iou': 'logcosh'}, metrics=['accuracy'],
                  #loss={'predictions_class': 'sparse_categorical_crossentropy', 'predictions_iou': 'mean_squared_error'}, metrics=['accuracy'])

    logging.info('optimizer:{}  learn_rate:{}  decay:{}  momentum:{}  activation:{}  dropout_rate:{}'.format(
        optimizer, learn_rate, decay, momentum, activation, dropout_rate))

    return model




def train_top_model(is_grid_search):

    # train data is basically saved weights of each training image of VGG16 Layer 'block5_pool (MaxPooling2D)'   (None, 4, 4, 512)  <--- bottleneck_features_train
    logging.debug('Loading bottleneck_features_train...')
    train_data = np.load(open('output/bottleneck_features_train.npy'))
    input_shape = train_data.shape[1:]
    logging.debug('input_shape {}'.format(input_shape))

    logging.debug('Loading bottleneck_features_validation...')
    validation_data = np.load(open('output/bottleneck_features_validation.npy'))

    # TODO:
    # ISSUE: Class names are read from dataset/train dir and train_labels are created in that order. So if data is
    # shuffled, there will be mismatch in image index and true label

    train_labels = []
    train_labels_iou = []
    for index, class_name in enumerate(class_names):
        score_iou_train, images_count  = get_images_count_recursive(os.path.join(dataset_train_path, class_name))
        # train_labels_iou.append(score_iou_train)
        #train_labels_iou = train_labels_iou + score_iou_train
        logging.debug('train_labels_iou {}'.format(type(train_labels_iou)))
        train_labels_iou = train_labels_iou + score_iou_train
        # logging.debug('images_count {}'.format(images_count))
        for _ in range(images_count):
            train_labels.append(index)
    train_labels = np.array(train_labels)
    train_labels_iou_arr = np.array(train_labels_iou)
    train_labels_iou = train_labels_iou_arr.astype(np.float)
    train_labels_iou = np.round(train_labels_iou, 2)
    logging.debug('train_labels_iou len {}'.format(len(train_labels_iou)))
    logging.debug('train_labels_iou {}'.format(train_labels_iou))
    logging.debug('train_labels {}'.format(train_labels))
    logging.debug('train_labels len {}'.format(len(train_labels)))
    logging.debug('train_labels type {}'.format(type(train_labels)))
    logging.debug('train_labels_iou {}'.format(train_labels_iou))
    logging.debug('train_labels_iou len {}'.format(len(train_labels_iou)))
    logging.debug('train_labels_iou type {}'.format(type(train_labels_iou)))
    logging.debug('train_labels_iou type {}'.format(type(train_labels_iou[0])))

    # for key,value in class_weight_dict.iteritems():
    #     logging.debug('class_weight_dict {} {}'.format(key, value))

    validation_labels = []
    validation_labels_iou = []
    for index, class_name in enumerate(class_names):
        score_iou_validation, images_count  = get_images_count_recursive(os.path.join(dataset_val_path, class_name))
        # logging.debug('images_count {}'.format(images_count))
        # validation_labels_iou.append(score_iou_validation)
        validation_labels_iou = validation_labels_iou + score_iou_validation
        for _ in range(images_count):
            validation_labels.append(index)
    validation_labels = np.array(validation_labels)
    validation_labels_iou_arr = np.array(validation_labels_iou)
    validation_labels_iou = validation_labels_iou_arr.astype(np.float)
    validation_labels_iou = np.round(validation_labels_iou, 2)
    logging.debug('validation_labels {}'.format(validation_labels))
    logging.debug('validation_labels len {}'.format(len(validation_labels)))
    logging.debug('validation_labels type {}'.format(type(validation_labels)))
    logging.debug('validation_labels_iou {}'.format(validation_labels_iou))
    logging.debug('validation_labels_iou len {}'.format(len(validation_labels_iou)))
    logging.debug('validation_labels_iou type {}'.format(type(validation_labels_iou)))

    # train_labels = np.array(
    #     [0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))                                # TODO  [0,0,0,1,1,1] if nb_train_samples=6; {{0 for dogs and 1 for cats}}
    # validation_labels = np.array(
    #     [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))                      # TODO: Change to Categorical for mutlilabel classification


    # nb_train_samples = []
    # for class_name in class_names:
    #     samples  = get_images_count_recursive(os.path.join(dataset_train_path, class_name))
    #     nb_train_samples.append(samples)
    # train_labels = np.array(
    #     [0] * (nb_train_samples[0]) + [1] * (nb_train_samples[1]))                                # TODO  [0,0,0,1,1,1] if nb_train_samples=6; {{0 for dogs and 1 for cats}}

    # nb_validation_samples = []
    # for class_name in class_names:
    #     samples  = get_images_count_recursive(os.path.join(dataset_val_path, class_name))
    #     nb_validation_sampleves.append(samples)
    # validation_labels = np.array(
    #     [0] * (nb_validation_samples[0]) + [1] * (nb_validation_samples[1]))




    # Callbacks
    filename = 'output/model_train.csv'
    csv_log = CSVLogger(filename, separator=' ', append=False)

    early_stopping = EarlyStopping(
        monitor='loss', patience=500, verbose=1, mode='min')                                         # Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,dense_4_loss,dense_2_acc,dense_2_loss,dense_4_acc
        #monitor='val_loss', patience=50, verbose=1, mode='min')

    #filepath = "output/best-weights-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"
    filepath = "output/best-weights-{epoch:03d}-{val_loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
        save_best_only=True, save_weights_only=False, mode='min', period=1)                     # min because we are monitoring val_loss that should decrease

    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

    callbacks_list = [csv_log, early_stopping, checkpoint, tensorboard]
    logging.debug('callbacks_list {}'.format(callbacks_list))


    # class_names_sort, class_image_count_sort = get_dataset_class_image_count_sorted('train')
    # logging.debug('class_names_sort {}'.format(class_names_sort))
    # logging.debug('class_image_count_sort {}'.format(class_image_count_sort))

    # total = sum(int(i) for i in class_weight_dict)
    # logging.debug('total {}'.format(total))

    # class_weight = 1 - np.array(class_weight_dict).astype(float)/total
    # logging.debug('class_weight {}'.format(class_weight))

    # np.set_printoptions(formatter={'float': '{: 0.6f}'.format})
    # print(class_weight)


    # Generate weights based on images count for each class
    class_weight_val = class_weight.compute_class_weight('balanced', np.unique(train_labels), train_labels)
    logging.info('class_weight_val {}'.format(class_weight_val))
    logging.info('train_labels {}'.format(train_labels))
    logging.info('train_labels len {}'.format(len(train_labels)))


    if not is_grid_search:

        # Default:
        # optimizer='Adagrad'
        # learn_rate=None
        # decay=0.0
        # momentum=0.0
        # activation='relu'
        # dropout_rate=0.5

        optimizer='Adagrad'
        learn_rate=0.001
        #learn_rate=0.0001
        decay=0.0
        momentum=0.9
        activation='relu'
        dropout_rate=0.2

        model = create_model_func2((input_shape), optimizer, learn_rate, decay, momentum, activation, dropout_rate)

        # model = create_model(input_shape=(input_shape))


        model.fit(train_data, [train_labels, train_labels_iou],
                class_weight=[class_weight_val, class_weight_val],                                                      # dictionary mapping classes to a weight value, used for scaling the loss function (during training only).
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(validation_data, [validation_labels, validation_labels_iou]),
                callbacks=callbacks_list)

        # TODO: These are not the best weights
        model.save_weights(top_model_weights_path)

        predict_data = np.load(open('output/bottleneck_features_btl.npy'))
        # logging.debug('predict_data {}'.format(predict_data))
        logging.debug('predict_data {}'.format(len(predict_data)))
        input_shape = train_data.shape[1:]
        logging.debug('input_shape {}'.format(input_shape))

        #prediction = model.predict(img, batch_size, verbose=1)
        prediction = model.predict(predict_data, batch_size, verbose=1)
        logging.debug('\n\nprediction \n{}'.format(prediction))

        for index,preds in enumerate(prediction):
            for pred in preds:
                #logging.debug('pred {0:6f}'.format(float(pred)))
                logging.debug('pred {}'.format((pred)))

                for p in pred:
                    logging.debug('pred {:6f}'.format(float(p)))




    else:


	# Create model
        model = KerasClassifier(build_fn=create_model, verbose=1)


        # Define the grid search parameters

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

        optimizers      = ['Adagrad']
        learn_rates     = [0.001]
        decays          = [0.0]
        momentums       = [0.0]
        activations     = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
        dropout_rates   = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9]
        batch_sizes     = [32]
        epochs_grid     = [100]

        param_grid = dict(input_shape=[(input_shape)],
                          optimizer=optimizers,
                          learn_rate=learn_rates,
                          #decay=decays,
                          #momentum=momentums,
                          #activation=activations,
                          #dropout_rate=dropout_rates,
                          batch_size=batch_sizes,
                          epochs=epochs_grid)


        callbacks_list = [csv_log, early_stoppina, tensorboard]
        logging.debug('callbacks_list {}'.format(callbacks_list))

        fit_params = dict(class_weight=class_weight_val,
                          validation_data=(validation_data, validation_labels),
                          callbacks=callbacks_list)

        #grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, fit_params=fit_params, n_jobs=1, verbose=1)
        grid_result = grid.fit(train_data, train_labels)

        # Summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("Mean:%f (Stdev:%f) with: %r" % (mean, stdev, param))


### MAIN ###

seed = 7
np.random.seed(seed)

# TESTING
class_names = get_subdir_list(dataset_train_path)
#class_names = ['Anorak', 'Blazer', 'Blouse', 'Bomber', 'Button-Down', 'Cardigan', 'Chinos', 'Coat', 'Culottes', 'Cutoffs', 'Flannel', 'Henley', 'Hoodie', 'Jacket', 'Jeans', 'Jeggings', 'Jersey','Joggers', 'Jumpsuit', 'Kaftan', 'Kimono', 'Leggings', 'Parka', 'Poncho', 'Robe', 'Romper', 'Shorts', 'Skirt', 'Sweater', 'Sweatpants', 'Sweatshorts', 'Tank', 'Top', 'Trunks', 'Turtleneck']
logging.debug('class_names {}'.format(class_names))

# TODO Uncomment
# Batch size should be 1 for generating bottleneck otherwise errors like
# ValueError: Input arrays should have the same number of samples as target arrays. Found 2112 input samples and 2138 target samples.
#save_bottlebeck_features()

save_bottlebeck_features_btl()

is_grid_search=False
train_top_model(is_grid_search)











