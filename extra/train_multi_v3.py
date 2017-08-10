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



VGG16 for input image of size (224, 224, 3)

Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
input_2 (InputLayer)             (None, 224, 224, 3)   0
____________________________________________________________________________________________________
block1_conv1 (Conv2D)            (None, 224, 224, 64)  1792        input_2[0][0]
____________________________________________________________________________________________________
block1_conv2 (Conv2D)            (None, 224, 224, 64)  36928       block1_conv1[0][0]
____________________________________________________________________________________________________
block1_pool (MaxPooling2D)       (None, 112, 112, 64)  0           block1_conv2[0][0]
____________________________________________________________________________________________________
block2_conv1 (Conv2D)            (None, 112, 112, 128) 73856       block1_pool[0][0]
____________________________________________________________________________________________________
block2_conv2 (Conv2D)            (None, 112, 112, 128) 147584      block2_conv1[0][0]
____________________________________________________________________________________________________
block2_pool (MaxPooling2D)       (None, 56, 56, 128)   0           block2_conv2[0][0]
____________________________________________________________________________________________________
block3_conv1 (Conv2D)            (None, 56, 56, 256)   295168      block2_pool[0][0]
____________________________________________________________________________________________________
block3_conv2 (Conv2D)            (None, 56, 56, 256)   590080      block3_conv1[0][0]
____________________________________________________________________________________________________
block3_conv3 (Conv2D)            (None, 56, 56, 256)   590080      block3_conv2[0][0]
____________________________________________________________________________________________________
block3_pool (MaxPooling2D)       (None, 28, 28, 256)   0           block3_conv3[0][0]
____________________________________________________________________________________________________
block4_conv1 (Conv2D)            (None, 28, 28, 512)   1180160     block3_pool[0][0]
____________________________________________________________________________________________________
block4_conv2 (Conv2D)            (None, 28, 28, 512)   2359808     block4_conv1[0][0]
____________________________________________________________________________________________________
block4_conv3 (Conv2D)            (None, 28, 28, 512)   2359808     block4_conv2[0][0]
____________________________________________________________________________________________________
block4_pool (MaxPooling2D)       (None, 14, 14, 512)   0           block4_conv3[0][0]
____________________________________________________________________________________________________
block5_conv1 (Conv2D)            (None, 14, 14, 512)   2359808     block4_pool[0][0]
____________________________________________________________________________________________________
block5_conv2 (Conv2D)            (None, 14, 14, 512)   2359808     block5_conv1[0][0]
____________________________________________________________________________________________________
block5_conv3 (Conv2D)            (None, 14, 14, 512)   2359808     block5_conv2[0][0]
____________________________________________________________________________________________________
block5_pool (MaxPooling2D)       (None, 7, 7, 512)     0           block5_conv3[0][0]                   <--- bottleneck_features_train
____________________________________________________________________________________________________


Bottleneck features are saved alphabetically(class names). So we are generating train_labels using this assumption
and hence cannot shuffle data

'''

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

    logging.info('optimizer:{}  learn_rate:{}  decay:{}  momentum:{}  activation:{}  dropout_rate:{}'.format(
        optimizer, learn_rate, decay, momentum, activation, dropout_rate))

    return model


# INPUT:
#           Input Image (None, 224, 224, 3) [fed to VGG16]
# OUTPUT:
#           Branch1 - Class Prediction
#           Branch2 - IOU Prediction

# NOTE: Both models in create_model_train() and  create_model_predict() should be exaclty same
def create_model_predict(input_shape, optimizer='Adagrad', learn_rate=None, decay=0.0, momentum=0.0, activation='relu', dropout_rate=0.5):
    logging.debug('input_shape {}'.format(input_shape))
    logging.debug('input_shape {}'.format(type(input_shape)))

    # Optimizer
    optimizer, learn_rate = get_optimizer(optimizer, learn_rate, decay, momentum)

    input_shape = (img_width, img_height, 3)                                                        # 224,224,3
    base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    logging.debug('base_model inputs {}'.format(base_model.input))                                  # shape=(?, 224, 224, 3)
    logging.debug('base_model outputs {}'.format(base_model.output))                                # shape=(?, 224, 224, 3)


    # TODO: Hardcoding
    input_shape_top_model_tensor = Input(shape=(7, 7, 512))
    #x_common = Dense(256, activation='relu')(input_shape_top_model)
    # x_common = Dense(256, activation='relu')(base_model.output)

    ## Model Classification
    # x = Flatten()(x_common)
    x = Flatten()(base_model.output)
    x = Dense(256, activation='tanh')(x)
    x = Dropout(dropout_rate)(x)
    predictions_class = Dense(len(class_names), activation='softmax', name='predictions_class')(x)


    ## Model (Regression) IOU score
    # x = Flatten()(x_common)
    x = Flatten()(base_model.output)
    x = Dense(256, activation='tanh')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(256, activation='tanh')(x)
    x = Dropout(dropout_rate)(x)
    predictions_iou = Dense(1, activation='sigmoid', name='predictions_iou')(x)

    # model_top.load_weights(top_model_weights_path)

    # This creates a model that includes the Input layer and three Dense layers
    logging.debug('Creating model {}')
    #model = Model(inputs=base_model.input, outputs=[predictions_class(base_model.output), predictions_iou(base_model.output)])
    model = Model(inputs=base_model.input, outputs=[predictions_class, predictions_iou])

    logging.debug('model summary {}'.format(model.summary()))

    # TODO: loads only top model weights as only those are present(also saved by name) in the file.
    # Test this assumption
    model.load_weights(top_model_weights_path, by_name=True)


    # Compile
    model.compile(optimizer=optimizer,
                  loss={'predictions_class': 'sparse_categorical_crossentropy', 'predictions_iou': 'mean_squared_error'}, metrics=['accuracy'],
                  loss_weights={'predictions_class': 0.5, 'predictions_iou': 0.5})
                  #loss={'predictions_class': 'sparse_categorical_crossentropy', 'predictions_iou': 'logcosh'}, metrics=['accuracy'],

    logging.info('optimizer:{}  learn_rate:{}  decay:{}  momentum:{}  activation:{}  dropout_rate:{}'.format(
        optimizer, learn_rate, decay, momentum, activation, dropout_rate))

    return model



def train_model():

    ## Variables
    batch_size = 1
    epochs = 20

    ## Preprocessing
    # VGG16 - VGG image prepossessing mean and no scale
    #datagen = ImageDataGenerator(rescale=1. / 255)                     # Not requried for VGG (check)
    datagen = ImageDataGenerator(rescale=1., featurewise_center=True)
    datagen.mean=np.array([103.939, 116.779, 123.68],dtype=np.float32)
    #datagen.mean=np.array([103.939, 116.779, 123.68],dtype=np.float32).reshape(3,1,1)


    ## Build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet', input_shape=(img_width, img_height, 3))                               # exclude 3 FC layers on top of network


    ## Training Data
    score_iou_train_g, nb_train_samples = get_images_count_recursive(dataset_train_path)
    logging.debug('nb_train_samples {}'.format(nb_train_samples))
    score_iou_validation_g, nb_validation_samples = get_images_count_recursive(dataset_val_path)
    logging.debug('nb_validation_samples {}'.format(nb_validation_samples))

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
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    logging.debug('bottleneck_features_train {}'.format(bottleneck_features_train.shape))           # bottleneck_features_train (10534, 4, 4, 512) where train images i.e Blazer+Jeans=5408+5126=10532 images;

    # logging.debug('Saving bottleneck_features_train...')
    # np.save(open('output/bottleneck_features_train.npy', 'w'),
    #         bottleneck_features_train)


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

    # logging.debug('Saving bottleneck_features_validation...')
    # np.save(open('output/bottleneck_features_validation.npy', 'w'),
    #         bottleneck_features_validation)


    train_data = bottleneck_features_train
    input_shape = train_data.shape[1:]
    validation_data = bottleneck_features_validation


    ## Label generation
    train_labels = []
    train_labels_iou = []
    for index, class_name in enumerate(class_names):
        score_iou_train, images_count  = get_images_count_recursive(os.path.join(dataset_train_path, class_name))
        train_labels_iou = train_labels_iou + score_iou_train
        for _ in range(images_count):
            train_labels.append(index)
    train_labels = np.array(train_labels)
    train_labels_iou_arr = np.array(train_labels_iou)
    train_labels_iou = train_labels_iou_arr.astype(np.float)
    train_labels_iou = np.round(train_labels_iou, 2)
    #train_labels_iou = np.random.randint(low=1, high=9, size=2138).astype(np.float)/100

    logging.debug('train_labels_iou len {}'.format(len(train_labels_iou)))
    logging.debug('train_labels_iou {}'.format(train_labels_iou))
    logging.debug('train_labels {}'.format(train_labels))
    logging.debug('train_labels len {}'.format(len(train_labels)))
    logging.debug('train_labels type {}'.format(type(train_labels)))
    logging.debug('train_labels_iou {}'.format(train_labels_iou))
    logging.debug('train_labels_iou len {}'.format(len(train_labels_iou)))
    logging.debug('train_labels_iou type {}'.format(type(train_labels_iou)))
    logging.debug('train_labels_iou type {}'.format(type(train_labels_iou[0])))


    validation_labels = []
    validation_labels_iou = []
    for index, class_name in enumerate(class_names):
        score_iou_validation, images_count  = get_images_count_recursive(os.path.join(dataset_val_path, class_name))
        validation_labels_iou = validation_labels_iou + score_iou_validation
        for _ in range(images_count):
            validation_labels.append(index)
    validation_labels = np.array(validation_labels)
    validation_labels_iou_arr = np.array(validation_labels_iou)
    validation_labels_iou = validation_labels_iou_arr.astype(np.float)
    validation_labels_iou = np.round(validation_labels_iou, 2)
    #validation_labels_iou = np.random.randint(low=1, high=9, size=400).astype(np.float)/100

    logging.debug('validation_labels {}'.format(validation_labels))
    logging.debug('validation_labels len {}'.format(len(validation_labels)))
    logging.debug('validation_labels type {}'.format(type(validation_labels)))
    logging.debug('validation_labels_iou {}'.format(validation_labels_iou))
    logging.debug('validation_labels_iou len {}'.format(len(validation_labels_iou)))
    logging.debug('validation_labels_iou type {}'.format(type(validation_labels_iou)))


    ## Callbacks
    filename = 'output/model_train.csv'
    csv_log = CSVLogger(filename, separator=' ', append=False)

    early_stopping = EarlyStopping(
        monitor='loss', patience=500, verbose=1, mode='min')                                         # Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,dense_4_loss,dense_2_acc,dense_2_loss,dense_4_acc
        #monitor='val_loss', patience=50, verbose=1, mode='min')

    #filepath = "output/best-weights-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"
    filepath = "output/best-weights-{epoch:03d}-{val_loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
        save_best_only=True, save_weights_only=False, mode='min', period=1)                         # min because we are monitoring val_loss that should decrease

    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

    callbacks_list = [csv_log, early_stopping, checkpoint, tensorboard]
    logging.debug('callbacks_list {}'.format(callbacks_list))


    # Generate weights based on images count for each class
    class_weight_val = class_weight.compute_class_weight('balanced', np.unique(train_labels), train_labels)
    logging.info('class_weight_val {}'.format(class_weight_val))
    logging.info('train_labels {}'.format(train_labels))
    logging.info('train_labels len {}'.format(len(train_labels)))


    optimizer='Adagrad'
    learn_rate=0.001
    decay=0.0
    momentum=0.0
    activation='relu'
    dropout_rate=0.5

    model = create_model_train((input_shape), optimizer, learn_rate, decay, momentum, activation, dropout_rate)

    model.fit(train_data, [train_labels, train_labels_iou],
            class_weight=[class_weight_val, class_weight_val],                                      # dictionary mapping classes to a weight value, used for scaling the loss function (during training only).
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(validation_data, [validation_labels, validation_labels_iou]),
            callbacks=callbacks_list)

    # TODO: These are not the best weights
    model.save_weights(top_model_weights_path)


def prediction_model_a():

    # METHOD A

    ## Variables
    batch_size = 1
    epochs = 20

    dataset_train_path='dataset_prediction/train'

    ## Preprocessing
    # VGG16 - VGG image prepossessing mean and no scale
    #datagen = ImageDataGenerator(rescale=1. / 255)                                                 # Not requried for VGG (check)
    datagen = ImageDataGenerator(rescale=1., featurewise_center=True)
    datagen.mean=np.array([103.939, 116.779, 123.68],dtype=np.float32)
    #datagen.mean=np.array([103.939, 116.779, 123.68],dtype=np.float32).reshape(3,1,1)


    ## Build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet', input_shape=(img_width, img_height, 3))                               # exclude 3 FC layers on top of network


    ## Training Data
    score_iou_train_g, nb_train_samples = get_images_count_recursive(dataset_train_path)
    logging.debug('nb_train_samples {}'.format(nb_train_samples))
    score_iou_validation_g, nb_validation_samples = get_images_count_recursive(dataset_val_path)
    logging.debug('nb_validation_samples {}'.format(nb_validation_samples))

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
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    logging.debug('bottleneck_features_train {}'.format(bottleneck_features_train.shape))           # bottleneck_features_train (10534, 4, 4, 512) where train images i.e Blazer+Jeans=5408+5126=10532 images;

    # logging.debug('Saving bottleneck_features_train...')
    # np.save(open('output/bottleneck_features_train.npy', 'w'),
    #         bottleneck_features_train)


    #train_data = bottleneck_features_train
    train_data = generator

    # predict_data = np.load(open('output/bottleneck_features_btl.npy'))
    #predict_data = bottleneck_features_train
    predict_data = generator
    # logging.debug('predict_data {}'.format(predict_data))
    #logging.debug('predict_data {}'.format(len(predict_data)))


def prediction_model():

    # METHOD B

    ## Variables
    batch_size = 1
    epochs = 20


    # Hardcoding
    #input_shape = train_data.shape[1:]
    input_shape = (img_width, img_height,3)
    logging.debug('input_shape {}'.format(input_shape))

    optimizer='Adagrad'
    learn_rate=0.001
    decay=0.0
    momentum=0.0
    activation='relu'
    dropout_rate=0.5

    model_top = create_model_predict((input_shape), optimizer, learn_rate, decay, momentum, activation, dropout_rate)
    # model_top.load_weights(top_model_weights_path)


    # image_path_name='dataset_btl/train/Robe/Lace-Paneled_Satin_Robe_img_00000002_gt_iou_1.0.jpg'
    # img = Image.open(image_path_name)

    images_path_name = sorted(glob.glob('dataset_prediction/validation/images/*.jpg'))
    # logging.debug('images_path_name {}'.format(images_path_name))


    images_list = []
    images_name_list = []
    for image in images_path_name:
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

        images_list.append(img)
        images_name_list.append(image)

    logging.debug('images_list {}'.format(len(images_list)))
    images_list_arr = np.array(images_list)
    logging.debug('images_list_arr type {}'.format(type(images_list_arr)))

    prediction = model_top.predict(images_list_arr, batch_size, verbose=1)
    #prediction = model_top.predict(predict_data, batch_size, verbose=1)
    # logging.debug('\n\nprediction \n{}'.format(prediction))
    logging.debug('prediction shape {} {}'.format(len(prediction), len(prediction[0])))
    print('')

    for index,preds in enumerate(prediction):
        for index2, pred in enumerate(preds):
            print('images_name_list index2 : {:110}     '.format(images_name_list[index2]), end='')
            for p in pred:
                print('{:8f} '.format(float(p)), end='')
            print('')
        print('')






### MAIN ###

seed = 7
np.random.seed(seed)

class_names = get_subdir_list(dataset_train_path)
logging.debug('class_names {}'.format(class_names))

#train_model()                                                                                       # Save weights at output/bottleneck_fc_model.h5
prediction_model()                                                                                  # Load weights VGG16 and bottleneck_fc_model.h5 weigths and predict







