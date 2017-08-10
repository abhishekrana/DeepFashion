#!/usr/bin/python

### IMPORTS
from __future__ import print_function

import config
from config import *
from utils import *


# INPUT:
#           VGG16 - block5_pool (MaxPooling2D) (None, 7, 7, 512)
# OUTPUT:
#           Branch1 - Class Prediction
#           Branch2 - IOU Prediction

# NOTE: Both models in create_model_train() and  create_model_predict() should be exaclty same
def create_model(is_input_bottleneck, is_load_weights, input_shape, output_classes, optimizer='Adagrad', learn_rate=None, decay=0.0, momentum=0.0, activation='relu', dropout_rate=0.5):

    logging.debug('input_shape {}'.format(input_shape))
    logging.debug('input_shape {}'.format(type(input_shape)))

    # Optimizer
    optimizer, learn_rate = get_optimizer(optimizer, learn_rate, decay, momentum)


    # Train
    if is_input_bottleneck is True:
        model_inputs = Input(shape=(input_shape))
        common_inputs = model_inputs

    # Predict
    else:                                                                                               #input_shape = (img_width, img_height, 3)
        base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        #base_model = applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape)
        logging.debug('base_model inputs {}'.format(base_model.input))                                  # shape=(?, 224, 224, 3)
        logging.debug('base_model outputs {}'.format(base_model.output))                                # shape=(?, 7, 7, 512)

        model_inputs = base_model.input
        common_inputs = base_model.output



    ## Model Classification
    x = Flatten()(common_inputs)
    x = Dense(256, activation='tanh')(x)
    x = Dropout(dropout_rate)(x)
    predictions_class = Dense(output_classes, activation='softmax', name='predictions_class')(x)


    ## Model (Regression) IOU score
    x = Flatten()(common_inputs)
    x = Dense(256, activation='tanh')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(256, activation='tanh')(x)
    x = Dropout(dropout_rate)(x)
    predictions_iou = Dense(1, activation='sigmoid', name='predictions_iou')(x)


    ## Create Model
    model = Model(inputs=model_inputs, outputs=[predictions_class, predictions_iou])
    # logging.debug('model summary {}'.format(model.summary()))


    ## Load weights
    if is_load_weights is True:
        model.load_weights(top_model_weights_path_load, by_name=True)


    ## Compile
    model.compile(optimizer=optimizer,
                  loss={'predictions_class': 'sparse_categorical_crossentropy', 'predictions_iou': 'mean_squared_error'}, metrics=['accuracy'],
                  loss_weights={'predictions_class': predictions_class_weight, 'predictions_iou': predictions_iou_weight})

    logging.info('optimizer:{}  learn_rate:{}  decay:{}  momentum:{}  activation:{}  dropout_rate:{}'.format(
        optimizer, learn_rate, decay, momentum, activation, dropout_rate))

    return model


