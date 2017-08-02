### IMPORTS
from __future__ import print_function

import os
import fnmatch
import numpy as np
import skimage.data
import cv2
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.optimizers import RMSprop, Adagrad
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping

import logging
FORMAT = "[%(lineno)4s : %(funcName)-30s ] %(message)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT)

from selective_search import selective_search_bbox

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
#finetune_model_weights_path = 'output/finetune_bottleneck_fc_model.h5'
#finetune_model_weights_path = 'output_6_categ/best-weights-finetune-000-0.2325-0.9062.hdf5'

#finetune_model_weights_path = 'output_6_categ_crop/best-weights-finetune-008-0.3453-0.8774.hdf5'

#finetune_model_weights_path = 'output/best-weights-finetune-000-1.5646-0.5217.hdf5'
#finetune_model_weights_path = 'results_36categ/best-weights-finetune-000-1.5646-0.5217.hdf5'
finetune_model_weights_path = 'output/finetune_bottleneck_fc_model.h5'

#epochs = 50
epochs = 5
#batch_size = 16
#batch_size = 32
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

nb_test_samples = get_images_count_recursive(dataset_test_path)
logging.debug('nb_test_samples {}'.format(nb_test_samples))

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
logging.debug('Model loaded.')
logging.debug('{}'.format(base_model.output_shape))                                                                           # (None, None, None, 512) if input_shape not given in applications.VGG16
logging.debug('{}'.format(base_model.output_shape[1:]))                                                                       # (None, None, 512)


### MODEL 1
# build a classifier model to put on top of the convolutional model
# top_model = Sequential()
# top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
# top_model.add(Dense(256, activation='relu'))
# top_model.add(Dropout(0.5))
# top_model.add(Dense(len(class_names), activation='softmax'))                                        # Binary to Multi classification changes
# #top_model.add(Dense(1, activation='sigmoid'))

# # note that it is necessary to start with a fully-trained
# # classifier, including the top classifier,
# # in order to successfully do fine-tuning
# # top_model.load_weights(top_model_weights_path)

# # add the model on top of the convolutional base
# # base_model.add(top_model)                                                                                # Not working; AttributeError: 'Model' object has no attribute 'add'
# model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
# logging.debug('{}'.format(model.summary()))

# model.compile(loss='sparse_categorical_crossentropy',
#               optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
#               metrics=['accuracy'])








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

# This creates a model that includes the Input layer and three Dense layers
#model = Model(inputs=inputs, outputs=[predictions_class(base_model.output), predictions_iou(base_model.output)])
model = Model(inputs=inputs, outputs=[predictions_class(base_model.output), predictions_iou])

logging.debug('model summary {}'.format(model.summary()))


model.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              loss={'predictions_class': 'sparse_categorical_crossentropy', 'predictions_iou': 'mean_squared_error'},
              metrics=['accuracy'])
















model.load_weights(finetune_model_weights_path)
logging.debug('weights loaded: {}'.format(finetune_model_weights_path))


def evaluate_test_dataset():
## Test
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        dataset_test_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='sparse',                                                                           # Binary to Multi classification changes
        save_to_dir=None,
        shuffle=False)

    scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)

    logging.debug('model.metrics_names {}'.format(model.metrics_names))
    logging.debug('scores {}'.format(scores))


def predict_image_dir():
# Predict
# TODO: Hardcoding
# Put all images in sample_images/test folder
    dataset_predict_path='sample_images'
    #dataset_predict_path='temp'
    logging.debug('dataset_predict_path {}'.format(dataset_predict_path))

    predict_datagen = ImageDataGenerator(rescale=1. / 255)

    predict_generator = predict_datagen.flow_from_directory(
        dataset_predict_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='sparse',                                                                           # Binary to Multi classification changes
        save_to_dir=None,
        shuffle=False)

    nb_predict_samples = get_images_count_recursive(dataset_predict_path)
    logging.debug('nb_predict_samples {}'.format(nb_predict_samples))

    prediction = model.predict_generator(predict_generator, nb_predict_samples // batch_size, verbose=1)
    logging.debug('\n\nprediction \n{}'.format(prediction))


    # Display predictions
    matches=[]
    for root, dirnames, filenames in os.walk(os.path.join(dataset_predict_path,'test')):
        for filename in fnmatch.filter(filenames, '*.jpg'):
            matches.append(os.path.join(root, filename))

    for index,preds in enumerate(prediction):
        logging.debug('\n{}'.format((matches[index])))
        for index2, pred in enumerate(preds):
            logging.debug('class_names {}'.format(class_names[index2]))
            logging.debug('pred {0:6f}'.format(float(pred)))


def pad_and_crop_image(old_im, new_width, new_height):

    # old_im = Image.open('someimage.jpg')
    old_size = old_im.size

    new_size = (new_width, new_height)
    new_im = Image.new("RGB", new_size)   # this is already black!
    new_im.paste(old_im, ((new_size[0]-old_size[0])/2,
                                                (new_size[1]-old_size[1])/2))

    # new_im.show()
    # new_im.save('someimage.jpg')

    return new_im




def predict_image_name(image_path_name):

    logging.debug('image_path_name {}'.format(image_path_name))

    candidates = selective_search_bbox(image_path_name)
    logging.debug('candidates {}'.format(candidates))

    image_name = image_path_name.split('/')[-1].split('.')[0]
    logging.debug('image_name {}'.format(image_name))

    # img = Image.open(image_path_name)
    # logging.debug('{} {} {}'.format(img.format, img.size, img.mode))
    #img2 = img.crop((0, 0, 100, 100))
    # img2.save("img2.jpg")
    # img2.show()

    #crop_img = img[200:400, 100:300] # Crop from x, y, w, h -> 100, 200, 300, 400
    # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]


    # img = cv2.imread(image_path_name)
    # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))


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

#         crop_img = img[x:y, w:h] # Crop from x, y, w, h -> 100, 200, 300, 400
#         logging.debug('crop_img {}'.format(crop_img.shape))
#         ax.imshow(crop_img)
#         # cv2.imshow('cropped', crop_img)
#         # cv2.waitKey(0)
#         plt.show()


# # Convert Image to array
# img = PIL.Image.open("foo.jpg").convert("L")
# arr = numpy.array(img)

# # Convert array to Image
# img = PIL.Image.fromarray(arr)

#       img = cv2.resize(cv2.imread(image_path_name), (224, 224)).astype(np.float32)

#       img2.save('temp/test/img_'+str(i)+'.jpg')

#         img3 = img2.thumbnail((img_width, img_height))
#         logging.debug('img3 {}'.format(type(img3)))
#         # img3.save('temp/test/img_'+str(i)+'_resized.jpg')
#         logging.debug('{} {} {}'.format(img3.format, img3.size, img3.mode))

#         img4 = pad_and_crop_image(img3, img_width, img_height)
#         logging.debug('{} {} {}'.format(img4.format, img4.size, img4.mode))
#         img4.save('temp/test/img_'+str(i)+'_resized1.jpg')


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




### MAIN ###

#evaluate_test_dataset()

#predict_image_dir()


# #image='dataset/test/Jeans/img_Distressed_Skinny_Jeans_img_00000004.jpg'
# #image='sample_images/test/img_Distressed_Denim_Jeans_img_00000001.jpg'
# image='sample_images/test/img_Acid_Wash_Denim_Romper_img_00000070.jpg'
image='sample_images/test/img_Acid_Wash_-_Skinny_Jeans_img_00000005.jpg'
#image='sample_images/test/img_Boxy_Faux_Fur_Jacket_img_00000001.jpg'
#image='sample_images/test/img_Athletic_Marled_Knit_Joggers_img_00000009.jpg'

predict_image_name(image)




