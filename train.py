#!/usr/bin/python

### IMPORTS
from __future__ import print_function

from config import *
from utils import *
from model import *


### GLOBALS
class_names=[]
batch_size = 0
input_shape=(0,0,0)


### FUNCTIONS ###

def init():

    global batch_size
    batch_size = batch_size_train
    logging.debug('batch_size {}'.format(batch_size))

    global class_names
    class_names = sorted(get_subdir_list(dataset_train_path))
    logging.debug('class_names {}'.format(class_names))

    global input_shape
    input_shape = (img_width, img_height, img_channel)
    logging.debug('input_shape {}'.format(input_shape))


    if not os.path.exists(output_path_name):
        os.makedirs(output_path_name)

    if not os.path.exists(logs_path_name):
        os.makedirs(logs_path_name)

    if not os.path.exists(btl_path):
        os.makedirs(btl_path)

    if not os.path.exists(btl_train_path):
        os.makedirs(btl_train_path)

    if not os.path.exists(btl_val_path):
        os.makedirs(btl_val_path)


def save_bottleneck():
    logging.debug('class_names {}'.format(class_names))
    logging.debug('batch_size {}'.format(batch_size))
    logging.debug('epochs {}'.format(epochs))
    logging.debug('input_shape {}'.format(input_shape))


    ## Build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    #model = applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape)


    for train_val in ['train', 'validation']:

        with open('bottleneck/btl_' + train_val + '.txt', 'w') as f_image:
            for class_name in class_names:
                dataset_train_class_path = os.path.join(dataset_path, train_val, class_name)
                logging.debug('dataset_train_class_path {}'.format(dataset_train_class_path))

                images_list = []
                images_name_list = []

                images_path_name = sorted(glob.glob(dataset_train_class_path + '/*.jpg'))
                logging.debug('images_path_name {}'.format(len(images_path_name)))

                for index, image in enumerate(images_path_name):
                    # logging.debug('image {}'.format(image))

                    img = Image.open(image)

                    img = preprocess_image(img)

                    current_batch_size = len(images_list)
                    # logging.debug('current_batch_size {}'.format(current_batch_size))

                    images_list.append(img)

                    image_name = image.split('/')[-1].split('.jpg')[0]
                    images_name_list.append(image)
                    images_list_arr = np.array(images_list)

                    # TODO: Skipping n last images of a class which do not sum up to batch_size
                    if (current_batch_size < batch_size-1):
                        continue


                    X = images_list_arr

                    bottleneck_features_train_class = model.predict(X, batch_size)
                    # bottleneck_features_train_class = model.predict(X, nb_train_class_samples // batch_size)


                    ## Save bottleneck file
                    btl_save_file_name = btl_path + train_val + '/btl_' + train_val + '_' + class_name + '.' + str(index).zfill(7) + '.npy'
                    logging.info('btl_save_file_name {}'.format(btl_save_file_name))
                    np.save(open(btl_save_file_name, 'w'), bottleneck_features_train_class)
                    for name in images_name_list:
                        f_image.write(str(name) + '\n')

                    images_list = []
                    images_name_list = []




def train_model():

    ## Build network
    model = applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    #model = applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape)



    # Get sorted bottleneck file names in a list
    btl_train_names = sorted(glob.glob(btl_train_path + '/*.npy'))
    btl_val_names = sorted(glob.glob(btl_val_path + '/*.npy'))



    ## Train Labels
    btl_train_list = []
    train_labels_class = []
    train_labels_iou = []

    # Get list of image IoU values
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
    # logging.debug('train_labels_class {}'.format(train_labels_class))

    train_labels_class = np.array(train_labels_class)
    train_labels_iou = np.array(train_labels_iou)
    logging.debug('train_labels_iou {}'.format(train_labels_iou))
    logging.debug('train_labels_iou {}'.format(type(train_labels_iou)))
    logging.debug('train_labels_class {}'.format(type(train_labels_class)))
    logging.debug('train_labels_class {}'.format((train_labels_class.shape)))

    # Load bottleneck files to create train set
    train_data = []
    for index, btl_name in enumerate(btl_train_names):
        temp = np.load(open(btl_name))
        train_data.append(temp)

    train_data = np.array(train_data)
    n1, n2, w, h, c = train_data.shape
    logging.info('train_data {}'.format(train_data.shape))
    train_data_ = train_data
    train_data = np.reshape(train_data_, (n1*n2, w, h, c))
    logging.info('train_data {}'.format(train_data.shape))



    ## Validation Labels
    btl_val_list = []
    val_labels_class = []
    val_labels_iou = []

    # Get list of image IoU values
    with open('bottleneck/btl_validation.txt') as f_btl_val:
        btl_val_list = f_btl_val.readlines()
        # logging.debug('btl_val_list {}'.format(btl_val_list))
    for btl_val_image in btl_val_list:
        val_labels_class.append(btl_val_image.split('/')[2])
        val = np.round(np.float( btl_val_image.split('_')[-1].split('.jpg')[0] ), 2)
        val_labels_iou.append(val)
        # logging.debug('val {}'.format(val))

    # logging.debug('val_labels_class {}'.format(val_labels_class))
    val_labels_class_int = []
    for index, class_name in enumerate(val_labels_class):
        val_labels_class_int.append(class_names.index(class_name))
    val_labels_class = val_labels_class_int
    # logging.debug('val_labels_class {}'.format(val_labels_class))

    val_labels_class = np.array(val_labels_class)
    # logging.debug('val_labels_class {}'.format(val_labels_class))
    val_labels_iou = np.array(val_labels_iou)
    # logging.debug('val_labels_iou {}'.format(val_labels_iou))
    logging.debug('val_labels_iou {}'.format(type(val_labels_iou)))
    logging.debug('val_labels_class {}'.format(type(val_labels_class)))
    logging.debug('val_labels_class {}'.format(val_labels_class.shape))

    # Load bottleneck files to create validation set
    val_data = []
    for index, btl_name in enumerate(btl_val_names):
        temp = np.load(open(btl_name))
        val_data.append(temp)

    val_data = np.array(val_data)
    n1, n2, w, h, c = val_data.shape
    logging.info('val_data {}'.format(val_data.shape))
    val_data_ = val_data
    val_data = np.reshape(val_data_, (n1*n2, w, h, c))
    logging.info('val_data {}'.format(val_data.shape))



    ## Register Callbacks
    filename = 'output/model_train.csv'
    csv_log = CSVLogger(filename, separator=' ', append=False)

    early_stopping = EarlyStopping(
        monitor='loss', patience=early_stopping_patience, verbose=1, mode='min')

    #filepath = "output/best-weights-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"
    filepath = "output/best-weights-{epoch:03d}-{val_loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
        save_best_only=True, save_weights_only=False, mode='min', period=1)

    tensorboard = TensorBoard(log_dir=logs_path_name, histogram_freq=0, write_graph=True, write_images=True)

    callbacks_list = [csv_log, early_stopping, checkpoint, tensorboard]
    logging.info('callbacks_list {}'.format(callbacks_list))


    ## Generate weights based on images count for each class
    class_weight_val = class_weight.compute_class_weight('balanced', np.unique(train_labels_class), train_labels_class)
    logging.debug('class_weight_val {}'.format(class_weight_val))




    input_shape_btl_layer = train_data.shape[1:]
    logging.debug('input_shape_btl_layer {}'.format(input_shape_btl_layer))
    #model = create_model(is_input_bottleneck=True, is_load_weights=False, input_shape, optimizer, learn_rate, decay, momentum, activation, dropout_rate)
    model = create_model(True, False, input_shape_btl_layer, len(class_names), optimizer, learn_rate, decay, momentum, activation, dropout_rate)

    logging.info('train_labels_iou {}'.format(train_labels_iou.shape))
    logging.info('train_labels_class {}'.format(train_labels_class.shape))
    logging.info('train_data {}'.format(train_data.shape))


    logging.info('val_labels_iou {}'.format(val_labels_iou.shape))
    logging.info('val_labels_class {}'.format(val_labels_class.shape))
    logging.info('val_data {}'.format(val_data.shape))

    # TODO: class_weight_val wrong
    model.fit(train_data, [train_labels_class, train_labels_iou],
            class_weight=[class_weight_val, class_weight_val],                                      # dictionary mapping classes to a weight value, used for scaling the loss function (during training only).
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(val_data, [val_labels_class, val_labels_iou]),
            callbacks=callbacks_list)

    # TODO: These are not the best weights
    model.save_weights(top_model_weights_path_save)



### MAIN ###

if __name__ == '__main__':

    init()
    save_bottleneck()
    train_model()







