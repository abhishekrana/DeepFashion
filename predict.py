#!/usr/bin/python

### IMPORTS
from __future__ import print_function

from config import *
from utils import *
from model import *
from selective_search import selective_search_bbox

### GLOBALS
class_names=[]
batch_size = 0
input_shape=(0,0,0)



def init():

    global batch_size
    batch_size = batch_size_predict
    logging.debug('batch_size {}'.format(batch_size))

    global input_shape
    input_shape = (img_width, img_height, img_channel)
    logging.debug('input_shape {}'.format(input_shape))

    global class_names
    # class_names = ['Anorak', 'Bomber', 'Button-Down', 'Capris', 'Chinos', 'Coat', 'Flannel', 'Hoodie', 'Jeans', 'Jeggings', 'Jersey', 'Kaftan', 'Parka', 'Peacoat', 'Poncho', 'Robe', 'Sweatshorts', 'Trunks', 'Turtleneck']
    class_names = get_subdir_list(dataset_train_path)
    logging.debug('class_names {}'.format(class_names))


def get_images():
    images_path_name = sorted(glob.glob(prediction_dataset_path + '/*.jpg'))
    # logging.debug('images_path_name {}'.format(images_path_name))

    return images_path_name


def get_bbox(images_path_name):
    # TODO: Currently for 1 image only
    for index, image in enumerate(images_path_name):
        bboxes = selective_search_bbox(image)
        logging.debug('bboxes {}'.format(bboxes))
        return bboxes



def predict_model(images, images_names=None):

    #model = create_model_predict((input_shape), optimizer, learn_rate, decay, momentum, activation, dropout_rate)
    model = create_model(False, True, input_shape, len(class_names), optimizer, learn_rate, decay, momentum, activation, dropout_rate)


    images_list = []
    images_name_list = []
    images_name_list2 = []
    prediction_class = []
    prediction_iou = []
    prediction_class_prob = []
    prediction_class_name = []


    ## Folder
    prediction_dataset_path='dataset_prediction/crops/'
    #images_path_name = sorted(glob.glob(prediction_dataset_path + '/*.jpg'))
    #for image in images_path_name:
    for index, image in enumerate(images_names):
        logging.debug('\n\n++++++++++++++++++++++++++++++++++++++++')
        image_path_name = prediction_dataset_path + image
        logging.debug('image_path_name {}'.format(image_path_name))

        img = Image.open(image_path_name)
        logging.debug('img {}'.format(img))
        logging.debug('img len {}'.format((img.size)))

        #img.save('output/a' + str(index) + '.jpg')

        img = preprocess_image(img)
        img = np.expand_dims(img, 0)

        prediction = model.predict(img, batch_size, verbose=1)
        # logging.debug('prediction {}'.format(prediction))

        prediction_class_=prediction[0][0]
        # logging.debug('prediction_class_ {}'.format(prediction_class_))
        prediction_class.append(prediction_class_)

        prediction_iou_ = prediction[1][0][0]
        logging.debug('prediction_iou_ {}'.format(prediction_iou_))
        prediction_iou.append(prediction_iou_)

        prediction_class_index = np.argmax(prediction[0])
        logging.debug('prediction_class_index {}'.format(prediction_class_index))

        prediction_class_prob_ = prediction[0][0][prediction_class_index]
        logging.debug('prediction_class_prob_ {}'.format(prediction_class_prob_))
        prediction_class_prob.append(prediction_class_prob_)

        prediction_class_name_ = class_names[prediction_class_index]
        logging.debug('prediction_class_name_ {}'.format(prediction_class_name_))
        prediction_class_name.append(prediction_class_name_)


        images_list.append(img)
        images_name_list.append(image_path_name)

    # logging.debug('prediction_class {}'.format(prediction_class))
    logging.debug('prediction_iou {}'.format(prediction_iou))
    logging.debug('prediction_class_prob {}'.format(prediction_class_prob))
    logging.debug('prediction_class_name {}'.format(prediction_class_name))

    # logging.debug('images_name_list {}'.format(images_name_list))




    bboxes = []
    for image_path_name in images_name_list:
        bbox_=image_path_name.split('/')[-1].split('.jpg')[0].split('-')[1]
        x = int(bbox_.split('_')[0])
        y = int(bbox_.split('_')[1])
        w = int(bbox_.split('_')[2])
        h = int(bbox_.split('_')[3])
        bbox = (x, y, w, h)
        bboxes.append(bbox)
    bboxes = set(bboxes)
    logging.debug('bboxes {}'.format(bboxes))

    # bboxes = set([(284, 372, 766, 1508), (250, 304, 806, 1576), (423, 1635, 130, 147), (250, 304, 257, 326), (477, 642, 488, 763), (659, 1003, 119, 210), (668, 509, 376, 210), (325, 383, 103, 203), (461, 844, 117, 152), (577, 375, 54, 95), (0, 0, 1279, 1919), (0, 12, 1233, 1907), (676, 456, 92, 96), (730, 1678, 108, 206), (512, 642, 277, 470), (730, 1678, 114, 206), (510, 1409, 150, 133), (512, 653, 277, 304), (461, 844, 56, 96), (672, 509, 86, 122), (555, 1183, 232, 364), (512, 653, 277, 459), (461, 1096, 52, 88), (250, 304, 257, 300), (250, 304, 257, 289), (304, 506, 352, 194), (408, 460, 143, 184), (250, 304, 800, 1576), (456, 1175, 56, 104), (457, 889, 96, 169), (418, 832, 160, 175), (545, 452, 168, 235), (499, 23, 248, 280), (668, 528, 376, 191), (700, 993, 67, 71), (284, 561, 124, 69), (461, 1025, 113, 159), (697, 372, 234, 299), (436, 1469, 268, 360), (729, 93, 251, 335), (903, 648, 118, 219), (698, 322, 84, 163), (582, 44, 123, 227), (457, 753, 98, 104), (546, 26, 299, 524), (706, 736, 63, 91), (308, 608, 268, 245), (594, 898, 62, 72), (447, 26, 398, 566), (568, 653, 221, 162), (536, 1409, 124, 133), (499, 23, 481, 405), (499, 1025, 75, 92), (323, 383, 105, 203), (679, 528, 365, 191), (576, 506, 80, 119), (436, 1376, 376, 504), (250, 304, 806, 1578), (456, 1378, 129, 119), (489, 962, 88, 107), (284, 524, 131, 106), (903, 382, 147, 187), (545, 452, 181, 235), (730, 1678, 108, 193), (284, 1651, 169, 146)])
    # bboxes = set([(284, 372, 766, 1508), (250, 304, 806, 1576), (423, 1635, 130, 147), (250, 304, 257, 326), (477, 642, 488, 763), (659, 1003, 119, 210), (668, 509, 376, 210), (325, 383, 103, 203), (461, 844, 117, 152), (577, 375, 54, 95), (0, 0, 1279, 1919), (0, 12, 1233, 1907), (676, 456, 92, 96), (730, 1678, 108, 206), (512, 642, 277, 470), (730, 1678, 114, 206), (510, 1409, 150, 133), (512, 653, 277, 304), (461, 844, 56, 96), (672, 509, 86, 122), (555, 1183, 232, 364), (512, 653, 277, 459), (461, 1096, 52, 88), (250, 304, 257, 300), (250, 304, 257, 289), (304, 506, 352, 194), (408, 460, 143, 184), (250, 304, 800, 1576), (456, 1175, 56, 104), (457, 889, 96, 169), (418, 832, 160, 175), (545, 452, 168, 235), (499, 23, 248, 280), (668, 528, 376, 191), (700, 993, 67, 71), (284, 561, 124, 69), (461, 1025, 113, 159), (697, 372, 234, 299), (436, 1469, 268, 360), (729, 93, 251, 335), (903, 648, 118, 219), (698, 322, 84, 163), (582, 44, 123, 227), (457, 753, 98, 104), (546, 26, 299, 524), (706, 736, 63, 91), (308, 608, 268, 245), (594, 898, 62, 72), (447, 26, 398, 566), (568, 653, 221, 162), (536, 1409, 124, 133), (499, 23, 481, 405), (499, 1025, 75, 92), (323, 383, 105, 203), (679, 528, 365, 191), (576, 506, 80, 119), (436, 1376, 376, 504), (250, 304, 806, 1578), (456, 1378, 129, 119), (489, 962, 88, 107), (284, 524, 131, 106), (903, 382, 147, 187), (545, 452, 181, 235), (730, 1678, 108, 193), (284, 1651, 169, 146)])


    #orig_image_path_name = ['dataset_prediction/images/img_00000061.jpg']
    #orig_image_path_name = ['dataset_prediction/images2/shahida-parides-floral-v-neckline-long-kaftan-dress.jpg']
    orig_image_path_name = sorted(glob.glob('dataset_prediction/images' + '/*.jpg'))
    logging.debug('orig_image_path_name {}'.format(orig_image_path_name))
    display_bbox(orig_image_path_name, bboxes, prediction_class_name, prediction_class_prob, prediction_iou, images_name_list)


    exit(0)

    ## Image
    # for index, image in enumerate(images):
    #     logging.debug('image {}'.format(image))
    #     logging.debug('image len {}'.format((image.size)))
    #     #image.save('output/b' + str(index) + '.jpg')
    #     img = preprocess_image(image)
    #     img = np.expand_dims(img, 0)
    #     #img.save('output/b' + str(index) + '.jpg')
    #     #img.dump('output/b' + str(index) + '.txt')
    #     prediction = model.predict(img, batch_size, verbose=1)
    #     logging.debug('prediction {}'.format(prediction))

    #     images_list.append(img)
    #     images_name_list.append(image)
    #     images_name_list2.append(images_names[index])

    logging.debug('images_list {}'.format(len(images_list)))
    images_list_arr = np.array(images_list)
    logging.debug('images_list_arr type {}'.format(type(images_list_arr)))

    prediction = model.predict(images_list_arr, batch_size, verbose=1)
    #prediction = model.predict(predict_data, batch_size, verbose=1)
    # logging.debug('\n\nprediction \n{}'.format(prediction))
    logging.debug('prediction shape {} {}'.format(len(prediction), len(prediction[0])))
    print('')

    for index,preds in enumerate(prediction):
        for index2, pred in enumerate(preds):
            #print('images_name_list index2 : {:110}     '.format(images_name_list[index2]), end='')
            #print('\n')
            print('images_name_list index2 : {:110}     '.format(images_name_list2[index2]), end='')
            for p in pred:
                print('{:8f} '.format(float(p)), end='')
            print('')
        print('')


### MAIN ###

if __name__ == '__main__':

    init()
    images_path_name = get_images()
    bboxes = get_bbox(images_path_name)
    logging.debug('bboxes {}'.format(bboxes))
    #display_bbox(images_path_name, bboxes)
    image_crops, image_crops_name = crop_bbox(images_path_name, bboxes)
    logging.debug('image_crops {}'.format(len(image_crops)))
    # logging.debug('image_crops {}'.format(image_crops))
    # logging.debug('image_crops_name {}'.format(image_crops_name))

    #for index, image_crop in enumerate(image_crops):
    predict_model(image_crops, image_crops_name)


