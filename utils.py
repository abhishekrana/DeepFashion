#!/usr/bin/python

### IMPORTS
import config

import os
import shutil
import numpy as np

import logging
#logging.basicConfig(level=logging.DEBUG, format=FORMAT)
logging.basicConfig(level=logging.INFO, format=FORMAT)


def get_dataset_class_image_count_sorted(split_name):

    class_name_sorted=[]
    class_image_count_sorted=[]

    classes_images_train_dict={}
    classes_images_val_dict={}
    classes_images_test_dict={}
    classes_images_train_dict, classes_images_val_dict, classes_images_test_dict = get_dataset_class_image_count()
    logging.debug('classes_images_train_dict {}'.format(type(classes_images_train_dict)))

    if split_name == 'train':
        for key in sorted(classes_images_train_dict.iterkeys()):
            class_name_sorted.append(key)
            class_image_count_sorted.append(classes_images_train_dict[key])

    elif split_name == 'validation':
        for key in sorted(classes_images_val_dict.iterkeys()):
            class_name_sorted.append(key)
            class_image_count_sorted.append(classes_images_val_dict[key])

    elif split_name == 'test':
        for key in sorted(classes_images_test_dict.iterkeys()):
            class_name_sorted.append(key)
            class_image_count_sorted.append(classes_images_test_dict[key])

    else:
        logging.error('Unknown split_name {}'.format(split_name))
        exit(0)

    return class_name_sorted, class_image_count_sorted



# Display category and images count
def get_dataset_class_image_count():
    split_name=[]
    classes_images_train_dict={}
    classes_images_val_dict={}
    classes_images_test_dict={}

    for path in [dataset_train_path, dataset_val_path, dataset_test_path]:
        split_name=path.split('/')[-1]
        logging.debug('split_name {}'.format(split_name))

        path1, dirs1, files1 = os.walk(path, topdown=True).next()
        for dirs1_name in dirs1:
            path2, dirs2, files2 = os.walk(os.path.join(path, dirs1_name), topdown=True).next()
            file_count2 = len(files2)

            class_name=path2.split('/')[-1]
            if split_name == 'train':
                classes_images_train_dict[class_name] = file_count2
            elif split_name == 'validation':
                classes_images_val_dict[class_name] = file_count2
            elif split_name == 'test':
                classes_images_test_dict[class_name] = file_count2
            else:
                logging.error('Unknown split_name {}'.format(split_name))
                exit(0)

            # logging.debug('file_count2 {}'.format(file_count2))
            logging.info('{:20s} : {}'.format(dirs1_name, file_count2))


    # logging.debug('\nclasses_images_train_dict {}'.format(classes_images_train_dict))
    # logging.debug('\nclasses_images_val_dict {}'.format(classes_images_val_dict))
    # logging.debug('\nclasses_images_test_dict {}'.format(classes_images_test_dict))

    return classes_images_train_dict, classes_images_val_dict, classes_images_test_dict



if __name__ == "__main__":
    # class_image_count = [23, 1047, 3416, 39, 46, 9, 0, 9, 1877, 73, 304, 1, 71, 243, 51, 8, 2, 106, 597, 1446, 985, 64, 106, 8, 575, 849, 17, 338, 724, 0, 11, 87, 17, 106, 21, 994, 6, 0, 2771, 1933, 0, 1784, 408, 160, 2128, 1411, 41, 29]
    #class_image_count = [2,5,1]

    class_names, class_image_count = get_dataset_class_image_count_sorted('train')
    logging.debug('class_names {}'.format(class_names))
    logging.debug('class_image_count {}'.format(class_image_count))

    total = sum(int(i) for i in class_image_count)
    logging.debug('total {}'.format(total))

    class_image_count_arr = 1 - np.array(class_image_count).astype(float)/total
    logging.debug('class_image_count_arr {}'.format(class_image_count_arr))

    np.set_printoptions(formatter={'float': '{: 0.6f}'.format})
    print(class_image_count_arr)


    # get_dataset_class_image_count()

    # class_names, image_count = get_dataset_class_image_count_sorted('train')
    # logging.debug('image_count {}'.format(image_count))
    # logging.debug('class_names {}'.format(class_names))

    # class_names, image_count = get_dataset_class_image_count_sorted('validation')
    # logging.debug('image_count {}'.format(image_count))
    # logging.debug('class_names {}'.format(class_names))

    # class_names, image_count = get_dataset_class_image_count_sorted('test')
    # logging.debug('image_count {}'.format(image_count))
    # logging.debug('class_names {}'.format(class_names))







