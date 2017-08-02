#!/usr/bin/python

'''
list_category_cloth.txt
50
    category_name  category_type

7   Blazer         1
11  Jacket         1
20  Skirt          2
21  Sweatpants     2
29  Jumpsuit       3


list_category_img.txt
289222
image_name                                                             category_label
img/Sheer_Pleated-Front_Blouse/img_00000001.jpg                        3
img/Sheer_Pleated-Front_Blouse/img_00000002.jpg                        3
img/Sheer_Pleated-Front_Blouse/img_00000003.jpg                        3
img/Sheer_Pleated-Front_Blouse/img_00000004.jpg                        3
img/Sheer_Pleated-Front_Blouse/img_00000005.jpg                        3
img/Sheer_Pleated-Front_Blouse/img_00000006.jpg                        3

'''

### IMPORTS
import os
import shutil
from PIL import Image

import logging
FORMAT = "[%(lineno)4s : %(funcName)-30s ] %(message)s"
#logging.basicConfig(level=logging.DEBUG, format=FORMAT)
logging.basicConfig(level=logging.INFO, format=FORMAT)


### GLOBALS
dataset_dest_path='dataset_crop'
dataset_train_path=os.path.join(dataset_dest_path, 'train')
dataset_val_path=os.path.join(dataset_dest_path, 'validation')
dataset_test_path=os.path.join(dataset_dest_path, 'test')

dataset_src_path='fashion_data'
max_categories=50


### FUNCTIONS

if os.path.exists(dataset_dest_path):
    shutil.rmtree(dataset_dest_path)
    os.makedirs(dataset_dest_path)
    os.makedirs(dataset_train_path)
    os.makedirs(dataset_val_path)
    os.makedirs(dataset_test_path)


def get_dataset_split_name(image_path_name, file_ptr):
    for line in file_ptr:
        if image_path_name in line:
            dataset_split_name=line.split()[1]
            logging.debug('dataset_split_name {}'.format(dataset_split_name))
            return dataset_split_name.strip()

def get_bbox(image_path_name, file_ptr):
    for line in file_ptr:
        if image_path_name in line:
            x1=int(line.split()[1])
            y1=int(line.split()[2])
            x2=int(line.split()[3])
            y2=int(line.split()[4])
            bbox = [x1, y1, x2, y2]
            logging.debug('bbox {}'.format(bbox))
            return bbox



# Get category list
category_name = []
with open('fashion_data/Anno/list_category_cloth.txt') as file_list_category_cloth:
    next(file_list_category_cloth)
    next(file_list_category_cloth)
    for line in file_list_category_cloth:
        word=line.strip()[:-1].strip().replace(' ', '_')
        category_name.append(word)


# Create category dir structure
for idx,category in enumerate(category_name):
    if idx < max_categories:

        # Train
        category_path_name=os.path.join(dataset_train_path, category)
        logging.debug('category_path_name {}'.format(category_path_name))
        if not os.path.exists(os.path.join(category_path_name)):
            os.makedirs(category_path_name)

        # Validation
        category_path_name=os.path.join(dataset_val_path, category)
        logging.debug('category_path_name {}'.format(category_path_name))
        if not os.path.exists(os.path.join(category_path_name)):
            os.makedirs(category_path_name)

        # Test
        category_path_name=os.path.join(dataset_test_path, category)
        logging.debug('category_path_name {}'.format(category_path_name))
        if not os.path.exists(os.path.join(category_path_name)):
            os.makedirs(category_path_name)





# Copy all images to train dir
count=0
with open('fashion_data/Anno/list_bbox.txt') as file_list_bbox_ptr:
    with open('fashion_data/Anno/list_category_img.txt') as file_list_category_img:
        with open('fashion_data/Eval/list_eval_partition.txt', 'r') as file_list_eval_ptr:

            next(file_list_category_img)
            next(file_list_category_img)
            idx_crop=1
            for line in file_list_category_img:
                line = line.split()
                image_path_name = line[0]
                logging.debug('image_path_name {}'.format(image_path_name))                                 # img/Tailored_Woven_Blazer/img_00000051.jpg
                image_name = line[0].split('/')[-1]
                logging.debug('image_name {}'.format(image_name))                                           # image_name img_00000051.jpg
                image_full_name = line[0].replace('/', '_')
                logging.debug('image_full_name {}'.format(image_full_name))                                 # img_Tailored_Woven_Blazer_img_00000051.jpg
                image_category_index=int(line[1:][0]) - 1
                logging.debug('image_category_index {}'.format(image_category_index))                       # 2


                if image_category_index < max_categories:

                    dataset_path = ''
                    dataset_split_name = get_dataset_split_name(image_path_name, file_list_eval_ptr)

                    if dataset_split_name == "train":
                        dataset_path = os.path.join(dataset_train_path, category_name[image_category_index], image_full_name)
                    elif dataset_split_name == "val":
                        dataset_path = os.path.join(dataset_val_path, category_name[image_category_index], image_full_name)
                    elif dataset_split_name == "test":
                        dataset_path = os.path.join(dataset_test_path, category_name[image_category_index], image_full_name)
                    else:
                        logging.error('Unknown dataset_split_name {}'.format(dataset_path))
                        exit(0)

                    logging.debug('image_category_index {}'.format(image_category_index))
                    logging.debug('category_name {}'.format(category_name[image_category_index]))
                    logging.debug('dataset_path {}'.format(dataset_path))

                    # CROP
                    x1, y1, x2, y2 = get_bbox(image_path_name, file_list_bbox_ptr)                            # x1, y1, x2, y2

                    img_read = Image.open(os.path.join(dataset_src_path, 'Img', image_path_name))
                    logging.debug('{} {} {}'.format(img_read.format, img_read.size, img_read.mode))

                    #  left, upper, right, and lower pixel; The cropped section includes the left column and
                    #  the upper row of pixels and goes up to (but doesn't include) the right column and bottom row of pixels

                    img_crop = img_read.crop((x1, y1, x2, y2))                                      # origin is top left
                    #img_crop.save('temp2/test/'+ image_name + '_' + str(idx_crop) + '_cropped_' + '.jpg')
                    # img_crop.save('temp2/test/'+ image_name)

                    # Not appending cropping index for now as each image is mapped to exactly 1
                    # crop for categories
                    img_crop.save(dataset_path)
                    logging.debug('img_crop {} {} {}'.format(img_crop.format, img_crop.size, img_crop.mode))

                    idx_crop = idx_crop + 1
                    logging.debug('idx_crop {}'.format(idx_crop))

                    # if idx_crop is 1000:
                    #     exit(0)



                #shutil.copyfile(os.path.join(dataset_src_path, 'Img', image_path_name), dataset_path)

                # logging.info('count {} {}'.format(count, line))

                # file_img_category = open(images_category_path + image_full_name + '.txt', 'w')
                # file_img_category.write(category_name[image_category_encoded-1])
                # file_img_category.close()

                #logging.info('count {} {}'.format(count, line))
                count = count+1
                logging.info('count {}'.format(count))


# Move validation images to validation dir from train directory






# Display category and images count
for path in [dataset_train_path, dataset_val_path, dataset_test_path]:
    logging.info('path {}'.format(path))
    path1, dirs1, files1 = os.walk(path).next()
    file_count1 = len(files1)
    for dirs1_name in dirs1:
        path2, dirs2, files2 = os.walk(os.path.join(path, dirs1_name)).next()
        file_count2 = len(files2)
        logging.info('{:20s} : {}'.format(dirs1_name, file_count2))























