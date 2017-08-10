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
from __future__ import print_function

from config import *
from selective_search import selective_search_bbox


### GLOBALS
max_categories=50

## Shorts               : 14195
# Skirt                : 10794
## Jacket               : 7548
# Top                  : 7270
# Jeans                : 5126
# Joggers              : 3260
# Hoodie               : 2910
# Sweatpants           : 2224
# Coat                 : 1539
# Sweatshorts          : 781
# Capris               : 57

#category_name_generate = ['Anorak', 'Blazer', 'Blouse', 'Bomber', 'Button-Down', 'Cardigan', 'Flannel', 'Halter', 'Henley', 'Hoodie', 'Jacket', 'Jersey', 'Parka', 'Peacoat', 'Poncho', 'Sweater', 'Tank', 'Tee', 'Top', 'Turtleneck', 'Capris', 'Chinos', 'Culottes', 'Cutoffs', 'Gauchos', 'Jeans', 'Jeggings', 'Jodhpurs', 'Joggers', 'Leggings', 'Sarong', 'Shorts', 'Skirt', 'Sweatpants', 'Sweatshorts', 'Trunks', 'Caftan', 'Cape', 'Coat', 'Coverup', 'Dress', 'Jumpsuit', 'Kaftan', 'Kimono', 'Nightdress', 'Onesie', 'Robe', 'Romper', 'Shirtdress', 'Sundress']
#category_name_generate=['Kaftan', 'Peacoat', 'Robe', 'Turtleneck']
#category_name_generate=['Kaftan', 'Peacoat', 'Robe']
#category_name_generate=['Skirt','Top','Jeans','Joggers','Hoodie','Sweatpants','Coat','Sweatshorts','Capris']

#category_name_generate=['Jeggings', 'Kaftan', 'Anorak', 'Flannel', 'Robe', 'Chinos', 'Parka', 'Jersey', 'Poncho', 'Trunks', 'Peacoat', 'Turtleneck', 'Button-Down', 'Capris', 'Bomber', 'Coat', 'Sweatshorts', 'Jeans', 'Hoodie']
#category_name_generate=['Chinos', 'Coat', 'Kaftan', 'Robe']
category_name_generate=['Coat', 'Kaftan', 'Robe']


### FUNCTIONS

# Create directory structure
def create_dataset_split_structure():
    # if os.path.exists(dataset_path):
    #     shutil.rmtree(dataset_path)

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    if not os.path.exists(dataset_train_path):
        os.makedirs(dataset_train_path)

    if not os.path.exists(dataset_val_path):
        os.makedirs(dataset_val_path)

    if not os.path.exists(dataset_test_path):
        os.makedirs(dataset_test_path)


def get_dataset_split_name(image_path_name, file_ptr):
    for line in file_ptr:
        if image_path_name in line:
            dataset_split_name=line.split()[1]
            logging.debug('dataset_split_name {}'.format(dataset_split_name))
            return dataset_split_name.strip()


def get_gt_bbox(image_path_name, file_ptr):
    for line in file_ptr:
        if image_path_name in line:
            x1=int(line.split()[1])
            y1=int(line.split()[2])
            x2=int(line.split()[3])
            y2=int(line.split()[4])
            bbox = [x1, y1, x2, y2]
            logging.debug('bbox {}'.format(bbox))
            return bbox


# Get category names list
def get_category_names():
    category_names = []
    with open(fashion_dataset_path + '/Anno/list_category_cloth.txt') as file_list_category_cloth:
        next(file_list_category_cloth)
        next(file_list_category_cloth)
        for line in file_list_category_cloth:
            word=line.strip()[:-1].strip().replace(' ', '_')
            category_names.append(word)
    return category_names


# Create category dir structure
def create_category_structure(category_names):

    for idx,category_name in enumerate(category_names):

        if category_name not in category_name_generate:
            logging.debug('Skipping category_names {}'.format(category_name))
            continue

        logging.debug('category_names {}'.format(category_name))

        if idx < max_categories:
            # Train
            category_path_name=os.path.join(dataset_train_path, category_name)
            logging.debug('category_path_name {}'.format(category_path_name))
            if not os.path.exists(os.path.join(category_path_name)):
                os.makedirs(category_path_name)

            # Validation
            category_path_name=os.path.join(dataset_val_path, category_name)
            logging.debug('category_path_name {}'.format(category_path_name))
            if not os.path.exists(os.path.join(category_path_name)):
                os.makedirs(category_path_name)

            # Test
            category_path_name=os.path.join(dataset_test_path, category_name)
            logging.debug('category_path_name {}'.format(category_path_name))
            if not os.path.exists(os.path.join(category_path_name)):
                os.makedirs(category_path_name)


# TODO: test this function
# http://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # Added due to comments on page
    if interArea < 0:
        interArea = 0

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def display_bbox(image_path_name, boxA, boxB):
    logging.debug('image_path_name {}'.format(image_path_name))

    # load image
    img = skimage.io.imread(image_path_name)
    logging.debug('img {}'.format(type(img)))

    # Draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)

    # The origin is at top-left corner
    x, y, w, h = boxA[0], boxA[1], boxA[2]-boxA[0], boxA[3]-boxA[1]
    rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='green', linewidth=1)
    ax.add_patch(rect)
    logging.debug('GT: boxA {}'.format(boxA))
    logging.debug('   x    y    w    h')
    logging.debug('{:4d} {:4d} {:4d} {:4d}'.format(x, y, w, h))

    x, y, w, h = boxB[0], boxB[1], boxB[2]-boxB[0], boxB[3]-boxB[1]
    rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1)
    ax.add_patch(rect)
    logging.debug('boxB {}'.format(boxB))
    logging.debug('   x    y    w    h')
    logging.debug('{:4d} {:4d} {:4d} {:4d}'.format(x, y, w, h))

    # plt.show()



def calculate_bbox_score_and_save_img(image_path_name, dataset_image_path, gt_x1, gt_y1, gt_x2, gt_y2):

    logging.debug('dataset_image_path {}'.format(dataset_image_path))
    logging.debug('image_path_name {}'.format(image_path_name))

    candidates = selective_search_bbox(image_path_name)
    logging.debug('candidates {}'.format(candidates))

    image_name = image_path_name.split('/')[-1].split('.')[0]
    logging.debug('image_name {}'.format(image_name))

    img_read = Image.open(image_path_name)
    logging.debug('{} {} {}'.format(img_read.format, img_read.size, img_read.mode))

    i=0
    for x, y, w, h in (candidates):
        #  left, upper, right, and lower pixel; The cropped section includes the left column and
        #  the upper row of pixels and goes up to (but doesn't include) the right column and bottom row of pixels
        logging.debug('Cropped image: i {}'.format(i))
        i=i+1

        boxA = (gt_x1, gt_y1, gt_x2, gt_y2)
        boxB = (x, y, x+w, y+h)
        iou = bb_intersection_over_union(boxA, boxB)
        logging.debug('boxA {}'.format(boxA))
        logging.debug('boxB {}'.format(boxB))
        logging.debug('iou {}'.format(iou))

        # Uncomment only for testing as too much cpu/memory wastage
        #display_bbox(image_path_name, boxA, boxB)

        #img_crop = img_read.crop((y, x, y+w, x+h))
        img_crop = img_read.crop((x, y, x+w, y+h))

        image_save_name = image_path_name.split('/')[-2] + '_' + image_path_name.split('/')[-1].split('.')[0]
        image_save_path = dataset_image_path.rsplit('/', 1)[0]
        image_save_path_name = image_save_path + '/' + image_save_name + '_crop_' +  str(x) + '-' + str(y) + '-' + str(x+w) + '-' + str(y+h) + '_iou_' +  str(iou) + '.jpg'
        logging.debug('image_save_path_name {}'.format(image_save_path_name))
        img_crop.save(image_save_path_name)
        logging.debug('img_crop {} {} {}'.format(img_crop.format, img_crop.size, img_crop.mode))

        # img_crop_resize = img_crop.resize((img_width, img_height))
        # img_crop_resize.save('temp/test/'+ image_name + '_' + str(i) + '_cropped_resize' + '.jpg')
        # logging.debug('img_crop_resize {} {} {}'.format(img_crop_resize.format, img_crop_resize.size, img_crop_resize.mode))


    # Ground Truth
    image_save_name = image_path_name.split('/')[-2] + '_' + image_path_name.split('/')[-1].split('.')[0]
    image_save_path = dataset_image_path.rsplit('/', 1)[0]
    image_save_path_name = image_save_path + '/' + image_save_name + '_gt_' +  str(gt_x1) + '-' + str(gt_y1) + '-' + str(gt_x2) + '-' + str(gt_y2) + '_iou_' +  '1.0' + '.jpg'
    logging.debug('image_save_path_name {}'.format(image_save_path_name))
    #img_crop = img_read.crop((gt_y1, gt_x1, gt_y2, gt_x2))
    img_crop = img_read.crop((gt_x1, gt_y1, gt_x2, gt_y2))
    img_crop.save(image_save_path_name)
    logging.debug('img_crop {} {} {}'.format(img_crop.format, img_crop.size, img_crop.mode))


# Generate images from fashon-data into dataset
def generate_dataset_images(category_names):


    count=0
    with open(fashion_dataset_path + '/Anno/list_bbox.txt') as file_list_bbox_ptr:
        with open(fashion_dataset_path + '/Anno/list_category_img.txt') as file_list_category_img:
            with open(fashion_dataset_path + '/Eval/list_eval_partition.txt', 'r') as file_list_eval_ptr:

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

                    if category_names[image_category_index] not in category_name_generate:
                        logging.debug('Skipping {} {}'.format(category_names[image_category_index], image_path_name))
                        continue


                    if image_category_index < max_categories:

                        dataset_image_path = ''
                        dataset_split_name = get_dataset_split_name(image_path_name, file_list_eval_ptr)

                        if dataset_split_name == "train":
                            dataset_image_path = os.path.join(dataset_train_path, category_names[image_category_index], image_full_name)
                        elif dataset_split_name == "val":
                            dataset_image_path = os.path.join(dataset_val_path, category_names[image_category_index], image_full_name)
                        elif dataset_split_name == "test":
                            dataset_image_path = os.path.join(dataset_test_path, category_names[image_category_index], image_full_name)
                        else:
                            logging.error('Unknown dataset_split_name {}'.format(dataset_image_path))
                            exit(1)

                        logging.debug('image_category_index {}'.format(image_category_index))
                        logging.debug('category_names {}'.format(category_names[image_category_index]))
                        logging.debug('dataset_image_path {}'.format(dataset_image_path))

                        # Get ground-truth bounding boxes
                        gt_x1, gt_y1, gt_x2, gt_y2 = get_gt_bbox(image_path_name, file_list_bbox_ptr)                              # Origin is top left, x1 is distance from y axis;
                                                                                                                                   # x1,y1: top left coordinate of crop; x2,y2: bottom right coordinate of crop
                        logging.debug('Ground bbox:  gt_x1:{} gt_y1:{} gt_x2:{} gt_y2:{}'.format(gt_x1, gt_y1, gt_x2, gt_y2))

                        image_path_name_src = os.path.join(fashion_dataset_path, 'Img', image_path_name)
                        logging.debug('image_path_name_src {}'.format(image_path_name_src))

                        calculate_bbox_score_and_save_img(image_path_name_src, dataset_image_path, gt_x1, gt_y1, gt_x2, gt_y2)

                        #TODO: Also cropping in test set. Check if required
                        #shutil.copyfile(os.path.join(fashion_dataset_path, 'Img', image_path_name), dataset_image_path)

                        idx_crop = idx_crop + 1
                        logging.debug('idx_crop {}'.format(idx_crop))

                        # if idx_crop is 1000:
                        #     exit(0)

                    count = count+1
                    logging.info('count {} {}'.format(count, dataset_image_path))





# Display category and images count
def display_category_data():
    for path in [dataset_train_path, dataset_val_path, dataset_test_path]:
        logging.info('path {}'.format(path))
        path1, dirs1, files1 = os.walk(path).next()
        file_count1 = len(files1)
        for dirs1_name in dirs1:
            path2, dirs2, files2 = os.walk(os.path.join(path, dirs1_name)).next()
            file_count2 = len(files2)
            logging.info('{:20s} : {}'.format(dirs1_name, file_count2))


if __name__ == '__main__':

    create_dataset_split_structure()
    category_names = get_category_names()
    logging.debug('category_names {}'.format(category_names))
    create_category_structure(category_names)
    generate_dataset_images(category_names)
    display_category_data()



