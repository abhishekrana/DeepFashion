#!/usr/bin/python

### IMPORTS
from config import *


### FUNCTIONS ###

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

    logging.debug('lr {}'.format(lr))
    logging.debug('optimizer_mod {}'.format(optimizer_mod))

    return optimizer_mod, lr


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


# Sorted subdirectories list
def get_subdir_list(path):
    names=[]
    for name in sorted(os.listdir(path)):
        if os.path.isdir(os.path.join(path, name)):
            names.append(name)
    logging.info('names {}'.format(names))
    return names


def preprocess_image(img):

    img = img.resize((img_width, img_height))

    img=np.array(img).astype(np.float32)

    # VGG16
    img[:,:,0] -= 103.939
    img[:,:,1] -= 116.779
    img[:,:,2] -= 123.68

    # img /= 255
    # img /= 255

    # img = np.expand_dims(img, 0)

    return img


def display_bbox_text(img, bbox, text):
    draw = ImageDraw.Draw(img)

    # font = ImageFont.truetype(<font-file>, <font-size>)
    # font = ImageFont.truetype("sans-serif.ttf", 16)
    #font = ImageFont.truetype("DroidSans.ttf", 16)
    font = ImageFont.truetype('fonts/alterebro-pixel-font.ttf', 20)
    #font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)

    # draw.text((x, y),"Sample Text",(r,g,b))
    draw.text((bbox[0], bbox[1]), text,(0,0,0),font=font)

    # img.save('output/sample-out.jpg')



def display_bbox(image_path_name, bboxes, prediction_class_name=None, prediction_class_prob=None, prediction_iou=None, images_name_list=None):
    logging.debug('image_path_name {}'.format(image_path_name))
    logging.debug('image_path_name {}'.format(type(image_path_name)))

    image_path_name_ = image_path_name[0]
    logging.debug('image_path_name_ {}'.format(image_path_name_))
    logging.debug('image_path_name {}'.format(type(image_path_name_)))

    # Load image
    #img = skimage.io.imread(image_path_name_)
    img = Image.open(image_path_name_)
    logging.debug('img {}'.format(type(img)))

    # Draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)


    # The origin is at top-left corner
    for index, bbox in enumerate(bboxes):

        iou_value = prediction_iou[index]
        logging.debug('iou_value {} {}'.format(iou_value, images_name_list[index]))
        if iou_value < prediction_iou_threshold:
            logging.debug('Discard')
            continue

        # x1,x2,y1,y2 (compared with selective search output plot; don't do w = bbox[2]-bbox[0])
        #x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        # Code modified; can do minus
        x, y, w, h = bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]

        rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
        logging.debug('bbox {}'.format(bbox))
        logging.debug('   x    y    w    h')
        logging.debug('{:4d} {:4d} {:4d} {:4d}'.format(x, y, w, h))

        if prediction_class_name is not None:

            pcn = prediction_class_name[index]
            pcp = prediction_class_prob[index]
            iou = prediction_iou[index]

            text='%s %s %s'%(pcn, pcp, iou)
            display_bbox_text(img, bbox, text)

        ax.imshow(img)

    plt.show()


def crop_bbox(image_path_name, bboxes):

    image_path_name_ = image_path_name[0]
    # image_path_name_ = image_path_name
    logging.debug('image_path_name_ {}'.format(image_path_name_))

    # load image
    # img = skimage.io.imread(image_path_name_)
    img = Image.open(image_path_name_)
    logging.debug('img {}'.format(type(img)))

    img_crops = []
    img_crops_name = []
    image_name = image_path_name_.split('/')[-1].split('.jpg')[0]
    logging.debug('image_name {}'.format(image_name))
    for index, bbox in enumerate(bboxes):
        x, y, w, h = bbox[0], bbox[1], bbox[2]+bbox[0], bbox[3]+bbox[1]
        logging.debug('crop {} {} {} {}'.format(x, y, w, h))
        img_crop = img.crop((x, y, w, h))
        img_crop_name = image_name + '_crop-' + str(x) + '_' + str(y) + '_' + str(w) + '_' + str(h) + '.jpg'
        img_crops_name.append(img_crop_name)
        logging.debug('img_crop_name {}'.format(img_crop_name))
        img_crop.save('dataset_prediction/crops/' + img_crop_name)
        logging.debug('img_crop {}'.format(img_crop))
        img_crops.append(img_crop)

        logging.debug('img_crop {}'.format(type(img_crop)))

    logging.debug('img_crops {}'.format(img_crops))
    return img_crops, img_crops_name


