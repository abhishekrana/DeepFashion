#!/usr/bin/python

# pip install selectivesearch

### IMPORTS
from config import *


### FUNCTIONS ###

def selective_search_bbox(image):
    logging.debug('image {}'.format(image))

    # load image
    img = skimage.io.imread(image)
    #img = Image.open(image)

    width, height, channels = img.shape
    logging.debug('img {}'.format(img.shape))
    logging.debug('img {}'.format(type(img)))
    region_pixels_threshold = (width*height)/100
    logging.debug('region_pixels_threshold {}'.format(region_pixels_threshold))

    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)
    #img_lbl, regions = selectivesearch.selective_search(img)
    # logging.debug('regions {}'.format(regions))
    logging.debug('regions {}'.format(len(regions)))

    candidates = set()
    for r in regions:
        # distorted rects
        x, y, w, h = r['rect']

        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue


        # # excluding regions smaller than 2000 pixels
        if r['size'] < region_pixels_threshold:
            logging.debug('Discarding - region_pixels_threshold - {} < {} - x:{} y:{} w:{} h:{}'.format(region_pixels_threshold, r['size'], x, y, w, h))
            continue


        # # Orig
        # if w / h > 1.2 or h / w > 1.2:
        #     continue

        if h != 0 and w / h > 6:
            logging.debug('Discarding w/h {} - x:{} y:{} w:{} h:{}'.format(w/h, x, y, w, h))
            continue

        if w != 0 and h / w > 6:
            logging.debug('Discarding h/w {} - x:{} y:{} w:{} h:{}'.format(h/w, x, y, w, h))
            continue


        candidates.add(r['rect'])


    # # Uncomment while debugging else cpu/memory wastage
    # # draw rectangles on the original image
    # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    # ax.imshow(img)

    # # The origin is at top-left corner
    # logging.debug('   x    y    w    h')
    # for x, y, w, h in candidates:
    #     logging.debug('{:4d} {:4d} {:4d} {:4d}'.format(x, y, w, h))
    #     rect = mpatches.Rectangle(
    #         (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
    #     ax.add_patch(rect)

    # # Display bbox
    # plt.show()

    logging.debug('candidates {}'.format(candidates))
    return candidates



if __name__ == "__main__":
    #image='dataset/test/Jeans/img_Distressed_Skinny_Jeans_img_00000004.jpg'
    #image='sample_images/test/img_Distressed_Denim_Jeans_img_00000001.jpg'
    #image='sample_images/test/img_Acid_Wash_Denim_Romper_img_00000070.jpg'
    #image='./dataset/train/Robe/Plush_Polka_Dot_Robe_img_00000057_crop_2-0-213-299_iou_0.683291770574.jpg'
    image='dataset_prediction/images/img_00000061.jpg'
    #image='sample_images/test/img_Acid_Wash_-_Skinny_Jeans_img_00000005.jpg'
    selective_search_bbox(image)



