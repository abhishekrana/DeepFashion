#!/usr/bin/python

# pip install selectivesearch

### IMPORTS
from __future__ import print_function
import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
from PIL import Image

import logging
FORMAT = "[%(lineno)4s : %(funcName)-30s ] %(message)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT)



def selective_search_bbox(image):

    # load image
    img = skimage.io.imread(image)
    logging.debug('img {}'.format(type(img)))

    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)
    #img_lbl, regions = selectivesearch.selective_search(img)

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 2000:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(r['rect'])

    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)

    # The origin is at top-left corner
    logging.debug('   x    y    w    h')
    for x, y, w, h in candidates:
        logging.debug('{:4d} {:4d} {:4d} {:4d}'.format(x, y, w, h))
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.show()

    return candidates

if __name__ == "__main__":
    #image='dataset/test/Jeans/img_Distressed_Skinny_Jeans_img_00000004.jpg'
    #image='sample_images/test/img_Distressed_Denim_Jeans_img_00000001.jpg'
    image='sample_images/test/img_Acid_Wash_Denim_Romper_img_00000070.jpg'
    #image='sample_images/test/img_Acid_Wash_-_Skinny_Jeans_img_00000005.jpg'
    selective_search_bbox(image)



