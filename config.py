#!/usr/bin/python

### IMPORTS
from __future__ import print_function

import os

import logging
FORMAT = "[%(lineno)4s : %(funcName)-30s ] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
#logging.basicConfig(level=logging.DEBUG, format=FORMAT)


### GLOBALS
dataset_path='dataset'
dataset_train_path=os.path.join(dataset_path, 'train')
dataset_val_path=os.path.join(dataset_path, 'validation')
dataset_test_path=os.path.join(dataset_path, 'test')
dataset_src_path='fashion_data'

