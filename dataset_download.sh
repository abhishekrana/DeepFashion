#!/bin/bash

mkdir -p fashion_data/Anno
mkdir -p fashion_data/Eval
mkdir -p fashion_data/Img

wget https://www.dropbox.com/sh/ryl8efwispnjw21/AABpzYsttt7DIQmb2PckgbPXa/Anno/list_attr_cloth.txt?dl=0 -O fashion_data/Anno/list_attr_cloth.txt
wget https://www.dropbox.com/sh/ryl8efwispnjw21/AADYszB-Pv6mgwtiPEtQkHTva/Anno/list_attr_img.txt?dl=0 -O fashion_data/Anno/list_attr_img.txt
wget https://www.dropbox.com/sh/ryl8efwispnjw21/AADr1hf3nsOZEV3sOTZ1-m98a/Anno/list_bbox.txt?dl=0 -O fashion_data/Anno/list_bbox.txt
wget https://www.dropbox.com/sh/ryl8efwispnjw21/AACiFqyjpb21GyVwLNBATFQXa/Anno/list_category_cloth.txt?dl=0 -O fashion_data/Anno/list_category_cloth.txt
wget https://www.dropbox.com/sh/ryl8efwispnjw21/AAD3Mm6b2e9vkVdb35OfCA3fa/Anno/list_category_img.txt?dl=0 -O fashion_data/Anno/list_category_img.txt
wget https://www.dropbox.com/sh/ryl8efwispnjw21/AAARD4rdUT8oBQsjl4HuYAXha/Anno/list_landmarks.txt?dl=0 -O fashion_data/Anno/list_landmarks.txt

wget https://www.dropbox.com/sh/ryl8efwispnjw21/AACTJyCl9bprY90Z3frUZ-H-a/Eval/list_eval_partition.txt?dl=0 -O fashion_data/Eval/list_eval_partition.txt

wget https://www.dropbox.com/sh/ryl8efwispnjw21/AABKePZxbIrUHD0RjFLGA9q1a/README.txt?dl=0 -O fashion_data/README.txt

wget -c https://www.dropbox.com/sh/ryl8efwispnjw21/AACpZU-UKs_snxFH5Bp8RwOwa/Img/img.zip?dl=0 -O fashion_data/Img/img.zip

unzip fashion_data/Img/img.zip -d fashion_data/Img/

