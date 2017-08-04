# DEEP FASHION v2

```sh
# Setup environment
sudo apt install -y virtualenv
sudo apt install -y python-tk

# Tensorflow (optional)
sudo apt-get install python-pip python-dev python-virtualenv # for Python 2.7
virtualenv --system-site-packages tensorflow121_py27_gpu # for Python 2.7
source tensorflow121_py27_gpu/bin/activate
pip install --upgrade tensorflow-gpu  # for Python 2.7 and GPU

pip install -r requirements.txt 

```

```sh
# Download DeepFashion Dataset (http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html)
./dataset_download.sh

# The directory structure should look like:
# fashion_data/
# ---Anno
# ------list_attr_cloth.txt
# ------list_attr_img.txt
# ------list_bbox.txt
# ------list_category_cloth.txt
# ------list_category_img.txt
# ------list_landmarks.txt
# ---Eval
# ------list_eval_partition.txt
# ---Img
# ------img

```


```sh
# Setup dataset: For every image in fashion_data, apply selective search algo to find ROI/bounding
# boxes, crop those regions and copy inside dataset
python dataset_init_categ_selective_search.py

tar -xzvf dataset_prediction.tar.gz

```

```sh
python train_multi_v3.py

```

##### Misc:
output	- all the trained weights and bottleneck features are saved here
logs	- logs required to run tensorboard
dataset	- images in this folder will be used for training, validation and testing

MODEL:
									->	Classification Head (Categories)
InputImage	->	VGG16 + Layers	--
									->	Regression Head	(Confidnence in the Classification head prediction)



