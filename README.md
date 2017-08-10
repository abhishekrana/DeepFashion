# DEEP FASHION

### Setup Environment
```sh
# Virtual environment (optional)
sudo apt install -y virtualenv

# Tensorflow (optional)
sudo apt-get install python-pip python-dev python-virtualenv # for Python 2.7
virtualenv --system-site-packages tensorflow121_py27_gpu # for Python 2.7
source tensorflow121_py27_gpu/bin/activate
pip install --upgrade tensorflow-gpu  # for Python 2.7 and GPU

# Dependencies
sudo apt install -y python-tk
pip install -r requirements.txt 
```

### Download DeepFashion Dataset 
```sh
# http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html
./dataset_download.sh

# The directory structure after downloading and extracting dataset:
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

### Create Dataset
```sh
# For images in fashion_data, apply selective search algo to find ROI/bounding boxes. Crop and copy these ROI inside dataset
python dataset_create.py
```

### Train
```sh
python train.py
```

### Predict
```sh
python predict.py
```

### Misc
dataset	- Contains images used for training, validation and testing.
output	- Contains trained weights and bottleneck features.
logs    - Contains logs and events used by tensorboard.


### MODEL
```sh
						->	Classification Head (Categories)
InputImage	->	VGG16 + Layers	--
						->	Regression Head	(Confidnence in the Classification head prediction)

```

### Acknowledgment
- [DeepFashion Dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)



