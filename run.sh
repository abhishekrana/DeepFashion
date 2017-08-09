
mkdir -p dataset/train
mkdir -p dataset/validation
mkdir -p dataset/test

#rm -rf bottleneck/train/*
#rm -rf bottleneck/validation/*
#rm -rf output/best-weights*
rm -rf logs/*

pkill -9 tensorboard
tensorboard --log=logs &

mkdir -p dataset_prediction/crops/
rm -rf dataset_prediction/crops/*

# TRAINING
python train.py

# VALIDATION
python predict.py

