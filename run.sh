#rm -rf bottleneck/train/*
#rm -rf bottleneck/validation/*
#rm -rf output/best-weights*
rm -rf logs/*

pkill -9 tensorboard
tensorboard --log=logs &


# TRAINING
python train_multi_v4.py

# VALIDATION
python train_multi_v3.py

ls -1 dataset/train/
