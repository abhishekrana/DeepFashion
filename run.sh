rm -rf output/best-weights*
rm -rf logs/*

pkill -9 tensorboard
tensorboard --log=logs &

python train_multi_v3.py

ls -1 dataset/train/
