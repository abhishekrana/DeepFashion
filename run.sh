rm -rf output/best-weights*
rm -rf logs/*

pkill -9 tensorboard
tensorboard --log=logs &

python train_multi.py

tree dataset_btl/train/

