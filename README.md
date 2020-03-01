# GOAD
This repository contains a PyTorch implementation of the method presented in ["Classification-Based Anomaly Detection for General Data"](https://openreview.net/pdf?id=H1lK_lBtvS) by Liron Bergman and Yedid Hoshen, ICLR 2020.

## Requirements
* Python 3 +
* Pytorch 1.0 +
* Tensorflow 1.8.0 +
* Keras 2.2.0 +
* sklearn 0.19.1 +

## Training
To replicate the results of the paper on the tabular-data:
```
python train_ad_tabular.py --n_rots=64 --n_epoch=25 --d_out=64 --ndf=32 --dataset=kdd 
python train_ad_tabular.py --n_rots=256 --n_epoch=25 --d_out=128 --ndf=128 --dataset=kddrev
python train_ad_tabular.py --n_rots=256 --n_epoch=1 --d_out=32 --ndf=8 --dataset=thyroid
python train_ad_tabular.py --n_rots=256 --n_epoch=1 --d_out=32 --ndf=8 --dataset=arrhythmia 
```
To replicate the results of the paper on CIFAR10:
```
python train_ad.py --m=0.1
```

## Citation
If you find this useful, please cite our paper:
```
@inproceedings{bergman2020goad,
  author    = {Liron Bergman and Yedid Hoshen},
  title     = {Classification-Based Anomaly Detection for General Data},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2020}
}
```
