![Baidu Logo](/doc/baidu-research-logo-small.png)

- [GPN](#ncrf)
- [Prerequisites](#prerequisites)
- [Data](#data)
- [Model](#model)
- [Training](#training)
- [Testing](#testing)


# GPN
This repository contains the code and data to reproduce the main results from the paper:

[Yi Li. Detecting Lesion Bounding Ellipses With Gaussian Proposal Networks, 2018.]()

If you find the code/data is useful, please cite the above paper. If you have any quesions, please post it on github issues or email at liyi17@baidu.com, yil8@uci.edu


# Prerequisites
* Python (3.6)

* Numpy (1.14.3)

* Scipy (1.0.1)

* [PyTorch (0.3.1)/CUDA 8.0](https://pytorch.org/previous-versions/) The specific binary wheel file is [cu80/torch-0.3.1-cp36-cp36m-linux_x86_64.whl](http://download.pytorch.org/whl/cu80/torch-0.3.1-cp36-cp36m-linux_x86_64.whl). I havn't tested on other versions, especially 0.4+, wouldn't recommend using other versions.

* torchvision (0.2.0)

* PIL (5.1.0)

* scikit-image (0.13.1)

* matplotlib (2.2.2)

* opencv (3.4.3.18)

* Cython (0.29.5)

* [tensorboardX](https://github.com/lanpa/tensorboard-pytorch) Standard along tensorboard that also works for PyTorch. This is mostly used in monitoring the training curves.

Most of the dependencies can be installed through pip install with version number, e.g. 
```
pip install 'numpy==1.14.3'
```
Or just simply
```
pip install numpy
```
A [requirements.txt](requirements.txt) file is also provided, so that you can install most of the dependencies at once:
```
pip install -r requirements.txt -i https://pypi.python.org/simple/
```
For PyTorch please consider downloading the specific wheel binary and use
```
pip install torch-0.3.1-cp36-cp36m-linux_x86_64.whl
```

# Data
The whole dataset can be downloaded from the [DeepLesion](https://nihcc.app.box.com/v/DeepLesion/) official release. The raw CT images are within `Images_png`, and the annotation is `DL_info.csv`. After downloading the whole dataset, please unzip all the `*.zip` files within `Images_png`.


# Model
![GPN](/doc/GPN.png)
The core idea of GPN is modelling the lesion bounding ellipses as 2D Gaussian distributions on the image plane, and use KL-divergence loss for bounding ellipse localization.


# Training
Train the model by the following command
```
python GPN/bin/train.py /CFG_PATH/cfg.json /SAVE_PATH/
```
where `/CFG_PATH/` is the path to the config file in json format, and `/SAVE_PATH/` is where you want to save your model in checkpoint(ckpt) format. Four config files are provided at [GPN/configs](/configs/), one is for gpn-5anchor
```json
{
 "DATAPATH": "/DEEPLESION_PATH/",
 "MAX_SIZE": 512,
 "RPN_FEAT_STRIDE": 8,
 "ANCHOR_SCALES": [2, 3, 4, 6, 12],
 "ANCHOR_RATIOS": [1],
 "NORM_SPACING": 0.8,
 "SLICE_INTV": 2,
 "HU_MIN": -1024,
 "HU_MAX": 3071,
 "PIXEL_MEANS": 50,
 "MAX_NUM_GT_BOXES": 3,
 "TRAIN.RPN_BATCHSIZE": 32,
 "TRAIN.IMS_PER_BATCH": 2,
 "TRAIN.RPN_FG_FRACTION": 0.5,
 "TRAIN.RPN_POSITIVE_OVERLAP": 0.7,
 "TRAIN.RPN_NEGATIVE_OVERLAP": 0.3,
 "TRAIN.FROC_EVERY": 10,
 "TEST.IMS_PER_BATCH": 8,
 "TEST.RPN_NMS_THRESH": 0.3,
 "TEST.RPN_PRE_NMS_TOP_N": 6000,
 "TEST.RPN_POST_NMS_TOP_N": 300,
 "TEST.RPN_MIN_SIZE": 8,
 "TEST.FROC_OVERLAP": 0.5,
 "USE_GPU_NMS": true,
 "ELLIPSE_PAD": 5,
 "ELLIPSE_LOSS": "KLD",
 "base_model": "vgg16",
 "pretrained": true,
 "log_every": 100,
 "epoch": 20, 
 "lr": 0.001,
 "lr_factor": 0.1,
 "lr_epoch": 10,
 "momentum": 0.9,
 "grad_norm": 10.0,
 "weight_decay": 0.0005
}

```
Please modify `/DEEPLESION_PATH/` to your own path of downloaded DeepLesion dataset.

By default, `train.py` use 1 GPU (GPU_0) to train model, 1 processes to load images. On one GTX 1080Ti, it took about 12 hours to finish 20 epoches. You can also use tensorboard to monitor the training process
```
tensorboard --logdir /SAVE_PATH/
```
![training_acc](/doc/training_FROC.png)
Typically, you will observe the GPN model with KL-divergence loss achieves higher training FROC than the RPN model with SmoothedL1 loss.

`train.py` will generate a `train.ckpt`, which is the most recently saved model, and a `best.ckpt`, which is the model with the best validation FROC.


# Testing
We can evaluate the average FROC score of lesion localization by
```
python GPN/bin/test.py /SAVE_PATH/
```
`/SAVE_PATH/` is where you saved your model. It will use the `best.ckpt` to compute the averge FROC score on the official test split of the DeepLesion dataset.

