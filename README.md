# Detecting the interaction between microparticles and biomass in biological wastewater treatment process with Deep Learning method

This repository contains the code used for the following publication:
```bash
  To do: XXXXXXXX
```

The aim of this code is to use deep learning models to the interaction between microparticles and biomass in biological wastewater treatment process.

Acknowledgement:

This project was inspired by an open source project "MMDetection" (https://github.com/open-mmlab/mmdetection). 
Learn more about MMDetection at [documentation](https://mmdetection.readthedocs.io/en/latest/).

## Dataset

"XXX" dataset is a new labelled dataset for detecting the interaction between microparticles and biomass in biological wastewater treatment process. This dataset and further details can be found in:

```bash
  To do: XXXXXXXX
```

## Requirements:
- Windows 10 and Linux
- Python 3.8.16
- Pytorch 1.13.1

(1) Install Pytorch 1.13.1 (CUDA 11.7) (for Windows 10)

```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

(2) Install MMCV using MIM
```bash
pip install -U openmim
mim install mmengine==0.8.4
mim install mmcv==2.0.1
```

(3) Install MMDetection
```bash
mim install mmdet==3.1.0
```

(2) Install other packages

```bash
  pip install -r requirements.txt
```

## Usage

-  `main_Train_.ipynb` is the code for training the Yolov8 model for object detection.
-  `main_Evaluate.ipynb` is the code for (1) evaluating model performances on test sets (e.g., output mAP50, precision and recall), (2) predicting objects in images and videos, and (3) outputing bounding box (bbox) information (e.g., the area of each bbox).

## Model weights

The trained model weight files from the pubilication can be found in:

```bash
https://doi.org/10.5281/zenodo.12800597
```

## Citing this dataste or paper

If you find this code and dataset are useful in your research or wish to refer to the paper, please use the following BibTeX entry.

```BibTeX
XXXXX
```

## Contact

➡️ Tianlong Jia ([T.Jia@tudelft.nl](mailto:T.Jia@tudelft.nl))
