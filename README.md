# MRI T1-T2 CycleGAN
This repository contains an implementation of CycleGAN for MRI T1-T2 image translation in PyTorch and Pytorch Lightning.


[![made-with-python](https://img.shields.io/badge/Coded%20with-Python-21496b.svg?style=for-the-badge&logo=Python)](https://www.python.org/)
[![made-with-latex](https://img.shields.io/badge/Documented%20with-LaTeX-4c9843.svg?style=for-the-badge&logo=Latex)](https://www.latex-project.org/)
![Github code size](https://img.shields.io/github/languages/code-size/agustinroviraquezada/MRI_T1_T2_CycleGAN?style=for-the-badge&logo=Github)
![GitHub license](https://img.shields.io/github/license/agustinroviraquezada/MRI_T1_T2_CycleGAN?style=for-the-badge&logo=Github)
![Github Follow](https://img.shields.io/github/followers/agustinroviraquezada?style=social&label=Follow)


***********

**Master's Thesis. Master's in Data Science at Universitat Oberta de Catalunya.**

#### Author
* **Agustin Rovira** - [agustinroviraquezada@gmail.com](mailto:agustinroviraquezada@gmail.com)


***********

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#Introduction)
- [Usage](#usage)
  - [Main-Script](#Main-Script)
  - [Custom-Script](#Custom-Script)
- [Package file Structure](#file-structure)
- [Data Storage](#data-storage)
- [Acknowledgements](#Acknowledgements)

## Prerequisites

- Python 3.10.11+
- PyTorch 2.0.0+
- PyTorch Lightning 2.0.2+
- CUDA 10.2+ (if using GPU)
- Google Colab (optional)

## Introduction

This repository contains an implementation of CycleGAN for performing image conversion between T1-weighted (T1) and T2-weighted (T2) MRI images. The model learned how to convert MRI images from one modality to another modality without using training data pairs. This code is based on the [CycleGAN paper](https://arxiv.org/abs/1703.10593) by Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros using PyTorch and PyTorch Lightning.


## Usage
Due resource limitation this model was created in google colab. Therefore, the installation and use has been designed to fits with google colab and google drive. You are provided with a Main.ipynb notebook in Google Colab or Jupyter Notebook, in case you want to reproduce the training of the model. If you want to recreate the training please be aware that some modifications have to be done before use it

### Main-Script
1.  Open the Main.ipynb notebook in Google Colab or Jupyter Notebook.
2.  Update the project_pat variable with the path to the project folder.
3.  Run the notebook cells to download the datasets, train the model, and generate results.

Take into account that the dataset from OpenNeuro  can weigh up to 40GB 


### Custom-Script
In case of create your own Main script, here I leave you hints about the usage of the package in colab. Mainly about how to clone and install the repo, and how to use the model.

```
project_pat="/PATH_PARENT/FOLDER_TO_CONTAINS_THE_REPO"
!git clone https://github.com/agustinroviraquezada/MRI_T1_T2_CycleGAN.git  $project_pat

requirements= project_pat+"/requirements.txt"
!pip install -r $requirements -q
```

## Package file Structure
```
.
├── cycle
│   ├── CallBacks.py:
│   ├── CycleGAN.py: 
│   ├── DataMod.py:
│   ├── Get_Process_Data.py:
│   └── Nets.py:
├── Main.ipynb:
└── README.md
```


1. CallBacks.py: This file contains custom callback classes for PyTorch Lightning. This includes a CustomModelCheckpoint used to save the best model during training and a CreateGIF to generate his GIF of images generated during training for visualization.
2. CycleGAN.py: This file contains the CycleGAN model implementation using PyTorch Lightning. It defines the overall structure of the model, including generators and discriminators, along with the loss function and training steps.
3. DataMod.py: This file contains the CycleGANDataModule class implementation for loading and processing the MRI dataset. It handles data loading, preprocessing, and splitting into training and validation sets, making it easy to use with PyTorch Lightning.
4. Get_Process_Data.py: This file contains functions to download and process the MRI dataset from OpenNeuro. It contains a DownloadData function that downloads the data set and a Process function that processes the NIfTI file into a PNG image.
5. Nets.py: This file contains the generator and discriminator network implementation for PyTorch. It includes a Generator class for image-to-image conversion between T1-weighted and T2-weighted MRI scans, and a Discriminator class for determining whether an image is real or generated. A ResBlock class is also defined in this file for use with generator networks.


#Acknowledgements
This project is based on the CycleGAN paper by Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros. The MRI datasets are from OpenNeuro.
