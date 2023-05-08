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
- [Data Processing](#Data Processing)
- [Data Storage](#Data Storage)
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


## Data Processing
In the `Get_Process_Data.py` file, there are two main classes: `DownloadData` and `Process`. Here, I will provide a description of each class and its methods.

### `Class DownloadData`

This class downloads MRI datasets from OpenNeuro and saves them as NIfTI files. It has the following methods:

1. `__init__(self, save_path_nii, Datasets, target_img)`: Constructor method that initializes the class with the provided arguments:
    - `save_path_nii`: The path where the downloaded NIfTI files will be saved.
    - `Datasets`: A dictionary containing dataset IDs (e.g., "ds002330", "ds002382") as keys and their corresponding versions (e.g., "1.1.0", "1.0.1") as values.
    - `target_img`: A tuple specifying the desired size of the output images (e.g., (128, 128)).

2. `Download_OpenNeuro(self)`: Downloads the specified MRI datasets from OpenNeuro using the dataset IDs and versions provided during initialization. It saves the downloaded NIfTI files in the specified folder.

### `Process` Class

This class processes the downloaded NIfTI files into PyTorch tensors. It has the following methods:

1. `__init__(self, DataFolder, save_path)`: Constructor method that initializes the class with the provided arguments:
    - `DataFolder`: The folder containing the downloaded NIfTI files.
    - `save_path`: The path where the processed PyTorch tensor files will be saved.

2. `process_data(self)`: This method reads the NIfTI files, extracts T1-weighted and T2-weighted MRI slices, resizes them to the desired size, and saves them as PNG images. It then converts these PNG images into PyTorch tensors and saves them in the specified folder.

   The processing steps are as follows:

   a. Iterate through the NIfTI files in the `DataFolder`.
   b. Load each NIfTI file using the `nibabel` library.
   c. Extract T1-weighted and T2-weighted MRI slices from the loaded NIfTI file.
   d. Resize each extracted slice to the desired size using the `cv2.resize()` function.
   e. Save the resized slices as PNG images.
   f. Convert the PNG images into PyTorch tensors using the `torchvision.transforms.ToTensor()` function.
   g. Save the resulting tensors in the specified folder, following the directory structure mentioned in a previous response.

3. `save_tensor(self, img, path)`: This method takes an input image (in the form of a NumPy array) and a file path, converts the image to a PyTorch tensor, and saves the tensor to the specified file path.

4. `load_and_preprocess(self, file_path, target_img)`: This method takes a file path to a NIfTI file and the desired output image size, loads the NIfTI file using the `nibabel` library, extracts T1-weighted and T2-weighted MRI slices, and resizes them to the target size. It returns a list of resized slices.

By using the `DownloadData` and `Process` classes together, you can download MRI datasets from OpenNeuro, process the NIfTI files into PyTorch tensors, and save them in a structured format for training the CycleGAN model.

## Data Storage
```
├── Desire_Folder
│  ├── T1
│  │   └── groupid_subject_slice._T1W.pt
│  │   └── groupid_subject_slice._T1W.pt
│  ├── T2
│  │   └── groupid_subject_slice._T2W.pt
│  │   └── groupid_subject_slice._T12W.pt
```

## Acknowledgements
This project is based on the CycleGAN paper by Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros. The MRI datasets are from OpenNeuro.
