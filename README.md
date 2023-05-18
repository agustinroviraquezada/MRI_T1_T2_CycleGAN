# MRI T1-T2 CycleGAN
This repository contains an implementation of CycleGAN for MRI T1-T2 image translation in PyTorch and Pytorch Lightning.


[![made-with-python](https://img.shields.io/badge/Coded%20with-Python-21496b.svg?style=for-the-badge&logo=Python)](https://www.python.org/)
[![made-with-latex](https://img.shields.io/badge/Documented%20with-LaTeX-4c9843.svg?style=for-the-badge&logo=Latex)](https://www.latex-project.org/)
![Github code size](https://img.shields.io/github/languages/code-size/agustinroviraquezada/MRI_T1_T2_CycleGAN?style=for-the-badge&logo=Github)
![GitHub license](https://img.shields.io/github/license/agustinroviraquezada/MRI_T1_T2_CycleGAN?style=for-the-badge&logo=Github)
[![Github Follow](https://img.shields.io/github/followers/agustinroviraquezada?style=social&label=Follow)](https://github.com/agustinroviraquezada)


***********

**Master's Thesis. Master's in Data Science at Universitat Oberta de Catalunya.**

#### Author
* **Agustin Rovira** - [agustinroviraquezada@gmail.com](mailto:agustinroviraquezada@gmail.com)


***********

## Table of Contents

- [Prerequisites](#prerequisites)
- [Introduction](#Introduction)
  -[CycleGAN](#CycleGAN)
  -[Arquitecture](#Arquitecture) 
- [Usage](#usage)
  - [Main-Script](#Main-Script)
  - [Custom-Script](#Custom-Script)
- [Data Processing](#Data-Processing)
- [Data Storage](#Data-Storage)
- [Model-Training](#Model-Training)
- [Acknowledgements](#Acknowledgements)

## Prerequisites

- Python 3.10.11+
- PyTorch 2.0.0+
- PyTorch Lightning 2.0.2+
- CUDA 10.2+ (if using GPU)
- Google Colab (optional)

## Introduction

This repository contains an implementation of CycleGAN for performing image conversion between T1-weighted (T1) and T2-weighted (T2) MRI images. The model learned how to convert MRI images from one modality to another modality without using training data pairs. This code is based on the [CycleGAN paper](https://arxiv.org/abs/1703.10593) by Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros using PyTorch and PyTorch Lightning.

### CycleGAN
In the CycleGAN model, as an extension of the GANs, The generator function uses a point from an abstract space, known as latent space, as an input, and creates new convincing images from a specific domain. Concurrently, the discriminator function evaluates an image input and determines whether it is authentic (sourced from a dataset) or counterfeit (created by the generator). Both models are continuously refined through a competitive process, where the generator is improved to better deceive the discriminator, while the discriminator is enhanced to more accurately identify artificially generated images. 

The CycleGAN extends the GANS by adding one generator and discriminator more to the architecture to be able of achieve the cycle consistency.This concept implies that an image produced by the first generator can be utilized as input for the second generator, and the output from the second generator should align with the initial image, and in the opposite way.


  * The key components of CycleGAN include two generator networks G and F, and two discriminator networks D_Y and D_X:
    1. **Generator Networks (G and F):** These networks are responsible for the actual transformation of the images. G transforms images from domain X to domain Y (G: X -> Y) and F does the inverse, transforming images from domain Y to domain X (F: Y -> X).

    2. **Discriminator Networks (D_X and D_Y):** These networks are trained to differentiate between real and generated images. D_X takes an image from domain X and outputs the probability that the image is real. Similarly, D_Y takes an image from domain Y and outputs the probability that the image is real.

  * The loss in the CycleGAN is defined by the following components:

    Full Objective Function:   
    $L(G, F, D_X, D_Y) = L_{GAN}(G, D_Y, X, Y) + L_{GAN}(F, D_X, Y, X) + \lambda L_{\text{cyc}}(G, F) + \lambda_{\text{identity}} L_{\text{identity}}(G, F)$
    
    $\lambda$: Determines the weight of the cycle consistency loss in the overall objective function. If it's too high, the generated images might be too similar to the input images. If it's too low, the generated images might not preserve the content of the input images.
    
    $\lambda_{\text{identity}}$: High value for the hyperparameter, the model places greater importance on retaining fine-grained characteristics during the image transformation process. On the other hand, a smaller value would steer the model's attention away from these aspects


      1.  Adversarial Loss:   
          For generator G: $L_{GAN}(G, D_Y, X, Y) = \mathbb{E}{y\sim p{data}(y)}[\log D_Y(y)] + \mathbb{E}{x\sim p{data}(x)}[\log(1 - D_Y(G(x)))]$  
          For generator F: $L_{GAN}(F, D_X, Y, X) = \mathbb{E}{x\sim p{data}(x)}[\log D_X(x)] + \mathbb{E}{y\sim p{data}(y)}[\log(1 - D_X(F(y)))]$   
          
          $y \sim p_{data}(y)$: This represents a real image sample drawn from the data distribution of domain Y.      
          $x \sim p_{data}(x)$: This represents a real image sample drawn from the data distribution of domain X.   
          $\log D_Y(y)$: This is the log probability that the discriminator D_Y correctly recognizes a real image from Y.   
          $\log(1 - D_Y(G(x)))$: This is the log probability that the discriminator D_Y correctly classifies an image generated by G as being  fake (not from Y).   

      2.  Cycle Consistency Loss:   
          $L_{cyc}(G, F) = \mathbb{E}{x\sim p_{data}(x)}[||F(G(x)) - x|| 1] + \mathbb{E}{y\sim p_{data}(y)}[||G(F(y)) - y||_1]$   
          
          $||F(G(x)) - x||_1$: This measures the absolute difference between the original image from X and the image that is translated to Y by G and then translated back to X by F.   
          $||G(F(y)) - y||_1$: This measures the absolute difference between the original image from Y and the image that is translated to X by F and then translated back to Y by G.  

      3.  Identity Loss:   
          $L_{\text{identity}}(G, F) = \mathbb{E}{y\sim p{data}(y)}[||G(y) - y||1] + \mathbb{E}{x\sim p_{data}(x)}[||F(x) - x||_1]$ 
         
          $y \sim p_{data}(y)$: This represents a real image sample drawn from the data distribution of domain Y.  
          $x \sim p_{data}(x)$: This represents a real image sample drawn from the data distribution of domain X.  
          $||G(y) - y||_1$ and $||F(x) - x||_1$: These measure the absolute differences between the original images and the images translated by G and F respectively.  

### CycleGAN Architecture




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

## Data-Processing
This project contains several Python classes that are used to download, process, and transform medical imaging data (specifically NIfTI files). The classes interact with each other to form a processing pipeline that automates the extraction and transformation of the imaging data.

### Class Descriptions

  1. HDBETProcessor

  This class is responsible for applying the HD-BET (High Density - Brain Extraction Tool) to the input NIfTI files. It reads the input file, applies the HD-BET, and writes the processed file to the output folder. It can also apply a mask to the input file if a mask file is provided.

  2. TransformImage

  The TransformImage class loads a NIfTI file, extracts its data, filters and crops the images, and then applies several transformations to the images. This includes resizing the images to a target size, re-scaling the pixel values to the range 0-1, converting the images to PyTorch tensors, and normalizing the tensors. The processed images are then saved in a specified location as tensors of 128 x 128

  3. Process

  The Process class handles the interaction with the OpenNeuro API to fetch the dataset files. It sends a GraphQL query to the API to get the dataset snapshot files, downloads the files, applies the HD-BET using the HDBETProcessor class, and then processes the images using the TransformImage class.

  4. DownloadData

  The DownloadData class coordinates the whole pipeline. It defines the save paths for the processed images, creates the necessary folders, and initializes the Process class for each dataset. It also deletes the temporary folders created during the processing.

  5. Sanity_Check

  The Sanity_Check class performs several checks on the processed images to ensure their integrity. It checks if the number of images in the T1 and T2 folders are the same, and removes any extra images. It also provides a function to count the number of slices per subject.

  ### Data Fetching and Processing Pipeline

  The pipeline is initiated by creating an instance of the DownloadData class. The DownloadData class takes as input the save path for the processed images, a dictionary of datasets to download from OpenNeuro, and optionally the save paths for the transformed images. When an instance of this class is created, it starts the data fetching and processing pipeline.

  The DownloadData class first creates the necessary folders for saving the processed images. It then iterates over the provided datasets, and for each dataset, it initializes an instance of the Process class. The Process class sends a GraphQL query to the OpenNeuro API to get the dataset snapshot files. It then downloads the files and applies the HD-BET using the HDBETProcessor class. After that, it processes the images using the TransformImage class.

  The TransformImage class loads a NIfTI file, extracts its data, filters and crops the images, and then applies several transformations to the images. This includes resizing the images to a target size, re-scaling the pixel values to the range 0-1, converting the images to PyTorch tensors, and normalizing the tensors and rescale again to [-1,1]. The processed images are then saved in the specified location.

  After all the datasets have been processed, the DownloadData class removes the temporary folders created during the processing.

  Finally, to ensure the integrity of the processed images, the Sanity_Check class is used. It checks if the number of images in the T1 and T2 folders are the same, and removes any extra images. It also provides a function to count the number of slices per subject.
<p align="center">
  <img src="https://github.com/agustinroviraquezada/MRI_T1_T2_CycleGAN/blob/main/docs/PipeLine.png" alt="Data processing pipeline" height="400px" width="700px">
</p>
## Data-Storage

After pre-processing, the slice are storeed as .pt in their respective folders, as the structure shows. In addition the package implement a function to create the tree structure where the checkpoints, the bests models, the tensorboard logs, and the image for the gift are going to be stored during the training.

The dataset folders, T1 and T2, size a total of 53 GB. This dataset was made from a total of 3 datasets from OpenNeuro:

  1. Dataset ds002330 -  version 1.1.0   
     Sunavsky, A. and Poppenk, J. (2020). Neuroimaging predictors of creativity in healthy adults. OpenNeuro. [Dataset DOI](https://openneuro.org/datasets/ds002330/versions/1.1.0)
      
  2. Dataset ds002382-  version 1.0.1   
     Rogers CS and Jones MS and McConkey S and Spehar B and Van Engen KJ and Sommers MS and Peelle JE (2022). Age-related differences in auditory cortex activity during spoken word recognition. OpenNeuro. [Dataset DOI](https://openneuro.org/datasets/ds002330/versions/1.1.0)
     
  3. Dataset ds003717 -  version 1.0.1   
     Peelle JE and Spehar B and Jones MS and McConkey S and Myerson J and Hale S and Sommers MS and Tye-Murray N (2022). Visual and audiovisual speech perception associated with increased functional connectivity between sensory and motor regions. OpenNeuro. [Dataset DOI](https://openneuro.org/datasets/ds003717/versions/1.0.1)


```
.
└── content
    └── drive
        └── MyDrive
            ├── T1
            |   ├── {Group_id}_{Subject Number}_{Slice Number}_{Modality}w.pt
            |   ├── ds003717_055_150_T1w.pt
            |   ├── ds003717_054_151_T1w.pt
            |   ├── ds003717_056_152_T1w.pt
            ├── T2
            |   ├── {Group_id}_{Subject Number}_{Slice Number}_{Modality}w.pt
            |   ├── ds003717_053_184_T2w.pt
            |   ├── ds003717_054_129_T2w.pt
            |   ├── ds003717_054_130_T2w.pt
            └── Model
                ├── BestModel
                |   ├── best_model_version_{Tensorboard_version}_{T2_SSIM}-{epoch}.ckpt
                |   ├── best_model_version_0_0.3-01.ckpt
                ├── CheckPoint
                |   ├── model__version_{Tensorboard_version}_{epoch}.ckpt
                |   ├── model__version_0_230.ckpt
                ├── GIF
                └──   cycleGAN
                     ├── version_{Tensorboard_version}
                        └── events.out.tfevents.1684251252.16c6ed093068.2517.0
                  
```


## Model-Training
<p align="center">
  <img src="https://github.com/agustinroviraquezada/MRI_T1_T2_CycleGAN/blob/main/docs/T1_GIFT.gif" alt="Alt Text" height="600px" width="800px">
</p>


<p align="center">
  <img src="https://github.com/agustinroviraquezada/MRI_T1_T2_CycleGAN/blob/main/docs/T2_GIFT.gif" alt="Alt Text" height="600px" width="800px">
</p>





## Acknowledgements
This project is based on the CycleGAN paper by Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros. The MRI datasets are from OpenNeuro.
