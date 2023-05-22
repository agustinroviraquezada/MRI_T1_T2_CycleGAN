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
  - [CycleGAN](#CycleGAN)   
  - [CycleGAN Architecture](#CycleGAN-Architecture)    
- [Usage](#usage)
  - [Main-Script](#Main-Script)
  - [Custom-Script](#Custom-Script)
- [Data Processing](#Data-Processing)
- [Data-Storage](#Data-Storage)
- [Model-Training](#Model-Training)
- [Acknowledgements](#Acknowledgements)

## Prerequisites

- Python 3.10.11+
- PyTorch 2.0.0+
- PyTorch Lightning 2.0.2+
- CUDA 10.2+ (if using GPU)
- Google Colab (optional)

```
!nvidia-smi

###########
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
| N/A   42C    P8     9W /  70W |      0MiB / 15360MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
###########
```

## Introduction

This repository contains an implementation of CycleGAN for performing image conversion between T1-weighted (T1) and T2-weighted (T2) MRI images. The model learned how to convert MRI images from one modality to another modality without using training data pairs. This code is based on the [CycleGAN paper](https://arxiv.org/abs/1703.10593) by Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros using PyTorch and PyTorch Lightning.

### CycleGAN
In the CycleGAN model, as an extension of the GANs, The generator function uses a point from an abstract space, known as latent space, as an input, and creates new convincing images from a specific domain. Concurrently, the discriminator function evaluates an image input and determines whether it is authentic (sourced from a dataset) or counterfeit (created by the generator). Both models are continuously refined through a competitive process, where the generator is improved to better deceive the discriminator, while the discriminator is enhanced to more accurately identify artificially generated images. 

The CycleGAN extends the GANS by adding one generator and discriminator more to the architecture to be able of achieve the cycle consistency.This concept implies that an image produced by the first generator can be utilized as input for the second generator, and the output from the second generator should align with the initial image, and in the opposite way.


  * The key components of CycleGAN include two generator networks G and F, and two discriminator networks $D_{Y}$ and $D_{X}$:
    1. **Generator Networks (G and F):** These networks are responsible for the actual transformation of the images. G transforms images from domain X to domain Y (G: X -> Y) and F does the inverse, transforming images from domain Y to domain X (F: Y -> X).

    2. **Discriminator Networks ($D_{X}$ and $D_{Y}$):** These networks are trained to differentiate between real and generated images. $D_{X}$ takes an image from domain X and outputs the probability that the image is real. Similarly, $D_{Y}$ takes an image from domain Y and outputs the probability that the image is real.

  * The loss in the CycleGAN is defined by the following components:

    Full Objective Function:   
    $L(G, F, D_X, D_Y) = L_{GAN}(G, D_Y, X, Y) + L_{GAN}(F, D_X, Y, X) + \lambda L_{\text{cyc}}(G, F) + \lambda_{\text{identity}} L_{\text{identity}}(G, F)$
    
    $\lambda$: Determines the weight of the cycle consistency loss in the overall objective function. If it's too high, the generated images might be too similar to the input images. If it's too low, the generated images might not preserve the content of the input images.
    
    $\lambda_{\text{identity}}$: High value for the hyperparameter, the model places greater importance on retaining fine-grained characteristics during the image transformation process. On the other hand, a smaller value would steer the model's attention away from these aspects


      1.  Adversarial Loss:   
          For generator G: $L_{GAN}(G, D_Y, X, Y) = \mathbb{E}{y\sim p_{data}(y)}[\log D_Y(y)] + \mathbb{E}{x\sim p_{data}(x)}[\log(1 - D_Y(G(x)))]$  
          For generator F: $L_{GAN}(F, D_X, Y, X) = \mathbb{E}{x\sim p_{data}(x)}[\log D_X(x)] + \mathbb{E}{y\sim p_{data}(y)}[\log(1 - D_X(F(y)))]$   
          
          $y \sim p_{data}(y)$: This represents a real image sample drawn from the data distribution of domain Y.      
          $x \sim p_{data}(x)$: This represents a real image sample drawn from the data distribution of domain X.   
          $\log D_Y(y)$: This is the log probability that the discriminator D_Y correctly recognizes a real image from Y.   
          $\log(1 - D_Y(G(x)))$: This is the log probability that the discriminator D_Y correctly classifies an image generated by G as being  fake (not from Y).   

      2.  Cycle Consistency Loss:   
          $L_{cyc}(G, F) = \mathbb{E}{x\sim p_{data}(x)}[||F(G(x)) - x|| 1] + \mathbb{E}{y\sim p_{data}(y)}[||G(F(y)) - y||_1]$   
          
          $||F(G(x)) - x||_1$: This measures the absolute difference between the original image from X and the image that is translated to Y by G and then translated back to X by F.   
          $||G(F(y)) - y||_1$: This measures the absolute difference between the original image from Y and the image that is translated to X by F and then translated back to Y by G.  

      3.  Identity Loss:   
          $L_{\text{identity}}(G, F) = \mathbb{E}{y\sim p_{data}(y)}[||G(y) - y||1] + \mathbb{E}{x\sim p_{data}(x)}[||F(x) - x||_1]$ 
         
          $y \sim p_{data}(y)$: This represents a real image sample drawn from the data distribution of domain Y.  
          $x \sim p_{data}(x)$: This represents a real image sample drawn from the data distribution of domain X.  
          $||G(y) - y||_1$ and $||F(x) - x||_1$: These measure the absolute differences between the original images and the images translated by G and F respectively.  

### CycleGAN Architecture
* Global Architecture   
  Forward Cycle Consistency Loss:
    1.  An image from domain X (T1w) is fed into Generator G (which is trained to translate from domain X to domain Y, e.g., T1w to T2w). Generator G translates this image into domain Y, producing a generated image T2w.
    2.  The generated T2w image is fed into the discriminator $D_{Y}$ to compute Adversarial loss, by comparing original T2w and generated T2w image
    3. This generated T2w image is then fed into Generator F (which is trained to translate from domain Y to domain X, e.g., from T2w to T1w).
    4. Generator F attempts to reconstruct the original X  (T1w cycle) image, as result $\hat{x}$  is produced
    5. The Forward Cycle Consistency Loss is computed as the difference between the original image from domain X and the reconstructed image, aiming to minimize this difference.

  Backward Cycle Consistency Loss:
    1.  An image from domain Y (e.g., T2w) is fed into Generator F to this image into domain X, producing a T1w image.
    2.  The generated T1w image is fed into the discriminator $D_{X}$ to compute Adversarial loss, by comparing original T1w and generated T1w image
    3.  This generated T1w image is then fed into Generator G.
    4.  Generator G attempts to reconstruct the original T2w image.
    5.  The Backward Cycle Consistency Loss is computed as the difference between the original image from domain Y and the reconstructed image, again aiming to minimize this difference.

  These cycle consistency losses are designed to ensure that if an image is transferred from one domain to another and then reverted back to the original domain, it should closely resemble the original image. 

  <p align="center">
    <img src="https://github.com/agustinroviraquezada/MRI_T1_T2_CycleGAN/blob/main/docs/CycleGANDraw.svg" alt="CycleGAN Architecture" height="600px" width="800px">
  </p>
  
* Components Architecture


  1.  Generator:    
      Modified version of the ResNet (Residual Network). For images that are larger than 128x128, they use a 9-block ResNet-based generator, otherwise, a 6-block one. Each block consists of a convolutional layer, a batch normalization layer (except in the output layer), and a ReLU activation function. In these ResNet blocks, the input goes through convolutional layers and is added to the original input, forming a sort of 'shortcut connection'. This is intended to help the model more easily learn identity mappings between the input and output, which is helpful for the type of image-to-image translation tasks that CycleGAN is designed for.

  <p align="center">
    <img src="https://github.com/agustinroviraquezada/MRI_T1_T2_CycleGAN/blob/main/docs/GeneraratorDraw.svg" height="600px" width="800px">
  </p>
  
  2.  Discriminator:   
      It is a PatchGAN classifier which tries to classify whether patches in an image are real or fake. This means that the discriminator does not try to classify the entire image as real or fake, but rather classifies each patch of the image separately. This gives a form of localised feedback to the generator about which specific parts of the image look unrealistic.
  
  <p align="center">
    <img src="https://github.com/agustinroviraquezada/MRI_T1_T2_CycleGAN/blob/main/docs/DiscriminatorDraw.svg" height="600px" width="800px">
  </p>

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

A detailed description of the classes and objects can be found in the [cycle package](https://github.com/agustinroviraquezada/MRI_T1_T2_CycleGAN/blob/main/cycle/readme.md)

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
  <img src="https://github.com/agustinroviraquezada/MRI_T1_T2_CycleGAN/blob/main/docs/PipeLine.svg" alt="Data processing pipeline" height="400px" width="700px">
</p>


## Data-Storage

After pre-processing, the slice are storeed as .pt in their respective folders, as the structure shows. In addition the package implement a function to create the tree structure where the checkpoints, the bests models, the tensorboard logs, and the image for the gift are going to be stored during the training.

The dataset folders, T1 and T2, size more than 53 GB. This dataset was made from a total of 5 datasets from OpenNeuro:

  1. Dataset ds002330 -  version 1.1.0   
     Sunavsky, A. and Poppenk, J. (2020). Neuroimaging predictors of creativity in healthy adults. OpenNeuro. [Dataset DOI](https://openneuro.org/datasets/ds002330/versions/1.1.0)
      
  2. Dataset ds002382-  version 1.0.1   
     Rogers CS and Jones MS and McConkey S and Spehar B and Van Engen KJ and Sommers MS and Peelle JE (2022). Age-related differences in auditory cortex activity during spoken word recognition. OpenNeuro. [Dataset DOI](https://openneuro.org/datasets/ds002330/versions/1.1.0)
     
  3. Dataset ds003717 -  version 1.0.1   
     Peelle JE and Spehar B and Jones MS and McConkey S and Myerson J and Hale S and Sommers MS and Tye-Murray N (2022). Visual and audiovisual speech perception associated with increased functional connectivity between sensory and motor regions. OpenNeuro. [Dataset DOI](https://openneuro.org/datasets/ds003717/versions/1.0.1)
  
  4. Dataset ds003745 -  version 1.0.0
     David V. Smith and Dominic Fareri (2021). Social Reward Processing and Decision Making in Younger and Older Adults. OpenNeuro. [Dataset DOI](https://openneuro.org/datasets/ds003745/versions/1.0.0)
  
  5. Dataset ds004285 -  version 1.0.0
    Chad S. Rogers and Michael S. Jones and Sarah McConkey and Jonathan E. Peelle (2022). Listening task. OpenNeuro. [Dataset DOI](https://openneuro.org/datasets/ds004285/versions/1.0.0)
     

Here is the file tree structure generated by the repository. Dataset files are stored like `{Group_id}_{Subject Number}_{Slice Number}_{Modality}w.pt`, where Group_id means the OpenNeuro dataset were they belong. 

```
.
└── content
    └── drive
        └── MyDrive
                ├── Data
                |    ├── T1
                |    |   ├── {Group_id}_{Subject Number}_{Slice Number}_{Modality}w.pt
                |    |   ├── ds003717_055_150_T1w.pt
                |    |   ├── ds003717_054_151_T1w.pt
                |    |   ├── ds003717_056_152_T1w.pt
                |    └── T2
                |        ├── {Group_id}_{Subject Number}_{Slice Number}_{Modality}w.pt
                |        ├── ds003717_053_184_T2w.pt
                |        ├── ds003717_054_129_T2w.pt
                |        ├── ds003717_054_130_T2w.pt
                | 
                ├── Model
                |      ├── BestModel
                |      |   ├── best_model_version_{Tensorboard_version}_{T2_SSIM}-{epoch}.ckpt
                |      |   ├── best_model_version_0_0.3-01.ckpt
                |      ├── CheckPoint
                |      |   ├── model__version_{Tensorboard_version}_{epoch}.ckpt
                |      |   ├── model__version_0_230.ckpt
                |      ├── GIF
                |      └──   cycleGAN
                |             ├── version_{Tensorboard_version}
                |                    └── events.out.tfevents.1684251252.16c6ed093068.2517.0
                └──Test
                      ├── T1
                      |     ├── {Group_id}_{Subject Number}_{Slice Number}_{Modality}w.pt
                      |     ├── ds003717_055_150_T1w.pt
                      |     ├── ds003717_054_151_T1w.pt
                      |     ├── ds003717_056_152_T1w.pt
                      └── T2
                            ├── {Group_id}_{Subject Number}_{Slice Number}_{Modality}w.pt
                            ├── ds003717_053_184_T2w.pt
                            ├── ds003717_054_129_T2w.pt
                            └── ds003717_054_130_T2w.pt
```

The total number of image of the dataset is 11600 image per modality. The following table shows the groups id from where the dataset comes, the subjects per group (since not all the subjects were valid) and the mean number of slices per subject (since not all the slice were valid)   

<div align="center">
  <p align="center">
    <table>
      <tr>
        <th></th>
        <th>ds004285</th>
        <th>ds003717</th>
        <th>ds003745</th>
        <th>ds002382</th>
        <th>ds002330</th>
      </tr>
      <tr>
        <td><b>Subjects</b></td>
        <td>76</td>
        <td>58</td>
        <td>12</td>
        <td>55</td>
        <td>59</td>
      </tr>
      <tr>
        <td><b>Avrg Image subject</b></td>
        <td>4</td>
        <td>5</td>
        <td>20</td>
        <td>46</td>
        <td>50</td>
      </tr>
    </table>
  </p>
</div>


Here is a sample from the final files, T1 and T2 that is used as input for the Model
<p align="center">
  <img src="https://github.com/agustinroviraquezada/MRI_T1_T2_CycleGAN/blob/main/docs/Sample.png" alt="T1 and T2 sample">
</p>   


Finally, we can see the density distribution from the pixel intensity of a random sample. The plot was limited from -0.95 to 1 since the black pixels shows a huge peak at -1. This was done in this way, to improve the visualization of the values.   
<p align="center">
  <img src="https://github.com/agustinroviraquezada/MRI_T1_T2_CycleGAN/blob/main/docs/DensityPlot.svg" alt="T1 and T2 Density plot">
</p>


## Model-Training

The training process used a randomly  subset of 10,000 images from the dataset. Over the course of 300 epochs, each involving 200 steps, the model was iteratively trained. Following the end of each epoch, a validation stage consisting of 150 steps was implemented to assess the performance of the model. During the training evaluation metrics were computed such as generator loss, discriminator loss, Structural Similarity Index Measure (SSIM) and the Peak Signal-to-Noise Ratio (PSNR). 

To evaluate, the performance of the network architecture and the builted model, several trials where ran and also a hyperparameter oprimization was performed. The different models are explained and analized bellow.


1.  Baseline   
    To start off, we'll establish a baseline for comparison by training the model using the hyperparameters and parameters that were    specified in the original research paper. Except for the linear decay that it was not applied.
     
    ```    
      'lr'            : 0.0002
      'lbc_T1'        : 10
      'lbc_T2'        : 10
      'lbi'           : 0.1
      'b1'            : 0.5
      'b2'            : 0.999
      'batch_size'    : 1
      'im_channel'    : 1
      'n_epochs'      : 9000   
      'n_epochs_decay': 9000    
      'mode'          : "linear"
      "target_shape"  : 1
      "resnet_neck"   : 6
      "features"      : 64
      "epochs"        : 300
      "steps/epoch"   : 200
    ```
    
    
    
The baseline was trained with 10k images. However, it counted with a data augmentation that increase the total dataset to 15k. Here is a sequence that shows the learning process of the model. 

<table>
  <tr>
    <td>
      <figure>
        <img src="https://github.com/agustinroviraquezada/MRI_T1_T2_CycleGAN/blob/main/docs/T2_10K_BaseLine_GIF.gif" height="350"  width="500" />
      </figure>
    </td>
    <td>
      <figure>
        <img src="https://github.com/agustinroviraquezada/MRI_T1_T2_CycleGAN/blob/main/docs/T1_10K_BaseLine_GIF.gif" height="350" width="500" />
      </figure>
    </td>
  </tr>
</table>   



A test set of 1k images were selected from the original dataset to test the models obtained in this run. The training metrics, bellow, shows that after 100 epochs the learning seems to reach the plateau. This seems reasanable taking into account that after 100 epochs the model was already seen all the dataset (100x200=20k images). This also match with the increase at the adversarial loss at the epoch 100.

<p align="center">
  <img src="https://github.com/agustinroviraquezada/MRI_T1_T2_CycleGAN/blob/main/docs/EvaluacionTraining_10k_Baseline.svg" alt="Trainig Metrics">
</p>

From the figure it is visible that learning model reach a steady stay, therefore to get the model with the best weights, the best 10 models were analized based on the SSIM and PSNR computed from the test set. On the following image you can see the 4 best models.

<p align="https://github.com/agustinroviraquezada/MRI_T1_T2_CycleGAN/blob/main/docs/EvaluacionMetrics_10k_Baseline.svg" alt="Trainig Metrics" height="500"  width="500">
</p>  


From the values of the models, the best model selected is  .... , with a SSIM of .... Here is a sample of the images generated by the model. 

   <p align="center">
      <img src="https://github.com/agustinroviraquezada/MRI_T1_T2_CycleGAN/blob/main/docs/EvaluationSample_10k_Baseline.svg" alt="Alt Text" height="600px" width="800px">
    </p>





2.  Hyper-parameters tunning
    
    Optuna is utilized to carry out the hyperparameter tuning for the CycleGAN model. This involves an exploration for the ideal hyperparameters that enhance the performance of our model. Optuna undertakes numerous trials, and within each, it optimizes a designated objective function based on the selected hyperparameters.

    The goal of the objective function in this scenario is to augment the SSIM T2 metric, meaning that the set of hyperparameters that yields the highest value of the objective function is considered the most suitable.

    Here are a list of the hyperparameters to be optimized:
    ```    
      'lr'            : (1e-4, 1e-3) # Learning rate
      'lbc_T1'        : (7, 12) # controls the weight for the Cycle loss component for T1 in the CycleGAN model
      'lbc_T2'        : (7, 12) # controls the weight for the Cycle loss component for T2 in the CycleGAN model
      'lbi'           : (0.05, 0.15) # controls the weight for the identity loss component in the CycleGAN model
      'b1'            : (0.2, 0.6) # It controls the exponential decay rate for the first-moment estimate of the gradients
      'b2'            : (0.9, 0.9999) # It controls the exponential decay rate for the second-moment estimate of the gradients 
      "resnet_neck"   : (5,9) # Controls the number of residual blocks in the neck of the ResNet architecture used in the CycleGAN model
      "features"      : (16,80) #number of features used in the CycleGAN model, impacts on the Generator and Discriminator
    ```

    Understanding the impact of each hyperparameter on the overall optimization process is a important step to know how to trick the optimization. Therefore, here is an [interactive representation](https://github.com/agustinroviraquezada/MRI_T1_T2_CycleGAN/blob/main/docs/plot_param_importances.html) of the relative importance of each hyperparameter in determining the objective value.

    <p align="center">
      <img src="https://github.com/agustinroviraquezada/MRI_T1_T2_CycleGAN/blob/main/docs/plot_param_importances.png" alt="Alt Text">
    </p>


     Here is an [interactive counter plot](https://github.com/agustinroviraquezada/MRI_T1_T2_CycleGAN/blob/main/docs/plot_contour_important_hyperparameters.html) to provide comprehensive view of the relationship between multiple hyperparameters and the objective value. This enabling us to understand how the objective value changes as different hyperparameter combinations are explored.

      <p align="center">
        <img src="https://github.com/agustinroviraquezada/MRI_T1_T2_CycleGAN/blob/main/docs/plot_contour_important_hyperparameters.png" alt="Alt Text" height="600px" width="800px">
      </p>


 
 
3.  Optimized Model

## Acknowledgements
This project is based on the CycleGAN paper by Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros. The MRI datasets are from OpenNeuro.
