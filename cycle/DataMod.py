#Necesary Packages
import os
import glob
import os.path
import numpy as np
import pandas as pd
import lightning.pytorch as pl
import torch
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split
import random
import re


##############################  Class 1 ############################## 
class CustomDataset(Dataset):
  def __init__(self, T1_files, T2_files,factor=1):
    """
    @Description: 
      A custom PyTorch Dataset class to  handle not paired T1 and T2 MRI image files.
    @Input:
        T1_files (List[str]): List of paths to T1 MRI image files.
        T2_files (List[str]): List of paths to T2 MRI image files.
    """
    self.T1_files = T1_files
    self.T2_files = T2_files
    self.factor=factor

  def __len__(self):
    return int(min(len(self.T1_files), len(self.T2_files))*self.factor)

  def __getitem__(self, idx):
    """
    @Description: 
      Retrieves an item from the dataset at a specified index.
    @Input:
        idx (int): The index of the item to be retrieved.
    @Output:
        tuple: T1 image, T1 image's corresponding T2 image, T2 image, and T2 image's corresponding T1 image.
    """

    #Get Image Paths
    img_path_T1 = self.T1_files[idx]
    #img_path_T2 = self.T2_files[idx]

    #The images are not pairs! however, I want to find the pair to compare
    img_path_T1_T2Corrs = re.sub(r'(T1)(w?)', r'T2\2', img_path_T1)
    #img_path_T2_T1Corrs = re.sub(r'(T2)(w?)', r'T1\2', img_path_T2)

    #Load
    img_T1 = torch.load(img_path_T1)
    #img_T2 = torch.load(img_path_T2)

    img_T1_T2Corrs = torch.load(img_path_T1_T2Corrs)
    #img_T2_T1Corrs = torch.load(img_path_T2_T1Corrs)

    return img_T1, img_T1_T2Corrs




##############################  Class 2 ############################## 
class CycleGANDataModule(pl.LightningDataModule):
    """
      @Description: 
        Custom PyTorch Lightning DataModule for MRI image data, pairs of T1 and T2.
      @Input:
          paths (dict): A dictionary containing the paths to T1 and T2 directories.
          batch_size (int) = 1: Batch size for data loading
          num_workers (int) = 4 : Number of workers for data loading
          train_ratio (float) = 0.7: The ratio of training data
          test_size (float) 0.1: The ratio of test data
          val_size (float) = 0.2:   The ratio of validation data
          seed (int) = 478: Seed for random number generation. 
          factor (int) = 1: Factor for modifying the dataset.
    """
    
    def __init__(self, paths,
      batch_size=1,
      num_workers=4,
      train_ratio=0.7,
      test_size=0.1,
      val_size=0.2,
      seed=478,
      factor=1):


      super().__init__()
      self.T1_dir = paths["T1"]
      self.T2_dir =  paths["T2"]
      self.batch_size = batch_size
      self.num_workers = num_workers
      self.train_ratio = train_ratio
      self.test_size = test_size
      self.val_size = val_size
      self.seed=seed
      self.factor=factor
      self.save_hyperparameters()



    def prepare_data(self):
      pass

    def setup(self, stage=None):
      #Get shuffled list of dirs in T1 and T2
      T1_file = [os.path.join(self.T1_dir, f) for f in os.listdir(self.T1_dir) if f.endswith('.pt')]
      T2_file = [os.path.join(self.T2_dir, f) for f in os.listdir(self.T2_dir) if f.endswith('.pt')]

      T1_shuffled = random.sample(T1_file, len(T1_file))
      T2_shuffled = random.sample(T2_file, len(T2_file))

      #Split to validation, training and test
      train_T1, test_T1, train_T2, test_T2 = train_test_split(T1_shuffled,T2_shuffled,test_size=self.test_size ,shuffle=True,random_state=self.seed)
      train_T1, val_T1, train_T2, val_T2 = train_test_split(train_T1,train_T2,test_size=(self.val_size / (1-self.test_size)), random_state=self.seed)
      
      self.trainSize=len(train_T1)
      self.valSize=len(val_T2)
      self.testSize=len(test_T2)

      self.train_dataset = CustomDataset(train_T1, train_T2,factor=self.factor)
      self.val_dataset = CustomDataset(val_T1, val_T2,factor=self.factor)
      self.test_dataset = CustomDataset(test_T1, test_T2,factor=self.factor)

    def train_dataloader(self):
      return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self):
      return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
      return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

  
