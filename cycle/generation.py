import argparse
from cycle.CycleGAN import CycleGAN
from HD_BET.run import run_hd_bet
import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
from random import sample
import cv2
import torch
from torchvision.transforms.functional import normalize


class CropNifTI:
  def __init__(self):
    pass

  def procesar(self, data, margin=5, target_shape=(128, 128)):
    data = (data-np.min(data))/(np.max(data)-np.min(data))
    mask = data != 0
    row_mask = np.any(mask, axis=0)
    col_mask = np.any(mask, axis=1)
    
    rmin = np.maximum(np.argmax(row_mask) - margin,0)
    rmax = row_mask.size - np.argmax(row_mask[::-1]) + margin
    cmin = np.maximum(np.argmax(col_mask) - margin,0)
    cmax = col_mask.size - np.argmax(col_mask[::-1]) + margin

    
    # Crop the image
    cropped = data[cmin:cmax,rmin:rmax]


    # Resize to target shape
    resized_img = cv2.resize(cropped, target_shape, interpolation=cv2.INTER_LINEAR)

    return np.array(resized_img)
    

class SliceOperation():
  def __init__(self):
    pass
  
  def procesar(self,img):

    #Check Numpy
    if not isinstance(img, np.ndarray):
      img=np.array(img)

    #Squeeze
    im = np.squeeze(img)
      
    img_crop=CropNifTI().procesar(img, margin=5, target_shape=(128, 128))
    #Resize image
    if (img_crop.shape[0]!=128 and img_crop.shape[1]!=128):
      img_crop = cv2.resize(img_crop, (128,128), interpolation=cv2.INTER_LINEAR)

    #re-scale image to 0-1
    images_scaled  = (img_crop-np.min(img_crop)) / (np.max(img_crop)-np.min(img_crop))

    # Convert the numpy array to a PyTorch tensor
    img_tensor = torch.tensor(images_scaled, dtype=torch.float32)[None,:,:]


    # Normalize the tensor with mean=0.5 and std=0.5
    normalized_tensor = normalize(img_tensor, (0.5,), (0.5,))
    return normalized_tensor



class ModelApply():
  def __init__(self):
    pass

  def procesar(self,data,ModelPath):
    params = {'lr'            : 0.0005132, #0.0002 
              'lbc_T1'        : 9.377, #7
              'lbc_T2'        : 8.834,  #8
              'lbi'           : 0.08638, #0.1
              'b1'            : 0.4984, # 0.5
              'b2'            : 0.9636, #0.999
              'batch_size'    : 1,
              'im_channel'    : 1,
              'n_epochs'      : 9000,     #When it start. High number to not apply this
              'n_epochs_decay': 9000,     #Every each epoch to do High number to not apply this
              'mode'          : "linear",
              "target_shape"  : 1,
              "resnet_neck"   : 7,      #6
              "features"      : 56}

    model=CycleGAN(params).load_from_checkpoint(checkpoint_path=ModelPath)
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval() 
    model.to(self.device)  
    n_data=[]

    if isinstance(data, list):
      for im in data:
        im = im.to(self.device)
        with torch.no_grad():
          n_data.append(torch.squeeze(model.G_T1_T2(im).to("cpu")).numpy())
    else:
      data = data.to(self.device)
      with torch.no_grad():
        n_data.append(torch.squeeze(model.G_T1_T2(data).to("cpu")).numpy())
      
      n_data=n_data[0]
    

    return n_data


def Save_plot(image,path):
  
  if isinstance(image, list):
    for im in image:
      np.save(os.path.join(path,f'image_{n}.npy'), im)
    im_sample=sample(image,1)

  else:
    im_sample=image
    np.save(os.path.join(path,f'image_T2Generated.npy'), image)

  fig, ax = plt.subplots()
  ax.imshow(im_sample,cmap="gray")
  ax.set_title('Generated Image')
  ax.axis('off')
  plt.show()



    

if __name__ == '__main__':
  # Definir el argumento de línea de comandos
  parser = argparse.ArgumentParser(description='Pipeline de procesamiento')
  parser.add_argument('--input', required=True , type=str,  help='Input file path. File must be a .nii or numpy 128 x 128. If it is. nii  add option --nii')
  parser.add_argument('--model', required=True , type=str, help='Path to the model checkpoint')

  # Obtener los argumentos de la línea de comandos
  args = parser.parse_args()

  #parameter
  _, file_extension = os.path.splitext(args.input)
  output_folder=os.path.dirname(args.input)
  ModelPath=args.model


  #Check inputs and pre-processing
  if file_extension == ".npy" or file_extension == ".npz" :
    im=np.load(args.input)
    data=SliceOperation().procesar(im)
    g_data=ModelApply().procesar(data,ModelPath)
    Save_plot(g_data,output_folder)
  else:
    print("Please check the help or the code")
