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



# Definir las clases de la pipeline
class ApplyHDBET:
  def __init__(self):
    pass
   
    #Apply process
  def procesar(self,file_input):
    output_folder=os.path.dirname(file_input)
    output_file = os.path.join(output_folder, os.path.basename(file_input)+"_HDBET")
    run_hd_bet(file_input,output_file)
    return output_file

 

class ExtractNifTI():
  def __init__(self):
    pass

  def procesar(self,file_path ,target_img=(128,128)):
    nii_object = nib.load(file_path)
    data = nii_object.get_fdata()
    return data



class FilterNifTI():
  def __init__(self):
    pass

  def procesar(self,data,BlckProp=0.6,axis2mov=1):
    data=np.moveaxis(data, axis2mov, 0)
    siz=data.shape[1]*data.shape[2]    
    select=np.sum(data == 0, axis=(1, 2))/siz < BlckProp
    idx=[index for index, value in enumerate(select) if value]
    filt_data=data[idx,:,:]

    return filt_data


class CropNifTI():
  def __init__(self):
    pass

  def procesar(self,data,margin=5,target_shape=(128, 128)):
    crop = []
    
    #Use a mask to find black margins and compute margings per each image
    for img in data:
      mask = img != 0
      row_mask = np.any(mask, axis=1)
      col_mask = np.any(mask, axis=0)

      rmin = np.maximum(np.argmax(row_mask) - margin, 0)
      rmax = row_mask.size - np.argmax(row_mask[::-1]) + margin
      cmin = np.maximum(np.argmax(col_mask) - margin, 0)
      cmax = col_mask.size - np.argmax(col_mask[::-1]) + margin

      rmax = min(rmax, img.shape[0] - 1)
      cmax = min(cmax, img.shape[1] - 1)

      #Crop the image
      cropped = img[rmin:rmax+1, cmin:cmax+1]
      #Resize to 256x256
      resized_img = cv2.resize(cropped, target_shape, interpolation=cv2.INTER_LINEAR)
      crop.append(resized_img)
    
    if not crop:
        print("No images to stack. Please check the input data since it could be not good.")
        return np.array([])

    return np.stack(crop)
    

class SliceOperation():
  def __init__(self):
    pass
  
  def procesar(self,img):

    #Check Numpy
    if not isinstance(img, np.ndarray):
      img=np.array(img)
      
    #Resize image
    if (img.shape[0]!=128 and img.shape[1]!=128):
      resized_img = cv2.resize(img, (128,128), interpolation=cv2.INTER_LINEAR)

    #re-scale image to 0-1
    images_scaled = resized_img / np.max(resized_img, axis=(1, 2))[:, np.newaxis, np.newaxis]

    # Convert the numpy array to a PyTorch tensor
    img_tensor = torch.tensor(images_scaled, dtype=torch.float32).unsqueeze(1)

    # Normalize the tensor with mean=0.5 and std=0.5
    normalized_tensor = normalize(img_tensor, (0.5,), (0.5,))
    return normalized_tensor





class CreateModel():
  def __init__(self):
    pass

  def BuiltModel(self,ModelPath):
    #built basic class instance
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
    return model

class ModelApply():
  def __init__(self,model):
    self.model=model.eval() 
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model=model.to(self.device)  
  
  def procesar(self,data):
    n_data=[]
    for im in data:
      im=im.to(self.device)
      n_data.append(torch.squeeze(self.model.G_T2_T1(im).to("cpu")).numpy())

    return n_data

def Save_plot(image,path):
  
  #save
  for n,im in enumerate(image):
    np.save(os.path.join(path,f'image_{n}.npy'), im)

  #Plot
  if isinstance(image, list):
    im_sample=sample(image,1)
  else:
    im_sample=image

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
  parser.add_argument('--nii', required=False , default='False',type=bool, help='To apply NifIT pipeline')

  # Obtener los argumentos de la línea de comandos
  args = parser.parse_args()

  #Create Model
  model=CreateModel().BuiltModel(args.model)
  applyModel=ModelApply(model)

  #parameter
  _, file_extension = os.path.splitext(args.input)
  output_folder=os.path.dirname(args.input)


  #Check inputs and pre-processing
  if file_extension == ".nii" and args.nii==False:
    print("activate --nii or use a numpy image")
  elif file_extension == ".nii" and args.nii==True:
    pipeline = [ApplyHDBET(),ExtractNifTI(),FilterNifTI(),CropNifTI()]
    inputline=args.input
    for step in pipeline:
      inputline = step.procesar(inputline)
    
    Pro_im=SliceOperation()
    data=[Pro_im.procesar(im) for im in inputline]
    g_data=applyModel.procesar(data) #Numpy 128x128
    Save_plot(g_data,output_folder)


  elif file_extension == ".npy" or file_extension == ".npz" :
    data=SliceOperation(args.input)
    g_data=applyModel.procesar(data)[0]
    Save_plot(g_data,output_folder)


  else:
    print("Please check the help or the code")
