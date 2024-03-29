from torch.utils.data import Dataset, DataLoader
import torch
import os
import matplotlib.pyplot as plt
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics import PeakSignalNoiseRatio as PSNR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from cycle.CycleGAN import CycleGAN
from cycle.DataMod import CycleGANDataModule,ImagePairTestSet
import random
import matplotlib.gridspec as gridspec
import imageio.v2 as imageio
from tqdm import tqdm

class ModelEval():
  def __init__(self,dataloader,device='cuda'):

    self.dataloader=dataloader
    self.device=device

  def RandomSamplePlot(self,ModelPath):
    params = {'lr'        : 0.0002,
          'lbc_T1'        : 10,
          'lbc_T2'        : 10,
          'lbi'           : 0.1,
          'b1'            : 0.5,
          'b2'            : 0.999,
          'batch_size'    : 1,
          'im_channel'    : 1,
          'n_epochs'      : 9000,     #When it start. High number to not apply this
          'n_epochs_decay': 9000,     #Every each epoch to do High number to not apply this
          'mode'          : "linear",
          "target_shape"  : 1,
          "resnet_neck"   : 6,
          "features"      : 64}
    
    model2=CycleGAN(params)
    model2load=model2.load_from_checkpoint(checkpoint_path=ModelPath)

    # Instantiate your CycleGAN class (assuming it is already trained)
    model2load.eval()  # Set the model to evaluation mode
    model2load.to(self.device)  # Assuming you're using GPU
    ct=0
    T1_sample,T2_sample,T1f_sample,T2f_sample=[],[],[],[]

    for T1,T2,T1_name,T2_name in self.dataloader:
      if random.random()>0.5:
        ct+=1
        T1 = T1.to(self.device)
        T2 = T2.to(self.device)

        T1_sample.append(torch.squeeze(T1.to("cpu")).numpy())
        T2_sample.append(torch.squeeze(T2.to("cpu")).numpy())

        # Generate a T2w image
        with torch.no_grad():  # Inference only
            T1f_sample.append(torch.squeeze(model2load.G_T2_T1(T2).to("cpu")).numpy())
            T2f_sample.append(torch.squeeze(model2load.G_T1_T2(T1).to("cpu")).numpy())
      if ct==6:
        break

    fig, ax = plt.subplots(3,4,figsize=(12, 12))
    fl,cl=[0,0,1,1,2,2],[0,2,0,2,0,2]
    for t2,t2f,f,c in zip(T2_sample,T2f_sample,fl,cl):
      ax[f,c].imshow(t2,cmap="gray")
      ax[f,c].set_title("Real")
      ax[f,c].axis('off')

      ax[f,c+1].imshow(t2f,cmap="gray")
      ax[f,c+1].set_title("False")
      ax[f,c+1].axis('off')
    plt.suptitle("Model Sample Real vs Generated T2")
    plt.savefig("Plot_SampleEval.svg", format='svg')
    plt.show()


  def GetMetrics(self,ModelPath):
    params = {'lr'        : 0.0002,
          'lbc_T1'        : 10,
          'lbc_T2'        : 10,
          'lbi'           : 0.1,
          'b1'            : 0.5,
          'b2'            : 0.999,
          'batch_size'    : 1,
          'im_channel'    : 1,
          'n_epochs'      : 9000,     #When it start. High number to not apply this
          'n_epochs_decay': 9000,     #Every each epoch to do High number to not apply this
          'mode'          : "linear",
          "target_shape"  : 1,
          "resnet_neck"   : 6,
          "features"      : 64}
    
    model2=CycleGAN(params)
    self.model2load=model2.load_from_checkpoint(checkpoint_path=ModelPath)


    # Instantiate your CycleGAN class (assuming it is already trained)
    self.model2load.eval()  # Set the model to evaluation mode
    self.model2load.to(self.device)  # Assuming you're using GPU
    Metrics=[]

    for T1,T2,T1_name,T2_name in tqdm(self.dataloader):
      T1_metics,T2_metrics=self.Compute_Metrics(T1,T2,T1_name,T2_name,ModelPath)
      Metrics.append(T1_metics)
      Metrics.append(T2_metrics)
    
    return Metrics 

  def Compute_Metrics(self,T1,T2,T1_name,T2_name,ModelPath):
    T1 = T1.to(self.device)
    T2 = T2.to(self.device)
    model=os.path.basename(ModelPath)

    # Generate a T2w image
    with torch.no_grad():  # Inference only
        f_T1 = self.model2load.G_T2_T1(T2)
        f_T2 = self.model2load.G_T1_T2(T1)

        C_T1=self.model2load.G_T2_T1(f_T2)
        C_T2=self.model2load.G_T1_T2(f_T2)

    #Define Metrics
    psnr= PSNR().to(self.device)
    ssim = SSIM().to(self.device)
    

    #SIIM
    C_T1_SSIM= ssim(C_T1, T1).to("cpu").detach().item()
    F_T1_SSIM= ssim(f_T1, T1).to("cpu").detach().item()

    C_T2_SSIM= ssim(C_T2, T2).to("cpu").detach().item()
    F_T2_SSIM= ssim(f_T2, T2).to("cpu").detach().item()

    #PNRS
    C_T1_PSNR= psnr(C_T1, T1).to("cpu").detach().item()
    F_T1_PSNR= psnr(f_T1, T1).to("cpu").detach().item()

    C_T2_PSNR= psnr(C_T2, T2).to("cpu").detach().item()
    F_T2_PSNR= psnr(f_T2, T2).to("cpu").detach().item()


    return (T1_name[0],"T1",model,C_T1_SSIM,F_T1_SSIM,C_T1_PSNR,F_T1_PSNR),(T2_name[0],"T2",model,C_T2_SSIM,F_T2_SSIM,C_T2_PSNR,F_T2_PSNR)

  def ComputePlot(self,ModelsPath):
    DataFrames=[self.GetMetrics(os.path.join(ModelsPath,i))for i in os.listdir(ModelsPath)]
    Stats=[item for sublist in DataFrames for item in sublist]
    df=pd.DataFrame(Stats,columns=["File","Modality","Model","SSIM_C","SSIM_F","PSNR_C","PSNR_F"])
    
    pltg=['SSIM_F' , 'PSNR_F' , 'SSIM_C', 'PSNR_C']
    pg_title=['SSIM Generated Image', 'PSNR Generated Image','SSIM Cycle Image' , 'PSNR Cycle Image']
    result=df.groupby(['Modality','Model'])[pltg].agg(['mean', 'std']).reset_index()


    g_T2=result[result["Modality"]=="T2"].sort_values(('SSIM_F', 'mean'), ascending=False).head(4)
    g_T1 = result[result["Modality"]=="T1"]
    g_T1 = g_T1[g_T1['Model'].isin(g_T2['Model'].tolist())]
    
    self.result=pd.concat([g_T1,g_T2])


    stat_T1=[(g_T1[i]["mean"],g_T1[i]["std"],g_T1["Model"]) for i in pltg]
    stat_T2=[(g_T2[i]["mean"],g_T2[i]["std"],g_T2["Model"]) for i in pltg]

    fig, ax = plt.subplots(2,2,figsize=(12, 12), sharex='col')
    for st1,st2,pg,a in zip(stat_T1,stat_T2,pg_title,ax.ravel()):
      self.PLotMetricEval(st1,st2,pg,a)

    plt.tight_layout()
    plt.savefig("Plot_Evaluation_Metrics.svg", format='svg', dpi=300)
    plt.show()
    


  def PLotMetricEval(self,group1,group2,pltg,ax):
    N = len(group1[2])     #Number of groups
    ind = np.arange(N)  # the y locations for the groups
    width = 0.35       # the width of the bars
    labels=[re.sub(r"best_model_version_0_([\d.]+)-([\d]+).ckpt", r"model_\2_\1", s) for s in group1[2]]

    # Data Mean - std
    group1_means = group1[0]
    group2_means = group2[0]
    group1_std = group1[1]
    group2_std = group2[1]


    ax.barh(ind - width/2, group1_means, width, xerr=group1_std,
                    color='SkyBlue', label='Group 1')
    ax.barh(ind + width/2, group2_means, width, xerr=group2_std,
                    color='IndianRed', label='Group 2')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.grid(True)
    ax.set_xlabel('Score')
    ax.set_title(f'Metric {pltg}')
    ax.set_yticks(ind)
    ax.set_yticklabels(labels)
    ax.legend(["T1","T2"])


  def TrainingMetrics(self,event):
    
    self.event = EventAccumulator(event)
    self.event.Reload()
    fig, ax = plt.subplots(2,2,figsize=(12, 12), sharex='col')

    # Plot1
    title="Discriminator loss"
    keys= ["D_loss_T1","D_loss_T2","Dval_loss_T1", "Dval_loss_T2"]
    labels=["Train Loss T1","Train Loss T2","Val Loss T1", "Val Loss T2"]
    self.TrainingMetricsPlots(keys,labels,title,ax[0,0])

    #Plot 2
    title="Generator loss"
    keys = ["G_loss","Gval_loss"]
    labels=["Train Loss","Val Loss T1"]
    self.TrainingMetricsPlots(keys,labels,title,ax[0,1])

    # Plot3
    title="Validation SSIM"
    keys= ["Gval_ssim_T2", "Gval_ssim_T1"]
    labels=["SSIM T2","SSIM T1"]
    self.TrainingMetricsPlots(keys,labels,title,ax[1,0])

    #Plot 4
    title="Generator loss components"
    keys= ["Cycle_term", "identity","Adver_term","Val_Cycle_term","Val_identity","Val_Adver_term"]
    labels=["Cycle","Identity","Adversarial","Val Cycle","Val Identity","Val Adversarial"]
    self.TrainingMetricsPlots(keys,labels,title,ax[1,1])
    plt.savefig("Plot_Training_Metrics.svg", format='svg', dpi=300)
    plt.show()
    

  def TrainingMetricsPlots(self,keys,lab,tit,ax,batches_per_epoch=200):
  
    data = [self.event.Scalars(key) for key in keys]
    window_size=5
     
    # Create a pandas dataframe for each key and plot it
    for d, k, l in zip(data,keys,lab):

      #Get Data
      df = pd.DataFrame(d)
      df.drop(columns="wall_time", inplace=True)
      df['Epoch'] = df['step'] // batches_per_epoch  # Convert steps to epochs
      df.rename(columns={"value": k}, inplace=True)
      df[k] = df[k].rolling(window_size).mean()  # Apply moving average
      
      #plot
      ax.plot(df["Epoch"],df[k])
      ax.set_xlabel("Epoch", fontsize=14)
      ax.set_ylabel("value", fontsize=14)
    ax.legend(lab)
    ax.set_title(tit)
    ax.grid(True)



class create_gif():
  """
  Create a GIF from a sequence of images and intensity distribution plots.

  @Description:
    This class creates a GIF by combining images and intensity distribution plots for each epoch of a given modality.
    The GIF displays three images (T2 Real, T2 Fake, T2 Cycle) and their corresponding intensity distribution plots.
    The images and plots are organized in a 2x3 grid layout.

  @Inputs:
    - image_path (str): The path to the folder containing the image files.
    - output_gif (str): The output filename for the GIF.
    - modality (str): The modality of the images (either "T1" or "T2").

  @Outputs:
    - output_gif (str): The path to the generated GIF file.

  """
  def __init__(self,image_path, output_gif,modality="T2"):
    """
    @Inputs:
      - image_path (str): The path to the folder containing the image files.
      - output_gif (str): The output filename for the GIF.
      - modality (str): The modality of the images (either "T1" or "T2").
    """

    self.image_path=image_path
    self.output_gif=output_gif

    # Define the image filenames and labels
    if modality=="T2":
      self.filenames = ['real_T2_epoch', 'f_T2_epoch', 'C_T2_epoch']
      self.labels = ['T2 Real', 'T2 Fake', 'T2 Cycle']
    elif modality== "T1":
      self.filenames = ['real_T1_epoch', 'f_T1_epoch', 'C_T1_epoch']
      self.labels = ['T1 Real', 'T1 Fake', 'T1 Cycle']
    else:
      print("please choose right modality T1 or T2")
    
    #Get epochs
    self.epochs = sorted(set([int(filename.split('_epoch')[1].split('.')[0]) for filename in os.listdir(self.image_path) if 'epoch' in filename]))
    ImageLoop=[self.createImage(e) for e in tqdm(self.epochs)]

    # Save the images as a GIF using imageio
    imageio.mimsave(output_gif, ImageLoop, duration=0.2)


  def GetTensor(self,epoch):
    return [torch.squeeze(torch.load(os.path.join(self.image_path,f"{f}{epoch:02d}.pt"))).cpu() for f in self.filenames]

  def createImage(self,epoch):

    #Load images
    images=self.GetTensor(epoch)
    #Create figure with grid
    gs = gridspec.GridSpec(2, 3)
    fig = plt.figure(figsize=(8, 6))
    
    ## Real_T2
    ax1 = fig.add_subplot(gs[0, 0])# Real_T2
    ax1.imshow(images[0], cmap='gray')
    ax1.set_title(f'{self.labels[0]} - epoch:{epoch:02d}')
    ax1.axis('off')

    ## Fake_T2
    ax2 = fig.add_subplot(gs[0, 1])# Fake_T2
    ax2.imshow(images[1], cmap='gray')
    ax2.set_title(f'{self.labels[1]} - epoch:{epoch:02d}')
    ax2.axis('off')

    ## Cycle_T2
    ax3 = fig.add_subplot(gs[0, 2])# Cycle_T2
    ax3.imshow(images[2], cmap='gray')
    ax3.set_title(f'{self.labels[2]} - epoch:{epoch:02d}')
    ax3.axis('off')

    # Histogram_T2
    ax4 = fig.add_subplot(gs[1, 0:])
    for im,lab,c in zip(images,self.labels,['r','b','g']):
      ax4.hist(im.flatten(), bins=100, density=True, histtype='step',facecolor=c,range=(-1,1),label=lab)
      ax4.set_xlim(-0.95,1)
      ax4.set_ylim(0,4)
      ax4.set_title('Intensity Distribution')
      ax4.set_xlabel('Intensity')
      ax4.set_ylabel('Frequency')
      ax4.legend()

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Save the figure as an image (e.g., PNG)
    plt.savefig('temp.png')
    completeImage=imageio.imread('temp.png')
    plt.clf()
    plt.close(fig)
    return completeImage
