import lightning.pytorch as pl
import torch
from torch import nn
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics import PeakSignalNoiseRatio as PSNR
import torchvision
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
from cycle.Nets import Generator,Discriminator


class CycleGAN(pl.LightningModule):
  '''
  CycleGAN Class: 
  @Based on the paper: 
   - [1] Jun-Yan Zhu*, Taesung Park*, Phillip Isola, and Alexei A. Efros.
    "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks",
    in IEEE International Conference on Computer Vision (ICCV), 2017
   - [2] Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2020).
    Generative adversarial networks. Communications of the ACM, 63(11), 139-144.
   - [3] https://arxiv.org/abs/1703.10593
   - [4] Dar, S. U., Yurt, M., Karacan, L., Erdem, A., Erdem, E., & Cukur, T. (2019). 
    Image synthesis in multi-contrast MRI with conditional generative adversarial networks.
    IEEE transactions on medical imaging, 38(10), 2375-2388.
   - [5] https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/basic-gan.html
   - [6] https://www.assemblyai.com/blog/pytorch-lightning-for-dummies/
    

  @Inputs:
      - params (dict): A dictionary of hyperparameters for the model, including "n_epochs", "n_epochs_decay", "lr", "b1", "b2", "lbc_T1", "lbc_T2", "batch_size", "target_shape", and "lbi".
        
        "n_epochs" (int): The total number of epochs to train the model.
        "n_epochs_decay" (int): The number of epochs after which the learning rate will start decaying.
        "lr" (float): The learning rate for the optimizer.
        "b1" (float): The beta1 parameter for the Adam optimizer.
        "b2" (float): The beta2 parameter for the Adam optimizer.
        "lbc_T1" (float): The weight for the cycle consistency loss term for domain T1.
        "lbc_T2" (float): The weight for the cycle consistency loss term for domain T2.
        "batch_size" (int): The batch size used for training and inference.
        "target_shape" (tuple): The desired output shape of the generated images.
        "lbi" (float): The weight for the identity loss term.
        "features" (int): Number of features for the first layer
        "resnet_neck"(int): lenght of the restnet neck
        "im_channel"(int): number of channels of the image
                
        Example usage:
        params = 
          {
            "n_epochs": 100,
            "n_epochs_decay": 50,
            "lr": 0.0002,
            "b1": 0.5,
            "b2": 0.999,
            "lbc_T1": 10,
            "lbc_T2": 10,
            "batch_size": 1,
            "target_shape": (256, 256),
            "lbi": 5,
            "features"=64,
            "resnet_neck"=6,
            "im_channel"=1
          }
        cyclegan = CycleGAN(params)

  @Outputs:
      - A PyTorch LightningModule instance that can be used for training and inference.
  '''
  def __init__(self,
               params):
    super(CycleGAN,self).__init__()

    ############# HyperParameters #############
    self.save_hyperparameters(params)
    self.automatic_optimization = False
    self.n_epochs = params["n_epochs"] 
    self.n_epochs_decay = params["n_epochs_decay"] 
    self.lr = params["lr"]   
    self.b1 = params["b1"]
    self.b2 = params["b2"]
    self.lbc_T1 = params["lbc_T1"]   
    self.lbc_T2 = params["lbc_T2"]
    self.btch_size = params["batch_size"]
    self.target_shape = params["target_shape"]
    self.lbi=params["lbi"]
    self.log_every_n_iterations = 1

    ############# Define components Gerators and discriminators #############   
    self.G_T1_T2=Generator(params["im_channel"],out_f=params["features"],lvl_resnt=params["resnet_neck"])
    self.D_T1=Discriminator(params["im_channel"],HCh=params["features"],n=3)

    self.G_T2_T1=Generator(params["im_channel"],out_f=params["features"],lvl_resnt=params["resnet_neck"])
    self.D_T2=Discriminator(params["im_channel"],HCh=params["features"],n=3)

    ############# Inicializar los pesos #############
    self.G_T1_T2=self.G_T1_T2.apply(self.weights_init)
    self.D_T1=self.D_T1.apply(self.weights_init)
    self.G_T2_T1=self.G_T2_T1.apply(self.weights_init)
    self.D_T2=self.D_T2.apply(self.weights_init)



   ############# Define loss #############
    self.identity_loss = torch.nn.L1Loss()
    self.adv_loss = torch.nn.MSELoss() #adversarial loss function to keep track of how well the GAN is fooling the discriminator and how well the discriminator is catching the GAN
    self.cycle_loss = torch.nn.L1Loss()

    ###### Define the optimizaer ########
    self.op,self.sch=self.configure_optimizers(mode="linear")
    

  def forward(self, x):
    '''
    @Description: 
      Performs the forward pass of the CycleGAN model,  T1 --> T2.
    @Inputs:
        - x (Tensor): The input tensor to the model. Images T1

    @Outputs:
        - output (Tensor): The output tensor after passing through the generator.
    '''
    #Just define the generator T1 --> T2
    x=self.G_T1_T2(x)
    return x


  def training_step(self, batch, batch_idx):
    '''
    @Based on:
      Original implementation, which can be found in  [1].
      It computes loss accorgin original implementation of CycleGAN.

    @Description: 
      Single training step of the CycleGAN model.

    @Inputs:
        - batch (Tuple): A tuple containing the input batch, which consists of real images from domains T1 and T2.
        - batch_idx (int): The index of the current batch.

    @Outputs:
        - loss (Dict): A dictionary containing the computed losses during the training step.
          * generator loss (G_loss)
          * discriminator losses for T1 (D_loss_T1)
          * discriminator losses for T2 (D_loss_T2)
          * identity term (identity)
          * cycle term (Cycle_term)
          *  adversarial term (Adver_term).

    @Note:
      This method follows the original implementation of CycleGAN [1] and performs the following steps:
      1. Initializes necessary variables and optimizers.
      2. Updates the discriminator networks (D_T1 and D_T2) separately for domains T1 and T2.
      3. Updates the generator networks (G_T1_T2 and G_T2_T1).
      4. Logs the losses and metrics.
      5. Manually updates the learning rate schedulers.
      6. Returns the computed losses.

    @References:
      [1] https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

    '''
    ############# Initialization #############
    real_T1,real_T2,T1_name,T2_name = batch
    
    Gopt=self.op[0]
    Dopt_T1=self.op[1]
    Dopt_T2=self.op[2]

    ############# update discriminator #############
    #### Discriminator T1
    self.toggle_optimizer(Dopt_T1)
   
    f_T1 = self.G_T2_T1(real_T2)
    T1Loss_Dics = self.DiscLoss(real_T1,f_T1,disc="T1")
    
    self.manual_backward(T1Loss_Dics,retain_graph=False)
    Dopt_T1.step()
    Dopt_T1.zero_grad() # Zero out the gradient before backpropagation
    self.untoggle_optimizer(Dopt_T1)

    #### Discriminator T2
    self.toggle_optimizer(Dopt_T2)
    
    f_T2 = self.G_T1_T2(real_T1)
    T2Loss_Dics = self.DiscLoss(real_T2,f_T2,disc="T2")
    
    self.manual_backward(T2Loss_Dics,retain_graph=False)
    Dopt_T2.step()
    Dopt_T2.zero_grad() # Zero out the gradient before backpropagation
    self.untoggle_optimizer(Dopt_T2)

    ############# update Generator #############
    self.toggle_optimizer(Gopt)
    gen_loss, f_T1, f_T2,Iden_term,Cycle_term,Adv_term,_,_ = self.GenLoss(real_T1, real_T2)
    
    self.manual_backward(gen_loss) # Update gradients
    Gopt.step() # Update optimizer
    Gopt.zero_grad()
    self.untoggle_optimizer(Gopt)
    
    ########### Loggers ###########
    self.log("D_loss_T1", T1Loss_Dics, prog_bar=True, on_epoch=True,on_step=False)
    self.log("D_loss_T2", T2Loss_Dics, prog_bar=True, on_epoch=True,on_step=False)
    self.log("G_loss", gen_loss, prog_bar=True, on_epoch=True,on_step=False)

    self.logger.experiment.add_scalars("D_Losses", {"D_loss_T1": T1Loss_Dics,"D_loss_T2": T2Loss_Dics}, global_step=self.current_epoch)
    self.logger.experiment.add_scalars("G_Losses", {"G_loss": gen_loss}, global_step=self.current_epoch)
    self.logger.experiment.add_scalars("G_Descomp", {"identity": Iden_term,"Cycle_term": Cycle_term,"Adver_term": Adv_term}, global_step=self.current_epoch)

    # Update the learning rate schedulers manually
    if self.current_epoch > self.n_epochs:
      for scheduler in self.sch:
        scheduler.step()

    #Loss
    loss={'G_loss': gen_loss, 
          'D_loss_T2': T2Loss_Dics, 
          'D_loss_T1': T1Loss_Dics, 
          'identity': Iden_term,
          'Cycle_term': Cycle_term, 
          "Adver_term":Adv_term}
        
    return loss

  def validation_step(self, batch, batch_idx):
    '''
      @Description: 
      Single validation step of the CycleGAN model.

      @Inputs:
          - batch (Tuple): A tuple containing the input batch, which consists of real images from domains T1 and T2.
          - batch_idx (int): The index of the current batch.

      @Outputs:
          - loss (Dict): A dictionary containing the computed losses and metrics during the validation step, 
            * Generator loss (Gval_loss)
            * discriminator losses for T1 (Dval_loss_T1)
            * discriminator losses for T2 (Dval_loss_T2)
            * identity term (Val_identity)
            * cycle term (Val_Cycle_term)
            * adversarial term (Val_Adver_term)
            * peak signal-to-noise ratio for T2 (Gval_psnr_T2)
            * structural similarity index for T2 (Gval_ssim_T2)
            * peak signal-to-noise ratio for T1 (Gval_psnr_T1)
            * structural similarity index for T1 (Gval_ssim_T1)

      @Note:
      This method performs the following steps:
      1. Initializes necessary variables.
      2. Updates the discriminator networks (D_T1 and D_T2) separately for domains T1 and T2.
      3. Updates the generator networks (G_T1_T2 and G_T2_T1).
      4. Computes the losses and metrics.
      5. Logs the losses and metrics.
      6. Updates the image grid in the logger.
      7. Returns the computed losses and metrics as a dictionary.
    '''


    ############# Initialization #############
    real_T1,real_T2,T1_name,T2_name = batch
    
    ############# update discriminator #############
    #### Discriminator T1
    f_T1 = self.G_T2_T1(real_T2)
    T1Loss_Dics = self.DiscLoss(real_T1,f_T1,disc="T1")
  
    #### Discriminator T2
    f_T2 = self.G_T1_T2(real_T1)
    T2Loss_Dics = self.DiscLoss(real_T2,f_T2,disc="T2")

    ############# update Generator #############
    gen_loss, f_T1, f_T2,Iden_term,Cycle_term,Adv_term,C_T1,C_T2 = self.GenLoss(real_T1, real_T2)

    ############# Compute Training metrics #############
    G_psnr_T2,G_ssim_T2,G_psnr_T1,G_ssim_T1=self.ComputeMetrics(f_T2, real_T2,f_T1, real_T1)

    ########### Loggers ###########
    self.log("Dval_loss_T1", T1Loss_Dics, prog_bar=True, on_epoch=True,on_step=False)
    self.log("Dval_loss_T2", T2Loss_Dics, prog_bar=True, on_epoch=True,on_step=False)
    self.log("Gval_loss", gen_loss, prog_bar=True, on_epoch=True,on_step=False)
    self.log("Gval_psnr_T2", G_psnr_T2, prog_bar=True,on_epoch=True,on_step=False)
    self.log("Gval_ssim_T2", G_ssim_T2, prog_bar=True,on_epoch=True,on_step=False)
    self.log("Gval_psnr_T1", G_psnr_T1, prog_bar=True,on_epoch=True,on_step=False)
    self.log("Gval_ssim_T1", G_ssim_T1, prog_bar=True,on_epoch=True,on_step=False)


    self.logger.experiment.add_scalars("D_Losses", {"Dval_loss_T1": T1Loss_Dics,"Dval_loss_T2": T2Loss_Dics}, global_step=self.current_epoch)
    self.logger.experiment.add_scalars("G_Losses", {"Gval_loss": gen_loss}, global_step=self.current_epoch)


    loss= {'Gval_loss': gen_loss,
            'Dval_loss_T2': T2Loss_Dics,
            'Dval_loss_T1': T1Loss_Dics,
            'Val_identity': Iden_term,
            'Val_Cycle_term': Cycle_term,
            "Val_Adver_term":Adv_term,
            "Gval_psnr_T2": G_psnr_T2,
            "Gval_ssim_T2": G_ssim_T2,
            "Gval_psnr_T1": G_psnr_T1,
            "Gval_ssim_T1": G_ssim_T1,
            "images":{"real_T1":real_T1,"f_T1":f_T1,"C_T1":C_T1,"real_T2":real_T2,"f_T2":f_T2,"C_T2":C_T2}
          }



    #if batch_idx == self.log_every_n_iterations:
      #grid=self.CreateGrid(real_T1, f_T1, C_T1, real_T2, f_T2, C_T2)
      #self.logger.experiment.add_image(T1_name[0], grid, global_step=self.trainer.current_epoch)
    #else:
    grid=self.CreateGrid(real_T1, f_T1, C_T1, real_T2, f_T2, C_T2)
    self.logger.experiment.add_image("Image Transform", grid, global_step=self.trainer.current_epoch)    

    return loss
    
  def test_step(self, batch, batch_idx):
    '''
      @Description: 
      Single test step of the CycleGAN model.

      @Inputs:
          - batch (Tuple): A tuple containing the input batch, which consists of real images from domains T1 and T2.
          - batch_idx (int): The index of the current batch.

      @Outputs:
          - loss (Dict): A dictionary containing the computed losses and metrics during the test step, 
            * Generator loss (Gval_loss)
            * discriminator losses for T1 (Dval_loss_T1)
            * discriminator losses for T2 (Dval_loss_T2)
            * identity term (Val_identity)
            * cycle term (Val_Cycle_term)
            * adversarial term (Val_Adver_term)
            * peak signal-to-noise ratio for T2 (Gval_psnr_T2)
            * structural similarity index for T2 (Gval_ssim_T2)
            * peak signal-to-noise ratio for T1 (Gval_psnr_T1)
            * structural similarity index for T1 (Gval_ssim_T1)

      @Note:
      This method performs the following steps:
      1. Initializes necessary variables.
      2. Updates the discriminator networks (D_T1 and D_T2) separately for domains T1 and T2.
      3. Updates the generator networks (G_T1_T2 and G_T2_T1).
      4. Computes the losses and metrics.
      5. Logs the losses and metrics.
      6. Updates the image grid in the logger.
      7. Returns the computed losses and metrics as a dictionary.
    '''


    ############# Initialization #############
    real_T1,real_T2,T1_name,T2_name = batch
    
    ############# update discriminator #############
    #### Discriminator T1
    f_T1 = self.G_T2_T1(real_T2)
    T1Loss_Dics = self.DiscLoss(real_T1,f_T1,disc="T1")
  
    #### Discriminator T2
    f_T2 = self.G_T1_T2(real_T1)
    T2Loss_Dics = self.DiscLoss(real_T2,f_T2,disc="T2")

    ############# update Generator #############
    gen_loss, f_T1, f_T2,Iden_term,Cycle_term,Adv_term,C_T1,C_T2 = self.GenLoss(real_T1, real_T2)

    ############# Compute Training metrics #############
    G_psnr_T2,G_ssim_T2,G_psnr_T1,G_ssim_T1=self.ComputeMetrics(f_T2, real_T2,f_T1, real_T1)

    ########### Loggers ###########
    self.log("Dtst_loss_T1", T1Loss_Dics, prog_bar=True, on_epoch=True,on_step=False)
    self.log("Dtst_loss_T2", T2Loss_Dics, prog_bar=True, on_epoch=True,on_step=False)
    self.log("Gtst_loss", gen_loss, prog_bar=True, on_epoch=True,on_step=False)
    self.log("Gtst_psnr_T2", G_psnr_T2, prog_bar=True,on_epoch=True,on_step=False)
    self.log("Gtst_ssim_T2", G_ssim_T2, prog_bar=True,on_epoch=True,on_step=False)
    self.log("Gtst_psnr_T1", G_psnr_T1, prog_bar=True,on_epoch=True,on_step=False)
    self.log("Gtst_ssim_T1", G_ssim_T1, prog_bar=True,on_epoch=True,on_step=False)


    self.logger.experiment.add_scalars("D_Losses", {"Dtst_loss_T1": T1Loss_Dics,"Dtst_loss_T2": T2Loss_Dics}, global_step=self.current_epoch)
    self.logger.experiment.add_scalars("G_Losses", {"Gtst_loss": gen_loss}, global_step=self.current_epoch)


    loss= {'Gtst_loss': gen_loss,
            'Dtst_loss_T2': T2Loss_Dics,
            'Dtst_loss_T1': T1Loss_Dics,
            'tst_identity': Iden_term,
            'tst_Cycle_term': Cycle_term,
            "tst_Adver_term":Adv_term,
            "Gtst_psnr_T2": G_psnr_T2,
            "Gtst_ssim_T2": G_ssim_T2,
            "Gtst_psnr_T1": G_psnr_T1,
            "Gtst_ssim_T1": G_ssim_T1
          }



    if batch_idx == self.log_every_n_iterations:
      grid=self.CreateGrid(real_T1, f_T1, C_T1, real_T2, f_T2, C_T2)
      self.logger.experiment.add_image(T1_name[0]+"_tst", grid, global_step=self.trainer.current_epoch)
    else:
      grid=self.CreateGrid(real_T1, f_T1, C_T1, real_T2, f_T2, C_T2)
      self.logger.experiment.add_image("Image_Transform_tst", grid, global_step=self.trainer.current_epoch)    

    return loss
  
  def configure_optimizers(self,mode="linear"):
    '''
    @Description: 
      Configures the optimizers and learning rate schedulers for the CycleGAN model.
      Inicialize the optimizers. As it describe in original CycleGAN implementation [1] the optimizers are ADAMS. 

    @Inputs:
        - mode (str, optional): The mode for configuring the optimizers. It can be "linear" or any other mode. Default is "linear".

    @Outputs:
        - optm (List): A list of optimizers for the generator and discriminators.
        - shc (List or None): A list of learning rate schedulers if the mode is not "linear", otherwise None.

    @Note:
      * This method performs the following steps:
        1. Initializes the learning rate (lr), beta1 (b1), and beta2 (b2) values.
        2. Creates Adam optimizers for the generator (Gopt) and discriminators for domains T1 (Dopt_T1) and T2 (Dopt_T2).
        3. Initializes the optm list with the optimizers.
        4. Initializes the shc variable with None.
        5. If the mode is "linear":
            - Retrieves the number of epochs (n_epochs) and epochs decay (n_epochs_decay) from the hyperparameters.
            - Creates LambdaLR learning rate schedulers for the generator (sched_Gopt), Dopt_T1 (sched_Dopt_T1), and Dopt_T2 (sched_Dopt_T2), using the linear_decay method.
            - Initializes the shc list with the learning rate schedulers.
        6. If the mode is not "linear":
            - Creates ReduceLROnPlateau learning rate schedulers for the generator (sched_Gopt), Dopt_T1 (sched_Dopt_T1), and Dopt_T2 (sched_Dopt_T2), with the specified parameters.
            - Initializes the shc list with dictionaries containing the scheduler, monitor, interval, and frequency.
        7. Returns the optm list and shc (if not None) as the outputs.

      * The linear_decay method is used to calculate the learning rate decay based on the current epoch, total epochs, and epochs decay.
        @Description:

    @Reference:
      [1] https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    '''

    lr = self.lr
    b1 = self.b1
    b2 = self.b2

    Dopt_T1= torch.optim.Adam(self.D_T1.parameters(), lr=lr, betas=(b1, b2))
    Dopt_T2 = torch.optim.Adam(self.D_T2.parameters(), lr=lr, betas=(b1, b2))
    Gopt= torch.optim.Adam(list(self.G_T1_T2.parameters()) + list(self.G_T2_T1.parameters()), lr=lr, betas=(b1, b2))

    optm=[Gopt,Dopt_T1,Dopt_T2]
    shc=None

    if mode=="linear":
      n_epochs = self.hparams.n_epochs
      n_epochs_decay = self.hparams.n_epochs_decay

      sched_Dopt_T1 = LambdaLR(Dopt_T1, lr_lambda=lambda epoch: self.linear_decay(epoch, n_epochs, n_epochs_decay))
      sched_Dopt_T2 = LambdaLR(Dopt_T2, lr_lambda=lambda epoch: self.linear_decay(epoch, n_epochs, n_epochs_decay))
      sched_Gopt = LambdaLR(Gopt, lr_lambda=lambda epoch: self.linear_decay(epoch, n_epochs, n_epochs_decay))

      shc= [sched_Gopt,sched_Dopt_T1,sched_Dopt_T2]
   
    else:

      sched_Dopt_T1= ReduceLROnPlateau(Dopt_T1, mode='min', factor=0.2, threshold=0.01, patience=5, verbose=True)
      sched_Dopt_T2 = ReduceLROnPlateau(Dopt_T2, mode='min', factor=0.2, threshold=0.01, patience=5, verbose=True)
      sched_Gopt = ReduceLROnPlateau(Gopt, mode='min', factor=0.2, threshold=0.01, patience=5, verbose=True)

      shc = [{"scheduler": sched_Gopt, "monitor": "G_loss", "interval": "epoch", "frequency": 3},
                    {"scheduler": sched_Dopt_T1, "monitor": "G_loss", "interval": "epoch", "frequency": 3},
                    {"scheduler": sched_Dopt_T2, "monitor": "D_loss", "interval": "epoch", "frequency": 3}]

    return optm,shc
  
  def DiscLoss(self,real,fake,disc="T1"):
    '''
    @Description
    This function computes the discriminator loss using the adversarial loss funtion
    MSE. Taking the target label and the discriminator predictions returns the adversarial loss.
    With adverarial loss from real and from fake image we compute the discriminator loss such as:

    discriminator loss= (adv_fake+adv_real)/2

    @Inputs
    real. Tensor, real image.
    fake. Tensor, fake image.
    
    @Outputs
    Discriminator loss



    @Description: Computes the discriminator loss for the CycleGAN model, using the adversarial loss funtion
    MSE. Taking the target label and the discriminator predictions returns the adversarial loss.
    Usinf adverarial loss from real and from fake image we compute the discriminator loss such as: (adv_fake+adv_real)/2

    @Inputs:
        - real (Tensor): Real image tensor.
        - fake (Tensor): Fake image tensor.
        - disc (str, optional): The discriminator type. It can be "T1" or any other value. Default is "T1".

    @Outputs:
        - loss (Tensor): The computed discriminator loss.

    @Note:
      This method performs the following steps:
      1. If the disc parameter is "T1":
          - Computes the discriminator's prediction for the fake image (disc_fake_hat) using the D_T1 discriminator.
          - Computes the discriminator's prediction for the real image (disc_real_hat) using the D_T1 discriminator.
      2. If the disc parameter is not "T1":
          - Computes the discriminator's prediction for the fake image (disc_fake_hat) using the D_T2 discriminator.
          - Computes the discriminator's prediction for the real image (disc_real_hat) using the D_T2 discriminator.
      3. Calculates the adversarial loss for the fake image (fake_loss) using the adv_loss function and target label of zeros.
      4. Calculates the adversarial loss for the real image (real_loss) using the adv_loss function and target label of ones.
      5. Computes the discriminator loss (loss) as the average of fake_loss and real_loss.
      6. Returns the loss as the output.
    '''

    if disc == "T1":
      disc_fake_hat = self.D_T1(fake.detach())      
      disc_real_hat = self.D_T1(real)
    else:
      disc_fake_hat = self.D_T2(fake.detach())
      disc_real_hat = self.D_T2(real)

    fake_loss = self.adv_loss(disc_fake_hat, torch.zeros_like(disc_fake_hat))
    real_loss = self.adv_loss(disc_real_hat, torch.ones_like(disc_real_hat))

    r=(fake_loss + real_loss) / 2
    return r

  def GenLoss(self, real_T1, real_T2):
    '''
    @Description: 
        Computes the generator loss for the CycleGAN model.

    @Inputs:
        - real_T1 (Tensor): Real image tensor from domain T1.
        - real_T2 (Tensor): Real image tensor from domain T2.

    @Outputs:
        - gen_loss (Tensor): The computed generator loss.
        - f_T1 (Tensor): Generated fake image tensor in domain T1.
        - f_T2 (Tensor): Generated fake image tensor in domain T2.
        - Iden_term (Tensor): Identity term loss.
        - Cycle_term (Tensor): Cycle consistency term loss.
        - Adv_term (Tensor): Adversarial term loss.
        - C_T1 (Tensor): Reconstructed image tensor in domain T1.
        - C_T2 (Tensor): Reconstructed image tensor in domain T2.

    @Note:
      This method performs the following steps:
      1. Computes the generated fake image in domain T1 (f_T1) using the G_T2_T1 generator.
      2. Computes the generated fake image in domain T2 (f_T2) using the G_T1_T2 generator.
      3. Computes the discriminator's prediction for f_T1 (dic_f_T1_hat) using the D_T1 discriminator.
      4. Computes the discriminator's prediction for f_T2 (dic_f_T2_hat) using the D_T2 discriminator.
      5. Computes the adversarial loss (Adv_term) as the sum of the adversarial losses for f_T1 and f_T2.
      6. Reconstructs the image in domain T1 (C_T1) by passing f_T2 through the G_T2_T1 generator.
      7. Reconstructs the image in domain T2 (C_T2) by passing f_T1 through the G_T1_T2 generator.
      8. Computes the cycle consistency loss (Cycle_term) as the weighted sum of the L1 losses between C_T1 and real_T1, and between C_T2 and real_T2.
      9. Computes the identity term loss (Iden_term) as the sum of the L1 losses between identity_T1 and real_T1, and between identity_T2 and real_T2.
      10. Computes the generator loss (gen_loss) as the weighted sum of Iden_term, Cycle_term, and Adv_term.
      11. Returns the gen_loss, f_T1, f_T2, Iden_term, Cycle_term, Adv_term, C_T1, and C_T2 as the outputs.
    '''

    #compute fakes
    f_T1 = self.G_T2_T1(real_T2)
    f_T2 = self.G_T1_T2(real_T1)

    #Compute Discriminators output
    dic_f_T1_hat = self.D_T1(f_T1)
    dic_f_T2_hat = self.D_T2(f_T2)

    # Compute adversarial loss AdvLoss_T2_T1 +  AdvLoss_T1_T2
    Adv_term=self.adv_loss(dic_f_T1_hat, torch.ones_like(dic_f_T1_hat)) + self.adv_loss(dic_f_T2_hat, torch.ones_like(dic_f_T2_hat))

    # Compute Cycles
    C_T1 = self.G_T2_T1(f_T2)
    C_T2 = self.G_T1_T2(f_T1)

    # Compute Cycle consistancy. 
    Cycle_term=self.lbc_T1*self.cycle_loss(C_T1,real_T1)+self.lbc_T2*self.cycle_loss(C_T2,real_T2)
        
    #Compute Identities
    identity_T1 = self.G_T2_T1(real_T1)
    identity_T2 = self.G_T1_T2(real_T2)

    # Compute Identity term
    Iden_term =  self.identity_loss (identity_T1, real_T1) + self.identity_loss (identity_T2, real_T2)

    # Compute Total loss
    gen_loss = self.lbi * Iden_term +  Cycle_term + Adv_term

    return gen_loss, f_T1, f_T2,Iden_term,Cycle_term,Adv_term,C_T1,C_T2



  def weights_init(self,m):
    '''
    @Description: 
      Initializes the weights of the convolutional and batch normalization layers.
      Used to initialize weights for the Generator and Discriminator models in the CycleGAN.

    @Inputs:
        - m (nn.Module): The net whose weights need to be initialized.

    @Note:
    This method performs the following steps:
    1. Checks if the module (m) is an instance of nn.Conv2d or nn.ConvTranspose2d, indicating a convolutional layer.
      - If it is, it initializes the weights of the module using a normal distribution with mean 0.0 and standard deviation 0.02.
    2. Checks if the module (m) is an instance of nn.BatchNorm2d, indicating a batch normalization layer.
      - If it is, it initializes the weights of the module using a normal distribution with mean 0.0 and standard deviation 0.02.
      - It also sets the bias term of the module to a constant value of 0. 
    '''
    
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)



  def ComputeMetrics(self,f_T2, real_T2,f_T1, real_T1):
    """
    @Description: Computes evaluation metrics for the generated images.

    @Inputs:
        real_T1 and real_T2 are not pairs.
        - f_T2 (Tensor): Generated images from domain T1 to T2.
        - real_T2 (Tensor): Real images from domain T2.
        - f_T1 (Tensor): Generated images from domain T2 to T1.
        - real_T1 (Tensor): Real images from domain T1.

    @Outputs:
        - G_psnr_T2 (float): Peak Signal-to-Noise Ratio (PSNR) for generated T2 images.
        - G_ssim_T2 (float): Structural Similarity Index Measure (SSIM) for generated T2 images.
        - G_psnr_T1 (float): PSNR for generated T1 images.
        - G_ssim_T1 (float): SSIM for generated T1 images.
    """

    ############# Metrics Generator #############
    psnr_metric = PSNR().to(self.device)
    ssim_metric = SSIM().to(self.device)

    G_psnr_T2 = psnr_metric(f_T2, real_T2)
    G_ssim_T2 = ssim_metric(f_T2, real_T2)

    G_psnr_T1 = psnr_metric(f_T1, real_T1)
    G_ssim_T1 = ssim_metric(f_T1, real_T1)
    return G_psnr_T2,G_ssim_T2,G_psnr_T1,G_ssim_T1

  def CreateGrid(self,r_T1, f_T1, C_T1, r_T2, f_T2, C_T2):
    """
    @Description: 
      Creates a grid of images for visualization purposes.

    @Inputs:
      - r_T1 (Tensor): Real images from domain T1.
      - f_T1 (Tensor): Generated images from domain T2 to T1.
      - C_T1 (Tensor): Reconstructed images from domain T1 to T2 and back to T1.
      - r_T2 (Tensor): Real images from domain T2.
      - f_T2 (Tensor): Generated images from domain T1 to T2.
      - C_T2 (Tensor): Reconstructed images from domain T2 to T1 and back to T2.

    @Outputs:
      - grid (Tensor): Grid of images, combining the input images from different domains.

    """
    images = torch.cat((r_T1, f_T1, C_T1, r_T2, f_T2, C_T2), 0)
    grid = torchvision.utils.make_grid(images, nrow=3).detach()
    return grid



  def linear_decay(self,epoch, n_epochs, n_epochs_decay):
    """
    @Description: 
      Performs linear decay of a value based on the current epoch.

    @Inputs:
      - epoch (int): Current epoch.
      - n_epochs (int): Total number of epochs.
      - n_epochs_decay (int): Number of epochs for linear decay.

    @Outputs:
      decay_factor (float): The decay factor for the given epoch.

    """
    if epoch < n_epochs:
        return 1.0
    else:
        return 1.0 - max(0, (epoch - n_epochs) / n_epochs_decay)
