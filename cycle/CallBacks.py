import os
import os.path
import lightning.pytorch as pl
import torchvision


##############################  Class 1 ################################

class CustomModelCheckpoint(pl.Callback):
   """
    @Description:
      This callback extends the functionality of the PyTorch Lightning ModelCheckpoint callback. It saves the best model based on a specified metric and also allows saving models periodically during training.

    @Input:
      - monitor (str): The metric to monitor for saving the best model.
      - save_path (str): The path where the model checkpoints will be saved.
      - save_best (str): The path where the best model checkpoint will be saved.
      - version (int): The version number to append to the saved model checkpoints.
      - save_top_k (int): The maximum number of best models to keep.
      - every_n_epochs (int): Save a model checkpoint every N epochs.
  """
  def __init__(self, monitor, save_path,save_best,version,save_top_k=3,every_n_epochs=None):
    super().__init__()
    self.monitor = monitor
    self.save_path = save_path
    self.save_top_k = save_top_k
    self.every_n_epochs = every_n_epochs
    self.best_k_models = {}
    self.save_best=save_best
    self.kth_best_value = None
    self.version=version
    


  def on_validation_epoch_end(self, trainer, pl_module):
    """
    @Description:
      This function is called at the end of each validation epoch. It saves the best model based on the monitored metric and periodically saves models during training.

    @Input:
      - trainer (pytorch_lightning.trainer.Trainer): The trainer object.
      - pl_module (pytorch_lightning.LightningModule): The PyTorch Lightning module.
    """
    current_value = trainer.logged_metrics.get(self.monitor)

    if len(self.best_k_models) < self.save_top_k or current_value > self.kth_best_value:
      ckpt_path = os.path.join(self.save_best, f"best_model_version_{self.version}_{current_value:.3f}-{trainer.current_epoch:02d}.ckpt")
      trainer.save_checkpoint(ckpt_path)

      if len(self.best_k_models) == self.save_top_k:
        # Remove the worst model
        worst_model_path = min(self.best_k_models, key=self.best_k_models.get)
        del self.best_k_models[worst_model_path]
        os.remove(worst_model_path)

      self.best_k_models[ckpt_path] = current_value
      self.kth_best_value = min(list(self.best_k_models.values()))

  def on_train_epoch_end(self, trainer, pl_module):
    """
    @Description:
      This function is called at the end of each training epoch. It saves a model checkpoint every N epochs, where N is the specified value in the 'every_n_epochs' parameter.

    @Input:
      - trainer (pytorch_lightning.trainer.Trainer): The trainer object.
      - pl_module (pytorch_lightning.LightningModule): The PyTorch Lightning module.
    """
    if self.every_n_epochs is not None and trainer.current_epoch % self.every_n_epochs == 0:
      ckpt_path = os.path.join(self.save_path, f"model__version_{self.version}_{trainer.current_epoch:02d}.ckpt")
      trainer.save_checkpoint(ckpt_path)



##############################  Class 2 ################################

class CreateGIF(pl.Callback):
   """
  @Description:
    This callback creates GIF images at the end of each validation epoch, showing the generated images and their reconstructions.

  @Input:
    - im_T1 (torch.Tensor): Input image from domain T1.
    - im_T2 (torch.Tensor): Input image from domain T2.
    - save_path (str): The path where the GIF images will be saved.
  """
  def __init__(self, im_T1, im_T2, save_path):
    super().__init__()
    self.save_path = save_path
    self.im_T1 = im_T1
    self.im_T2 = im_T2

  def on_validation_epoch_end(self, trainer, pl_module):
    f_T1 = pl_module.G_T2_T1(self.im_T2)
    f_T2 = pl_module.G_T1_T2(self.im_T1)

    C_T1 = pl_module.G_T2_T1(f_T2)
    C_T2 = pl_module.G_T1_T2(f_T1)

    epoch=trainer.current_epoch

    path_f_T1 = os.path.join(self.save_path, f"f_T1_epoch{epoch:02d}.png")
    path_f_T2 = os.path.join(self.save_path, f"f_T2_epoch{epoch:02d}.png")
    path_C_T1 = os.path.join(self.save_path, f"C_T1_epoch{epoch:02d}.png")
    path_C_T2 = os.path.join(self.save_path, f"C_T2_epoch{epoch:02d}.png")

    torchvision.utils.save_image(f_T1, path_f_T1)
    torchvision.utils.save_image(f_T2, path_f_T2)
    torchvision.utils.save_image(C_T1, path_C_T1)
    torchvision.utils.save_image(C_T2, path_C_T2)
