import os
import os.path
import lightning.pytorch as pl
import torchvision
import torch

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

class CreateTensor(pl.Callback):
  """
  @Description:
  This callback saves an image as a PyTorch tensor at the end of the first step in each validation epoch.

  @Input:
  - save_path (str): The path where the tensor will be saved.
  """
  def __init__(self, save_path):
    super().__init__()
    self.save_path = save_path

  def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    if batch_idx == 0:
      # Get the generated image from the validation step outputs
      generated_image = outputs['images']
      
      # Save the tensor
      epoch = trainer.current_epoch

      for k,v in generated_image.items():
        path_generated_image = os.path.join(self.save_path, f"{k}_epoch{epoch:02d}.pt")
        torch.save(v, path_generated_image)
