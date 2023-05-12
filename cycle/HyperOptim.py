import logging
import sys
import optuna
import os
from lightning.pytorch.callbacks import Callback
from optuna.integration import PyTorchLightningPruningCallback
import lightning.pytorch as pl
from cycle.CycleGAN import CycleGAN
from cycle.DataMod import CycleGANDataModule



class PyTorchLightningPruningCallback(PyTorchLightningPruningCallback, Callback):
  def __init__(self, trial, monitor):
    super().__init__(trial, monitor)


class HyperParametrization():
  def __init__(self,funcParam,hyperparameters):
    
    #Get parameters
    self.funcParam=funcParam
    self.hyperparameters=hyperparameters

    # Create a logger for Python
    logger_python = logging.getLogger()
    logger_python.setLevel(logging.INFO)  # Set the logging level

    # Create a logger for Optuna
    logger_optuna = logging.getLogger('optuna')
    logger_optuna.setLevel(logging.INFO)

    # Create a file handler
    handler = logging.FileHandler(self.funcParam["logs"])
    handler.setLevel(logging.INFO)

    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handlers to the loggers
    logger_python.addHandler(handler)
    logger_optuna.addHandler(handler)

    # Set Optuna's logging verbosity
    optuna.logging.set_verbosity(optuna.logging.INFO)

    # Now, when Optuna logs information, it will also be written to 'log.txt'
    func = lambda trial: self.objective(trial, self.hyperparameters, self.funcParam)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(func, n_trials=funcParam["n_trials"], timeout=9000)

    # Print the model report to the log file
    with open(self.funcParam["logs"], 'a') as f:
      f.write("Best trial:\n")
      f.write("  Trial Number: {}\n".format(study.best_trial.number))
      f.write("  Value: {}\n".format(study.best_trial.value))
      f.write("  Params:\n")
      for key, value in study.best_trial.params.items():
          f.write("    {}: {}\n".format(key, value))




  def objective(self, trial: optuna.trial.Trial,hyperparameters:dict,funcParam:dict) -> float:
    
    # Create the CycleGAN model with the specified params and Networks
    model = CycleGAN(hyperparameters)

    # DataModule
    datamodule = CycleGANDataModule(funcParam["paths"],
                                    batch_size=funcParam["batch_size"],
                                    num_workers=funcParam["num_workers"],
                                    factor=funcParam["factor"])

    trainer = pl.Trainer(
        logger=True,
        limit_train_batches=funcParam["limit_train_batches"],
        limit_val_batches=funcParam["limit_val_batches"],
        max_epochs=funcParam["epoch"],
        accelerator="gpu",
        callbacks=[PyTorchLightningPruningCallback(trial, monitor=funcParam["monitor"])],
    )

    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics[funcParam["monitor"]].item()
