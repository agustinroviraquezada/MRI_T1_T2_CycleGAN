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
  """
    @Description:
      This class extends the functionality of PyTorchLightningPruningCallback and Callback classes. It is used for handling pruning in PyTorch Lightning trials.

    @Input:
      - trial (optuna.trial.Trial): Optuna trial object.
      - monitor (str): Name of the metric to monitor for pruning.

    @Output:
      None
  """
  def __init__(self, trial, monitor):
    super().__init__(trial, monitor)


class HyperParametrization():
  """
  @Description:
    The HyperParametrization class performs hyperparameter optimization using Optuna library. It searches for the best set of hyperparameters for a given function.

  @Input:
    - funcParam (dict): Dictionary containing the parameters for the objective function.
    - hyperparameters (dict): Dictionary containing the hyperparameters to be optimized.
  """
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
    #func = lambda trial: self.objective(trial, self.hyperparameters, self.funcParam)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(self.objective, n_trials=funcParam["n_trials"], timeout=9000)

    # Print the model report to the log file
    with open(self.funcParam["logs"], 'a') as f:
      f.write("Best trial:\n")
      f.write("  Trial Number: {}\n".format(study.best_trial.number))
      f.write("  Value: {}\n".format(study.best_trial.value))
      f.write("  Params:\n")
      for key, value in study.best_trial.params.items():
          f.write("    {}: {}\n".format(key, value))


    self.BestParameter=study.best_trial.params   

  def objective(self, trial: optuna.trial.Trial) -> float:
    """
    @Description:
      This method defines the objective function that is used for hyperparameter optimization. It trains the CycleGAN model and returns the value of the metric being optimized.

    @Input:
      - trial (optuna.trial.Trial): Optuna trial object.

    @Output:
      - float: The value of the metric being optimized.
    """

    
    #Define hyperParameters
    hyperp=self.Gen_HyperPar(trial)


    # Create the CycleGAN model with the specified params and Networks
    model = CycleGAN(hyperp)

    # DataModule
    datamodule = CycleGANDataModule(self.funcParam["paths"],
                                    batch_size=self.funcParam["batch_size"],
                                    num_workers=self.funcParam["num_workers"],
                                    factor=self.funcParam["factor"])

    trainer = pl.Trainer(
        logger=True,
        limit_train_batches=self.funcParam["limit_train_batches"],
        limit_val_batches=self.funcParam["limit_val_batches"],
        max_epochs=self.funcParam["epoch"],
        accelerator="gpu",
        callbacks=[PyTorchLightningPruningCallback(trial, monitor=self.funcParam["monitor"])],
        enable_model_summary=False
    )

    trainer.logger.log_hyperparams(hyperp)
    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics[self.funcParam["monitor"]].item()

  def Gen_HyperPar(self,trial: optuna.trial.Trial)-> dict:
    """
    @Description:
      This method generates the hyperparameters for a given trial.

    @Input:
      - trial (optuna.trial.Trial): Optuna trial object.

    @Output:
      - dict: Dictionary containing the generated hyperparameters.
    """
    
    hyperparameters = {}
    for key, value in self.hyperparameters.items():
        if key=="lr":
          hyperparameters[key]=trial.suggest_float(key, value[0], value[1], log=True)
        else:
          if isinstance(value, tuple):
              if all(isinstance(i, int) for i in value):
                  hyperparameters[key] = trial.suggest_int(key, value[0], value[1])
              elif all(isinstance(i, (float, int)) for i in value):
                  hyperparameters[key] = trial.suggest_float(key, value[0], value[1])
          else:
              hyperparameters[key] = value
    return hyperparameters
