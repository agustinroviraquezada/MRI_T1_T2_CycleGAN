import logging
import sys
import optuna
import os
from lightning.pytorch.callbacks import Callback
from optuna.integration import PyTorchLightningPruningCallback
import lightning.pytorch as pl
from cycle.CycleGAN import CycleGAN
from cycle.DataMod import CycleGANDataModule
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_slice
from optuna.visualization import plot_contour
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_parallel_coordinate
import plotly.io as pio
import optuna.importance as importance
import pandas as pd



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

    
    self.pruner = optuna.pruners.MedianPruner()
    self.study = optuna.create_study(direction="maximize", pruner=self.pruner)
    self.study.optimize(self.objective, n_trials=funcParam["n_trials"], timeout=9000)

    # Print the model report to the log file
    with open(self.funcParam["logs"], 'a') as f:
      f.write("Best trial:\n")
      f.write("  Trial Number: {}\n".format(self.study.best_trial.number))
      f.write("  Value: {}\n".format(self.study.best_trial.value))
      f.write("  Params:\n")
      for key, value in self.study.best_trial.params.items():
          f.write("    {}: {}\n".format(key, value))


    self.BestParameter=self.study.best_trial.params   

    ######## Create Plots.
    sorted_hyperparameters = sorted(importance.get_param_importances(self.study).items(), key=lambda x: x[1], reverse=True)
    # Select the three most important hyperparameters
    important_hyperparameters = [param[0] for param in sorted_hyperparameters[:3]]
    all_hyperparameters = [param[0] for param in sorted_hyperparameters]

    # Get a list of Hyperparameters
    p=os.path.dirname(self.funcParam["logs"])
    data = []
    for trial in self.study.trials:
        data.append({**trial.params, 'Objective Value': trial.value})

    df = pd.DataFrame(data)
    df=df.sort_values('Objective Value',ascending=False)
    df.to_csv(os.path.join(p,"ListHyperValues"),index=False)
    
    #Counter Plots
    figure=plot_contour(self.study, params=all_hyperparameters)  # Specify the parameters to plot
    figure.update_layout(width=1600, height=1200)
    pio.write_html(figure, file=os.path.join(p,'plot_contour.html'))
    #-----
    figure=plot_contour(self.study, params=important_hyperparameters)  # Specify the parameters to plot
    figure.update_layout(width=1600, height=1200)
    pio.write_html(figure, file=os.path.join(p,'plot_contour_important_hyperparameters.html'))
    #-----
    figure=plot_slice(self.study)
    pio.write_html(figure, file=os.path.join(p,'plot_slice.html'))
    #-----
    figure=plot_param_importances(self.study)
    pio.write_html(figure, file=os.path.join(p,'plot_param_importances.html'))




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
  
  
  
 

def GetBestCombination(path):
  df=pd.read_csv(path).sort_values(by=['Objective Value'],ascending=False,ignore_index=True)
  df=df[df["Objective Value"]>0.50]

  # Get variables
  columns_to_average = list(df.columns)[:-1]
  objective_values = df['Objective Value']

  # Computes the weighted average
  weighted_average = [np.average(df[c], weights=objective_values) for c in columns_to_average]
  print(f"{columns_to_average}\n{weighted_average}")
