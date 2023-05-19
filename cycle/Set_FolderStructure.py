import os

class DirectoryCreator:
  """
  @Description:
    The DirectoryCreator class provides a method to create multiple directories for organizing files. It creates directories for T1, T2, Model, CheckPoint, CheckPoint_Opt, GIF, and BestModel.

  @Input:
    - base_path (str): The base path where the directories will be created.

  @Output:
    - dict: A dictionary containing the created directory paths for T1 and T2.
    - str: The path of the Model directory.
    - str: The path of the CheckPoint directory.
    - str: The path of the BestModel directory.
    - str: The path of the CheckPoint_Opt directory.
    - str: The path of the GIF directory.
  """
  def __init__(self, base_path):
    self.base_path = base_path

  def create_directories(self):
    """
    @Description
      Create directories for T1, T2, Model, CheckPoint, CheckPoint_Opt, GIF, and BestModel.

    @Output:
      - dict: A dictionary containing the created directory paths for T1 and T2.
      - str: The path of the Model directory.
      - str: The path of the CheckPoint directory.
      - str: The path of the BestModel directory.
      - str: The path of the CheckPoint_Opt directory.
      - str: The path of the GIF directory.
    """
    T1_path = os.path.join(self.base_path, "T1")
    T2_path = os.path.join(self.base_path, "T2")
    
    Test_T1_path = os.path.join(self.base_path, "Test/T1")
    Test_T2_path = os.path.join(self.base_path, "Test/T2")
    
    model_path = os.path.join(self.base_path, "Model")
    checkpoint_path = os.path.join(model_path, "CheckPoint")
    checkOptimization_path = os.path.join(model_path, "CheckPoint_Opt")
    gif_path = os.path.join(model_path, "GIF")
    best_path = os.path.join(model_path, "BestModel")

    # Create T1 directory
    os.makedirs(T1_path, exist_ok=True)
    os.makedirs(T2_path, exist_ok=True)
    
    # Create T2 directory
    os.makedirs(Test_T1_path, exist_ok=True)
    os.makedirs(Test_T2_path, exist_ok=True)

    # Create Model directory
    os.makedirs(model_path, exist_ok=True)

    # Create CheckPoint directory
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Create CheckPoint optimization directory
    os.makedirs(checkOptimization_path, exist_ok=True)
    
    # Create CheckPoint optimization directory
    os.makedirs(gif_path, exist_ok=True)

    # Create Best directory
    os.makedirs(best_path, exist_ok=True)
    return {"T1":T1_path,"T2":T2_path},{"T1":Test_T1_path,"T2":Test_T2_path},model_path,checkpoint_path,best_path,checkOptimization_path,gif_path
