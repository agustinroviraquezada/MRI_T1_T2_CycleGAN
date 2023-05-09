import os

class DirectoryCreator:
    def __init__(self, base_path):
        self.base_path = base_path

    def create_directories(self):
        T1_path = os.path.join(self.base_path, "T1")
        T2_path = os.path.join(self.base_path, "T2")
        model_path = os.path.join(self.base_path, "Model")
        checkpoint_path = os.path.join(model_path, "CheckPoint")
        best_path = os.path.join(model_path, "BestModel")

        # Create T1 directory
        os.makedirs(T1_path, exist_ok=True)
        
        # Create T2 directory
        os.makedirs(T2_path, exist_ok=True)

        # Create Model directory
        os.makedirs(model_path, exist_ok=True)

        # Create CheckPoint directory
        os.makedirs(checkpoint_path, exist_ok=True)

        # Create Best directory
        os.makedirs(best_path, exist_ok=True)
        return {"T1":T1_path,"T2":T2_path},model_path,checkpoint_path,best_path