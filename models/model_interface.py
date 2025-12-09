import pickle
import os
from abc import ABC,abstractmethod
class model_interface(ABC):
    def __init__(self, model_path:str| None = None) -> None:
        if model_path is not None:
            with open(model_path,'rb') as f:
                self.model = pickle.load(f)
        else:
            self.model = None

    @abstractmethod
    def get_loaded_model_details(self):
        if self.model is None:
            print("No model loaded. Train or load a model first.")
            return

        print("Loaded Model Details")
        print("-" * 40)
        print(f"Model Type: {type(self.model).__name__}")

        try:
            params = self.model.get_params()
            print("\nModel Parameters:")
            for k, v in params.items():
                print(f"  {k}: {v}")
        except:
            print("\n(No parameters available for this model)")

        print("-" * 40)



    @abstractmethod
    def run(self,val):
        pass

    @abstractmethod
    def fine_tune_model(self,train,test,filepath) -> object:
        pass

    @abstractmethod
    def train_model_with_params(self,train,test,**kwargs):
        pass

    @abstractmethod
    def save_model(self, filepath: str):
        if self.model == None:
            ValueError("please train the model")

        directory = os.path.dirname(filepath)

        # Ensure directory exists
        if directory:
            os.makedirs(directory, exist_ok=True)

        # Save model with pickle
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)
        print(f"model saved to {filepath}")
    
    