import pickle
from abc import ABC,abstractmethod
class model_interface(ABC):
    def __init__(self, model_path:str| None = None) -> None:
        if model_path is not None:
            with open(model_path,'rb') as f:
                self.model = pickle.load(f)
        else:
            self.model = None

    @abstractmethod
    def run(self,val,**kwargs):
        pass

    @abstractmethod
    def fine_tune_model(self,train,test):
        pass

    @abstractmethod
    def train_model_with_params(self,train,test,**kwargs):
        pass