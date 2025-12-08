import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score,recall_score,confusion_matrix
from typing import Dict
import pickle
import os
from config import Config
from model_interface import model_interface

class TrainRandomFClassifier(model_interface):
    def __init__(self, random_state: int = 42, model_path:str | None=None):
        super().__init__(model_path=model_path)
        self.random_state = random_state
        self.best_params = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.cfg = Config()
   
    def train_model_with_params(self,train:pd.DataFrame,test:pd.DataFrame,**kwargs):
        self.model = RandomForestClassifier()

        #extracting x and y's
        self.x_train = train.drop(columns = self.cfg.CLASS_TARGET)
        self.y_train = train[self.cfg.CLASS_TARGET]

        #fitting the model
        self.model.fit(self.x_train,y=self.y_train)

        #metric evaluation       



    def run(self,val, **kwargs):
        pass
