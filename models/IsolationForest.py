import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Dict, Any
import os
from Config.config import Config
from models.model_interface import model_interface

class TrainIsolationForest(model_interface):
    """A class for running Isolation Forest for Anomaly Detection."""
    
    def __init__(self, random_state: int = 42, contamination: str | float = 'auto', model_path: str | None = None, **kwargs):
        """Initializes the Isolation Forest model with key parameters."""
        super().__init__(model_path=model_path)
        self.random_state = random_state
        self.contamination = contamination 
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.cfg = Config()

    def get_loaded_model_details(self):
        """Returns details about the loaded model."""
        return super().get_loaded_model_details()

    def train_model_with_params(self, train, test, **kwargs):
        return super().train_model_with_params(train, test, **kwargs)
    
    def fine_tune_model(self, train, test, filepath) -> object:
        return super().fine_tune_model(train, test, filepath)

    def run(self, val: pd.DataFrame):
        """Fits the Isolation Forest on the data and predicts anomaly scores and labels."""

        print("\n--- Running Isolation Forest Anomaly Detection ---")

        
        # Drop the target column as Isolation Forest is unsupervised
        feature_val = val.drop(columns=[self.cfg.CLASS_TARGET], errors='ignore')
            
        # Fit the model to the features
        self.model.fit(feature_val)

        # Predict anomaly scores and labels
        val["anomaly_score"] = self.model.decision_function(feature_val)
        val["anomaly"] = self.model.predict(feature_val)

        anomaly_count = val["anomaly"].value_counts()
        print("Isolation Forest Detection Complete. Anomaly Labels:")
        print(anomaly_count)
        


    def save_model(self, filepath: str,filename:str):
        """Saves the trained model to the specified filepath."""
        return super().save_model(filepath,filename)