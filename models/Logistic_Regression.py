import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from typing import Dict
import pickle
import os
from Config.config import Config
from models.model_interface import model_interface

class TrainLogisticRegression(model_interface):
    def __init__(self, random_state: int = 42, model_path:str | None=None):
        super().__init__(model_path=model_path)
        self.random_state = random_state
        self.best_params = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.cfg = Config()
    
    def get_loaded_model_details(self):
        return super().get_loaded_model_details()

    def train_model_with_params(self, train: pd.DataFrame, test: pd.DataFrame, **kwargs):
        # Initialize model (incorporating kwargs like n_estimators, class_weight, etc.)
        self.model = LogisticRegression(**kwargs,verbose=5) 

        # Extracting X and Y's
        self.x_train = train.drop(columns=self.cfg.CLASS_TARGET)
        self.y_train = train[self.cfg.CLASS_TARGET]
        
        # Extract X_test and y_test
        self.x_test = test.drop(columns=self.cfg.CLASS_TARGET)
        self.y_test = test[self.cfg.CLASS_TARGET]

        # Fitting the model
        self.model.fit(self.x_train, y=self.y_train)

        # Metric evaluation (Calculate key metrics using the test set)
        y_pred, y_prob = self.model.predict(self.x_test), self.model.predict_proba(self.x_test)[:, 1]
        print("--- Model Evaluation ---")
        print(classification_report(self.y_test, y_pred))
        print(f"ROC-AUC Score: {roc_auc_score(self.y_test, y_prob):.4f}")
        

    def run(self,val:pd.DataFrame):

        y_val = val[self.cfg.CLASS_TARGET]    
        x_val = val.drop(columns = self.cfg.CLASS_TARGET)
        

        if self.model == None:
            raise ValueError("no model given please either train one or enter path to model when instantiating the class")

        val_predictions,val_probability = self.model.predict(x_val),self.model.predict_proba(x_val)[:, 1]
        print(classification_report(y_val,val_predictions))
        print(f"ROC-AUC Score: {roc_auc_score(y_val, val_probability):.4f}")

        return val_predictions


    def fine_tune_model(self, train: pd.DataFrame, test: pd.DataFrame,filepath:str,**kwargs):
        # Extract X and Y for both sets
        X_train = train.drop(columns=self.cfg.CLASS_TARGET)
        y_train = train[self.cfg.CLASS_TARGET]
        X_test = test.drop(columns=self.cfg.CLASS_TARGET)
        y_test = test[self.cfg.CLASS_TARGET]

        param_grid = {
            'penalty': ['l1', 'l2', 'elasticnet'],
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'class_weight': ['balanced', None],
            'max_iter': [100, 200, 500],
        }

        # Initialize base model for Grid Search
        base_model = LogisticRegression(random_state=self.random_state, n_jobs=-1)
        
        # Setup Grid Search CV to optimize for ROC-AUC
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=3,
            scoring='roc_auc',
            verbose=2,
            n_jobs=-1
        )
        
        # Fit Grid Search on training data
        print("\n--- Starting Grid Search for Hyperparameter Tuning ---")
        grid_search.fit(X_train, y_train)

        # Update the class attributes with the best model found
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_ 

        # Evaluate the best model on the test set
        y_pred, y_prob = self.model.predict(X_test), self.model.predict_proba(X_test)[:, 1]
        print("\n--- Fine-Tuned Model Evaluation on Test Set ---")
        print(f"Best Parameters Found: {self.best_params}")
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

        #save it to a markdown file as a table
        directory = os.path.dirname(filepath)
    
        # 2. Check if the directory exists and create it if necessary
        if directory:
            os.makedirs(directory, exist_ok=True)

        # 3. Write the scores to the file
        try:
            with open(filepath, "w") as f:
                f.write(f"# Logistic Regression Classifier Results\n\n")
                f.write(f"Best Parameters Found: {self.best_params}\n\n")
                f.write("## Classification Report\n")
                f.write(f"{classification_report(y_test, y_pred)}")
                f.write(f"\nROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}\n")

            print(f"✅ Scores successfully saved to: {filepath}")

        except Exception as e:
            print(f"❌ Error saving scores to file: {e}")    


        return grid_search.best_params_
    
    def save_model(self, filepath: str,filename:str):
        """Saves the trained model to the specified filepath."""
        return super().save_model(filepath,filename)