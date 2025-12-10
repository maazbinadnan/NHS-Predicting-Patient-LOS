import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from typing import Dict
import os
from Config.config import Config
from models.model_interface import model_interface


class TrainRandomForestRegressor(model_interface):
    """A class for training and tuning Random Forest Regression models."""
    
    def __init__(self, random_state: int = 42, model_path: str | None = None, **kwargs):
        """Initializes the Random Forest regressor with a random state."""
        super().__init__(model_path=model_path)
        self.random_state = random_state
        self.best_params = None
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None
        self.cfg = Config()

    def get_loaded_model_details(self):
        """Returns details about the loaded model."""
        return super().get_loaded_model_details()

    # -----------------------------------------------------
    # TRAIN WITH SPECIFIED PARAMETERS
    # -----------------------------------------------------
    def train_model_with_params(self, train: pd.DataFrame, test: pd.DataFrame, **kwargs):
        """Initializes, trains, and evaluates the Random Forest model."""
        
        # Initialize Random Forest Regressor with given params
        self.model = RandomForestRegressor(
            random_state=self.random_state,
            n_jobs=-1,
            **kwargs
        )

        # Extract X and Y using Config
        self.x_train = train.drop(columns=self.cfg.TARGET)
        self.y_train = train[self.cfg.TARGET]
        self.x_test = test.drop(columns=self.cfg.TARGET)
        self.y_test = test[self.cfg.TARGET]

        # Fit model
        self.model.fit(self.x_train, self.y_train)

        # Evaluate
        preds = self.model.predict(self.x_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, preds))
        mae = mean_absolute_error(self.y_test, preds)
        r2 = r2_score(self.y_test, preds)

        print("--- Random Forest Model Evaluation ---")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")

    # -----------------------------------------------------
    # RUN ON VALIDATION SET
    # -----------------------------------------------------
    def run(self, val: pd.DataFrame):
        """Runs the trained model on a validation set and prints metrics."""

        y_val = val[self.cfg.TARGET]
        x_val = val.drop(columns=[self.cfg.TARGET])

        if self.model is None:
            raise ValueError("No model provided. Train or load one first.")

        preds = self.model.predict(x_val)        
        return preds

    def evaluate_model(self,original_values,predictions):
        rmse = np.sqrt(mean_squared_error(original_values, predictions))
        mae = mean_absolute_error(original_values, predictions)
        r2 = r2_score(original_values, predictions)

        print("--- Evaluation  (Random Forest Regression) ---")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R²:   {r2:.4f}")     


    # -----------------------------------------------------
    # HYPERPARAM TUNING
    # -----------------------------------------------------
    def fine_tune_model(self, train: pd.DataFrame, test: pd.DataFrame, filepath: str):
        """Performs Grid Search to find the best hyperparameters for Random Forest."""

        # 1. Extract X and Y
        X_train = train.drop(columns=self.cfg.TARGET)
        y_train = train[self.cfg.TARGET]
        X_test = test.drop(columns=self.cfg.TARGET)
        y_test = test[self.cfg.TARGET]

        print("\n--- Starting Hyperparameter Tuning for Random Forest ---")

        # 2. Setup tuning parameters
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }

        # 3. Define base model and Grid Search
        base_model = RandomForestRegressor(
            random_state=self.random_state,
            n_jobs=-1
        )
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=5,
            verbose=2,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        # 4. Extract and evaluate best model
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_

        preds = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        print("\n--- Fine-Tuned Random Forest Model Evaluation ---")
        print(f"Best Parameters: {self.best_params}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R²:   {r2:.4f}")

        # 5. Save markdown output
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write("# Random Forest Regression Results\n\n")
            f.write(f"Best Parameters: {self.best_params}\n\n")
            f.write("## Performance Metrics\n")
            f.write(f"- RMSE: {rmse:.4f}\n")
            f.write(f"- MAE: {mae:.4f}\n")
            f.write(f"- R² Score: {r2:.4f}\n")

        return grid_search.best_params_

    # -----------------------------------------------------
    # SAVE MODEL
    # -----------------------------------------------------
    def save_model(self, filepath: str,filename:str):
        """Saves the trained model to the specified filepath."""
        return super().save_model(filepath,filename)