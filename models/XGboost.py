import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from typing import Dict
import os
from Config.config import Config
from models.model_interface import model_interface


class TrainXGBoostRegressor(model_interface):
    def __init__(self, random_state: int = 42, model_path: str | None = None):
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

    # -----------------------------------------------------
    # TRAIN WITH SPECIFIED PARAMETERS
    # -----------------------------------------------------
    def train_model_with_params(self, train: pd.DataFrame, test: pd.DataFrame, **kwargs):
        
        # initialize XGBoost Regressor with given params
        self.model = XGBRegressor(
            random_state=self.random_state,
            n_jobs=-1,
            objective="reg:squarederror",
            **kwargs
        )

        # Extract X and Y
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

        print("--- XGBoost Model Evaluation ---")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")

        return preds

    # -----------------------------------------------------
    # RUN ON VALIDATION SET
    # -----------------------------------------------------
    def run(self, val: pd.DataFrame):

        y_val = val[self.cfg.TARGET]
        x_val = val.drop(columns=[self.cfg.TARGET])

        if self.model is None:
            raise ValueError("No model provided. Train or load one first.")

        preds = self.model.predict(x_val)

        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)

        print("--- Validation Run (XGBoost) ---")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R²:   {r2:.4f}")

    # -----------------------------------------------------
    # HYPERPARAM TUNING
    # -----------------------------------------------------
    def fine_tune_model(self, train: pd.DataFrame, test: pd.DataFrame, filepath: str):

        X_train = train.drop(columns=self.cfg.TARGET)
        y_train = train[self.cfg.TARGET]

        X_test = test.drop(columns=self.cfg.TARGET)
        y_test = test[self.cfg.TARGET]

        param_grid = {
            "n_estimators": [200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        }

        base_model = XGBRegressor(
            random_state=self.random_state,
            n_jobs=-1,
            objective="reg:squarederror"
        )

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=5,
            verbose=2,
            n_jobs=-1
        )

        print("\n--- Starting Hyperparameter Tuning for XGBoost ---")
        grid_search.fit(X_train, y_train)

        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_

        preds = self.model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        print("\n--- Fine-Tuned XGBoost Model Evaluation ---")
        print(f"Best Parameters: {self.best_params}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R²:   {r2:.4f}")

        # Save markdown output
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write("# XGBoost Regression Results\n\n")
            f.write(f"Best Parameters: {self.best_params}\n\n")
            f.write("## Performance Metrics\n")
            f.write(f"- RMSE: {rmse:.4f}\n")
            f.write(f"- MAE: {mae:.4f}\n")
            f.write(f"- R² Score: {r2:.4f}\n")

        return self.model

    # -----------------------------------------------------
    # SAVE MODEL
    # -----------------------------------------------------
    def save_model(self, filepath: str):
        return super().save_model(filepath)
