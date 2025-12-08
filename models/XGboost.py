import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict

class TrainXGBoostRegressor:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.best_params = None

    def run_xgboost(self,
                    X_train: pd.DataFrame,
                    y_train: pd.Series,
                    X_test: pd.DataFrame,
                    y_test: pd.Series,
                    filepath: str,
                    learning_rate: float = 0.1,
                    max_depth: int = 6,
                    n_estimators: int = 200,
                    subsample: float = 1.0,
                    colsample_bytree: float = 1.0) -> pd.DataFrame:
        """
        Runs an XGBoost regression model and evaluates it.
        Returns: A dataframe comparing Actual vs Predicted values.
        """

        print(f"\n[Running XGBOOST Regression] (n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate})")

        # 1. Select Model Architecture
        self.model = XGBRegressor(
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=self.random_state,
            n_jobs=-1,
            objective="reg:squarederror"
        )

        # 2. Train (Fit)
        self.model.fit(X_train, y_train)

        # 3. Predict & Evaluate
        preds = self.model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        print(f"  -> RMSE: {rmse:.4f}")
        print(f"  -> MAE:  {mae:.4f}")
        print(f"  -> R2:   {r2:.4f}")

        # Save markdown performance report
        with open(filepath, "w") as f:
            f.write(f"# XGBoost Regression Results\n\n")
            f.write("## Performance Metrics\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| RMSE | {rmse:.4f} |\n")
            f.write(f"| MAE | {mae:.4f} |\n")
            f.write(f"| R² | {r2:.4f} |\n")

        # 4. Return Results
        results = X_test.copy()
        results["Actual_LOS"] = y_test
        results["Predicted_LOS"] = preds
        return results

    def tune_xgboost(self,
                     X_train: pd.DataFrame,
                     y_train: pd.Series) -> Dict:
        """
        Performs GridSearchCV to find the best hyperparameters for XGBoost.
        """

        print("\n[Starting Hyperparameter Tuning for XGBOOST]...")

        # Define Search Space
        param_grid = {
            "n_estimators": [200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        }

        # Define Model
        estimator = XGBRegressor(
            random_state=self.random_state,
            n_jobs=-1,
            objective="reg:squarederror"
        )

        # Run Grid Search
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=5,
            verbose=1
        )
        grid_search.fit(X_train, y_train)

        print(f"   Best Params: {grid_search.best_params_}")
        print(f"   Best CV Score (RMSE): {-grid_search.best_score_:.4f}")

        self.best_params = grid_search.best_params_
        return self.best_params

