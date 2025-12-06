import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, Optional
from sklearn.ensemble import IsolationForest

class TrainLinearRegressor:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.best_params = None

    def run_linear_regression(self, 
                              X_train: pd.DataFrame,
                              y_train: pd.Series,
                              X_test: pd.DataFrame,
                              y_test: pd.Series,
                              filepath:str,
                              alpha: float = 1.0,
                              regularization: str = 'none') -> pd.DataFrame:
        """
        Runs a regression model (Basic OLS, Ridge, or Lasso) and evaluates it.
        Returns: A dataframe comparing Actual vs Predicted values.
        """
        print(f"\n[Running {regularization.upper()} Regression] (alpha={alpha})")
        
        # 1. Select Model Architecture
        if regularization == 'ridge':
            self.model = Ridge(alpha=alpha, random_state=self.random_state)
        elif regularization == 'lasso':
            self.model = Lasso(alpha=alpha, random_state=self.random_state)
        else:
            self.model = LinearRegression(n_jobs=-1)

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

        """Saves regression scores to a markdown file."""
        with open(filepath, "w") as f:
            f.write(f"# Linear Regression Results\n\n")
            f.write("## Performance Metrics\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| RMSE | {rmse:.4f} |\n")
            f.write(f"| MAE | {mae:.4f} |\n")
            f.write(f"| R² | {r2:.4f} |\n")
        # 4. Return Results
        results = X_test.copy()
        results['Actual_LOS'] = y_test
        results['Predicted_LOS'] = preds
        return results

    def tune_linear_regression(self, 
                               X_train: pd.DataFrame, 
                               y_train: pd.Series, 
                               model_type: str = 'ridge') -> Dict:
        """
        Performs GridSearchCV to find the best 'alpha' (regularization strength).
        Note: We tune Ridge/Lasso because basic Linear Regression has no params to tune.
        """
        print(f"\n[Starting Hyperparameter Tuning for {model_type.upper()}]...")
        
        # Define Search Space (Alphas to try)
        param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]}
        
        # Define Model
        if model_type == 'lasso':
            estimator = Lasso(random_state=self.random_state)
        else:
            estimator = Ridge(random_state=self.random_state)

        # Run Grid Search
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring='neg_root_mean_squared_error', # Optimize for lowest error
            cv=5,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"  ✔ Best Alpha Found: {grid_search.best_params_['alpha']}")
        print(f"  ✔ Best CV Score (RMSE): {-grid_search.best_score_:.4f}")
        
        return grid_search.best_params_

    def run_isolation_forest(self,
                             df: pd.DataFrame,
                             feature_list: list,
                             contamination="auto",
                             random_state=42):

        print("\n[Running Isolation Forest for Anomaly Detection]")

        df_copy = df.copy()

        self.iso_model = IsolationForest(
            contamination=contamination,
            random_state=random_state
        )
        self.iso_model.fit(df_copy[feature_list])

        df_copy["anomaly_score"] = self.iso_model.decision_function(df_copy[feature_list])
        df_copy["anomaly"] = self.iso_model.predict(df_copy[feature_list])

        print("Isolation Forest Completed")
        print(df_copy["anomaly"].value_counts())

        return df_copy

      

