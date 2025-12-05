import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# 1. Trainer Class
# -------------------------------
class ModelTrainer:
    def __init__(self, save_dir="saved_models"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def _evaluate(self, model, X_test, y_test):
        """Helper to calculate metrics."""
        preds = model.predict(X_test)
        return {
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "R2": r2_score(y_test, preds)
        }

    def train_test_split(self, df: pd.DataFrame, target_col: str, training_phase: str,test_size=0.3):
        """
        Trains LR and RF models for a specific experiment (e.g., 'Global_Inliers').
        """
        print(f"\n🚀 train test split: {training_phase}")
        
        # 1. Data Prep
        if target_col not in df.columns:
            raise ValueError(f"Target '{target_col}' not found in dataframe.")
            
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_and_save(self, df: pd.DataFrame, target_col: str, experiment_name: str):
        """
        Trains LR and RF models for a specific experiment (e.g., 'Global_Inliers').
        """
        print(f"\n🚀 Starting Training: {experiment_name}")
        
        # 1. Data Prep
        if target_col not in df.columns:
            raise ValueError(f"Target '{target_col}' not found in dataframe.")
            
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        results = []

        # 2. Linear Regression
        lr_name = f"{experiment_name}_LinearRegression"
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        
        # Save & Evaluate
        joblib.dump(lr_model, os.path.join(self.save_dir, f"{lr_name}.pkl"))
        metrics_lr = self._evaluate(lr_model, X_test, y_test)
        metrics_lr['Model'] = "LinearRegression"
        metrics_lr['Experiment'] = experiment_name
        results.append(metrics_lr)

        # 3. Random Forest
        rf_name = f"{experiment_name}_RandomForest"
        rf_model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Save & Evaluate
        joblib.dump(rf_model, os.path.join(self.save_dir, f"{rf_name}.pkl"))
        metrics_rf = self._evaluate(rf_model, X_test, y_test)
        metrics_rf['Model'] = "RandomForest"
        metrics_rf['Experiment'] = experiment_name
        results.append(metrics_rf)


        # 3. Random Forest
        xg_name = f"{experiment_name}_XGboost"
        xg_model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=600,
        learning_rate=0.03,
        max_depth=10,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_alpha=1,
        reg_lambda=2,
        tree_method="hist",
        random_state=42
            )
        xg_model.fit(X_train, y_train)
        
        # Save & Evaluate
        joblib.dump(rf_model, os.path.join(self.save_dir, f"{xg_name}.pkl"))
        metrics_xg = self._evaluate(xg_model, X_test, y_test)
        metrics_xg['Model'] = "XGboost Regressor"
        metrics_xg['Experiment'] = experiment_name
        results.append(metrics_xg)

        return pd.DataFrame(results)

# -------------------------------
# 2. Main Execution
# -------------------------------
if __name__ == "__main__":
    # Configuration
    MODEL_DIR = os.path.join(os.getcwd(), "saved_models")
    TARGET = 'spell_episode_los'
    
    # Initialize Trainer
    trainer = ModelTrainer(save_dir=MODEL_DIR)
    
    # Load Data (Assuming these are the clean, ENCODED files from previous steps)
    # If using the pipeline from the previous step, pass the dataframes directly here.
    try:
        path_inliers = os.path.join(os.getcwd(), "Normal_Data.csv")
        path_outliers = os.path.join(os.getcwd(), "Outlier_Data.csv")
        
        df_inliers = pd.read_csv(path_inliers)
        df_outliers = pd.read_csv(path_outliers)

        # ⚠️ CRITICAL: Ensure Data is Numeric
        # If the CSVs still have strings, you must run the Encoding Pipeline first.
        # This block assumes df_inliers and df_outliers are ready for ML.

        # Train Global Inlier Model
        res_inliers = trainer.train_and_save(
            df=df_inliers, 
            target_col=TARGET, 
            experiment_name="Global_Inliers"
        )

        # Train Global Outlier Model
        res_outliers = trainer.train_and_save(
            df=df_outliers, 
            target_col=TARGET, 
            experiment_name="Global_Outliers"
        )

        # Show Final Report
        final_report = pd.concat([res_inliers, res_outliers], ignore_index=True)
        print("\n🏆 Final Performance Report:")
        print(final_report.to_string(index=False))

    except FileNotFoundError as e:
        print(f"❌ Error: Could not find data files. {e}")
    except Exception as e:
        print(f"❌ An error occurred: {e}")