import pandas as pd
import numpy as np
from sklearn.preprocessing import TargetEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from typing import Tuple
from config import Config
import pickle


 
# -------------------------------
# 2. Medical Data Processor
# -------------------------------
class MedicalDataPreprocessor:
    def __init__(self, config=Config):
        self.cfg = config
        self.std_scaler = StandardScaler()
        self.rob_scaler = RobustScaler()
        # Initialize Encoder (smooth='auto' prevents overfitting on rare categories)
        self.encoder = TargetEncoder(target_type='continuous', smooth='auto', random_state=42)

    def initial_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes ID columns, leakage columns, and cleans specific formats."""
        # Drop columns if they exist
        drop_current = [c for c in self.cfg.DROP_COLS if c in df.columns]
        df = df.drop(columns=drop_current)

        # Fix Frailty Score
        if 'frailty_score' in df.columns:
            df['frailty_score'] = (
                df['frailty_score'].astype(str).str.extract(r'(\d+)').astype(float)
            )
        return df

    def fill_na_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imputes missing values using defined defaults."""
        return df.fillna(self.cfg.NA_DEFAULTS)

    def split_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits data based on IQR of the Target.
        Returns: (df_normal, df_outliers)
        """
        target = self.cfg.TARGET
        if target not in df.columns:
            raise ValueError("Cannot split outliers: Target column missing.")

        # Calculate IQR
        Q1 = df[target].quantile(0.25)
        Q3 = df[target].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds
        upper_limit = Q3 + 1.5 * IQR
        lower_limit = Q1 - 1.5 * IQR
        
        # Create mask
        mask_outlier = (df[target] > upper_limit) | (df[target] < lower_limit)
        
        df_outliers = df[mask_outlier].copy()
        df_normal = df[~mask_outlier].copy()
        
        print(f"  >> Outlier Split Strategy: IQR ({lower_limit:.2f} to {upper_limit:.2f})")
        print(f"  >> Normal Samples: {len(df_normal)} | Outlier Samples: {len(df_outliers)}")
        
        return df_normal, df_outliers
    
    def tag_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Splits data based on IQR of the Target.
        Returns: (df_normal, df_outliers)
        """
        target = self.cfg.TARGET
        if target not in df.columns:
            raise ValueError("Cannot split outliers: Target column missing.")

        # Calculate IQR
        Q1 = df[target].quantile(0.25)
        Q3 = df[target].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds
        upper_limit = Q3 + 1.5 * IQR
        lower_limit = Q1 - 1.5 * IQR
        
        # Create mask
        mask_outlier = (df[target] > upper_limit) | (df[target] < lower_limit)
        
        df['is_outlier'] = mask_outlier.astype(int)

        #drop spell_episode_los
        df = df.drop(columns= self.cfg.TARGET)

        return df
    
    def fit_processors(self, df_train: pd.DataFrame,task:str="prediction"):
        """Fits all scalers and encoders on the TRAINING set."""
        # 1. Fit Scalers
        if self.cfg.STD_SCALE_COLS:
            self.std_scaler.fit(df_train[self.cfg.STD_SCALE_COLS])
        if self.cfg.ROBUST_SCALE_COLS:
            self.rob_scaler.fit(df_train[self.cfg.ROBUST_SCALE_COLS])
        
        if task == "classification":
            # 2. Fit Target Encoder
            X_encode = df_train[self.cfg.ENCODE_COLS]
            y_encode = df_train[self.cfg.CLASS_TARGET]
            self.encoder.fit(X_encode, y_encode)
        else:
            # 2. Normal Target Encoder fit
            X_encode = df_train[self.cfg.ENCODE_COLS]
            y_encode = df_train[self.cfg.TARGET]
            self.encoder.fit(X_encode, y_encode)

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies learned scalers and encoders to a dataset."""
        df = df.copy()
        
        # 1. Apply Scaling
        if self.cfg.STD_SCALE_COLS:
            df[self.cfg.STD_SCALE_COLS] = self.std_scaler.transform(df[self.cfg.STD_SCALE_COLS])
        if self.cfg.ROBUST_SCALE_COLS:
            df[self.cfg.ROBUST_SCALE_COLS] = self.rob_scaler.transform(df[self.cfg.ROBUST_SCALE_COLS])

        # 2. Apply Target Encoding
        df[self.cfg.ENCODE_COLS] = self.encoder.transform(df[self.cfg.ENCODE_COLS])
        
        return df

    def transform_target_log(self, df: pd.DataFrame) -> pd.DataFrame:
        """Log-transforms the target variable."""
        if self.cfg.TARGET in df.columns:
            df[self.cfg.TARGET] = np.log1p(df[self.cfg.TARGET]) 
        return df
    
    def save_pre_processing_transformers(self,save_path:str,encoder_name:str):
        with open(save_path, 'wb') as f:
            pickle.dump({
                'standard_scaler': self.std_scaler,
                'robust_scaler': self.rob_scaler,
                f'{encoder_name}': self.encoder
            }, f)