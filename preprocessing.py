import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import TargetEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict

# -------------------------------
# 1. Configuration Class
# -------------------------------
class Config:
    """Central configuration for column groups and pipeline settings."""
    TARGET = "spell_episode_los"
    
    # Columns to drop immediately (IDs, Dates, Leakage)
    DROP_COLS = [
        "site_description", "site_local_code", "specialty_local_code", "specialty_spec_desc",
        "ward_name_admission", "ward_name_discharge", "ward_code_discharge", "ward_type_discharge", # Leakage!
        "date_of_birth_dt", "date_of_death_dt", "discharge_created_datetime_dt",
        "discharge_delay_reason_national_code", "social_worker_date_time_referred",
        "spell_primary_diagnosis_description", "covid19_diagnosis_description", "ID",
        "discharge_letter_sent_in_24hrs", "discharge_letter_status", "discharge_letter_sent",
        "sex_description.y", "spell_dominant_proc_description",
        'Admission_Date', 'admission_date_dt', 'discharge_date_dt',
        'Arrival_Date', 'arrival_date_time', 'initial_assessment_date_time',
        'medically_optimised', 'patient_age_on_discharge', 'IP_discharge',
        # Leakage Columns explicitly identified
        'spell_los_hrs', 'spell_days_elective', 'spell_days_non_elective',
        'delayed_discharges_no_of_days', 'delayed_discharges_flag',
        'discharge_delay_reason_description'
    ]

    # Columns for Target Encoding
    ENCODE_COLS = [
        'general_medical_practice_desc', 'ethnic_origin_description',
        'specialty_spec_code', 'specialty_division', 'specialty_directorate',
        'hrg_group', 'hrg_sub_group', 'location', 'IP_admission',
        'ward_code_admission', 'arrival_mode_description', 'source_of_ref_description',
        'place_of_incident', 'attendancetype', 'presenting_complaint',
        'inj_or_ail', 'attend_dis_description', 'spell_primary_diagnosis',
        'spell_secondary_diagnosis', 'spell_dominant_proc', 'ward_type_admission','site_national_code'
    ]

    # Scaling Groups
    STD_SCALE_COLS = ["patient_age_on_admission"]
    ROBUST_SCALE_COLS = ["duration_elective_wait"]

    # Default values for NA imputation
    NA_DEFAULTS = {
        'place_of_incident': "Not Specified",
        'ward_code_discharge': "GAST",
        'general_medical_practice_desc': "Unknown",
        'location': "Unknown Location",
        'inj_or_ail': 'Not Applicable',
        'attendancetype': 'Not Applicable',
        'arrival_mode_description': 'Not Applicable',
        'presenting_complaint': 'Not Applicable',
        'source_of_ref_description': 'Not Applicable',
        'attend_dis_description': 'Not Applicable',
        'covid19_diagnosis_flag': 0,
        'NEWS2': -1,
        'duration_elective_wait': -1,
        'ae_unplanned_attendance': 0,
        'acuity_code': -1
    }
 
# -------------------------------
# 2. Medical Data Processor
# -------------------------------
class MedicalDataPreprocessor:
    def __init__(self, config=Config):
        self.cfg = config
        self.std_scaler = StandardScaler()
        self.rob_scaler = RobustScaler()
        self.encoders = {} 

    def initial_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes ID columns, leakage columns, and cleans specific formats."""
        # Drop columns if they exist
        drop_current = [c for c in self.cfg.DROP_COLS if c in df.columns]
        df = df.drop(columns=drop_current)

        # Fix Frailty Score (Extract number from string)
        if 'frailty_score' in df.columns:
            df['frailty_score'] = (
                df['frailty_score']
                .astype(str)
                .str.extract(r'(\d+)')
                .astype(float)
            )
        return df

    def fill_na_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imputes missing values using defined defaults."""
        return df.fillna(self.cfg.NA_DEFAULTS)

    def fit_scaling(self, df: pd.DataFrame):
        """Fits scalers on the training data."""
        if self.cfg.STD_SCALE_COLS:
            self.std_scaler.fit(df[self.cfg.STD_SCALE_COLS])
            
        if self.cfg.ROBUST_SCALE_COLS:
            self.rob_scaler.fit(df[self.cfg.ROBUST_SCALE_COLS])
            

    def transform_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies scaling to data."""
        df = df.copy()
        if self.cfg.STD_SCALE_COLS:
            scaled = self.std_scaler.transform(df[self.cfg.STD_SCALE_COLS])
            df[self.cfg.STD_SCALE_COLS] = scaled
            
        if self.cfg.ROBUST_SCALE_COLS:
            scaled = self.rob_scaler.transform(df[self.cfg.ROBUST_SCALE_COLS])
            df[self.cfg.ROBUST_SCALE_COLS] = scaled
        return df

    def transform_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Log-transforms the target variable for training."""
        if self.cfg.TARGET in df.columns:
            # log1p is safer for LOS which can be 0
            df[self.cfg.TARGET] = np.log1p(df[self.cfg.TARGET]) 
        return df

    def classify_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Splits data based on IQR of the Target.
        WARNING: Only usable during TRAINING. 
        """
        target = self.cfg.TARGET
        if target not in df.columns:
            raise ValueError("Cannot split outliers: Target column missing.")

        Q1 = df[target].quantile(0.25)
        Q3 = df[target].quantile(0.75)
        IQR = Q3 - Q1
        upper_limit = Q3 + 1.5 * IQR
        lower_limit = Q1 - 1.5 * IQR

        df["is_outlier"]  = (df[target] > upper_limit) | (df[target] < lower_limit)
        df = df.drop(columns=self.cfg.TARGET)
        return df

    def split_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits data based on IQR of the Target.
        WARNING: Only usable during TRAINING. 
        returns: Inliers and then Outliers
        """
        target = self.cfg.TARGET
        if target not in df.columns:
            raise ValueError("Cannot split outliers: Target column missing.")

        Q1 = df[target].quantile(0.25)
        Q3 = df[target].quantile(0.75)
        IQR = Q3 - Q1
        upper_limit = Q3 + 1.5 * IQR
        lower_limit = Q1 - 1.5 * IQR

        mask_outlier = (df[target] > upper_limit) | (df[target] < lower_limit)
        
        df_outliers = df[mask_outlier].copy()
        df_normal = df[~mask_outlier].copy()
        
        print(f"Split Statistics: Normal={len(df_normal)}, Outliers={len(df_outliers)}")
        return df_normal, df_outliers

    def handle_target_encoding(self, df: pd.DataFrame, encoder_key: str, train_mode: bool = True) -> pd.DataFrame:
        """
        Fits or applies Target Encoding.
        """
        df_encoded = df.copy()
        cols = self.cfg.ENCODE_COLS
        target = self.cfg.TARGET

        # Initialize encoder if training and not present
        if train_mode:
            enc = TargetEncoder(target_type='continuous', smooth='auto', random_state=42)
            # Fit and Transform
            transformed = enc.fit_transform(df_encoded[cols], df_encoded[target])
            self.encoders[encoder_key] = enc
        else:
            # Load existing encoder
            if encoder_key not in self.encoders:
                raise ValueError(f"Encoder {encoder_key} not trained yet.")
            enc = self.encoders[encoder_key]
            # Transform only
            transformed = enc.transform(df_encoded[cols])

        df_encoded[cols] = transformed
        return df_encoded


