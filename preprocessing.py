import pandas as pd
import numpy as np
from sklearn.preprocessing import TargetEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from typing import Tuple

# -------------------------------
# 1. Configuration Class
# -------------------------------
class Config:
    TARGET = "spell_episode_los"
    
    # Columns to drop (IDs, Dates, Leakage)
    DROP_COLS = [
        "site_description", "site_local_code", "specialty_local_code", "specialty_spec_desc",
        "ward_name_admission", "ward_name_discharge", "ward_code_discharge", "ward_type_discharge", # Leakage
        "date_of_birth_dt", "date_of_death_dt", "discharge_created_datetime_dt",
        "discharge_delay_reason_national_code", "social_worker_date_time_referred",
        "spell_primary_diagnosis_description", "covid19_diagnosis_description", "ID",
        "discharge_letter_sent_in_24hrs", "discharge_letter_status", "discharge_letter_sent",
        "sex_description.y", "spell_dominant_proc_description",
        'Admission_Date', 'admission_date_dt', 'discharge_date_dt',
        'Arrival_Date', 'arrival_date_time', 'initial_assessment_date_time',
        'medically_optimised', 'patient_age_on_discharge', 'IP_discharge',
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
    ROBUST_SCALE_COLS = ["duration_elective_wait","comorbidity_score"]

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
        
        df['is_outlier'] = mask_outlier.astype(bool)

        return df
    
    def fit_processors(self, df_train: pd.DataFrame):
        """Fits all scalers and encoders on the TRAINING set."""
        # 1. Fit Scalers
        if self.cfg.STD_SCALE_COLS:
            self.std_scaler.fit(df_train[self.cfg.STD_SCALE_COLS])
        if self.cfg.ROBUST_SCALE_COLS:
            self.rob_scaler.fit(df_train[self.cfg.ROBUST_SCALE_COLS])
        
        # 2. Fit Target Encoder
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