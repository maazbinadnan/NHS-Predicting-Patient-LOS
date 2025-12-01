import pandas as pd
import os
import numpy as np
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import TargetEncoder

# -------------------------------
# 1. Configuration & Categories
# -------------------------------

categories = {
    "Patient Demographics": [
        'ethnic_origin_description', 'patient_age_on_admission', 'patient_age_on_discharge',
        'sex_national_code', 'general_medical_practice_desc', 'Deprivation Decile'
    ],
    "Hospital Demographics": [
        'site_national_code', 'specialty_spec_code', 
        'ward_code_admission', 'ward_code_discharge', # <--- KEPT THESE
        'specialty_division', 'specialty_directorate', 'hrg_group', 'hrg_sub_group',
        'ward_type_admission', 'ward_type_discharge', 'location', 'IP_admission', 'IP_discharge',
        'arrival_mode_description', 'source_of_ref_description', 'place_of_incident',
        'attendancetype' # <--- ADDED THIS
    ],
    "Clinical & Medical": [
        'acuity_code', 'attend_dis_description', 'comorbidity_score', 'frailty_score',
        'inj_or_ail', 'is_NEWS2_flag', 'NEWS2', 'presenting_complaint',
        'spell_primary_diagnosis', 'spell_secondary_diagnosis'
    ],
    "Target": "spell_episode_los",
    
    "Target_Encode_Cols": [
        'general_medical_practice_desc', 
        'ethnic_origin_description',
        'specialty_spec_code', 'specialty_division', 'specialty_directorate', 'hrg_group', 'hrg_sub_group',
        'location', 'IP_admission', 'IP_discharge',
        'ward_code_admission',
        'arrival_mode_description', 'source_of_ref_description', 'place_of_incident','attendancetype',
        'presenting_complaint', 'inj_or_ail', 'attend_dis_description', 
        'spell_primary_diagnosis', 'spell_secondary_diagnosis','spell_dominant_proc'
    ],
    
    "To_Drop_Early": [
        "site_description", "site_local_code", "specialty_local_code", "specialty_spec_desc",
        "ward_name_admission", "ward_name_discharge", "date_of_birth_dt", "date_of_death_dt",
        "discharge_delay_reason_national_code", "social_worker_date_time_referred",
        "discharge_created_datetime_dt", "spell_primary_diagnosis_description",
        "covid19_diagnosis_description", "ID", 
        "discharge_letter_sent_in_24hrs", "discharge_letter_status", "discharge_letter_sent", 
        "sex_description.y", "spell_dominant_proc_description",
        'Admission_Date', 'admission_date_dt', 'discharge_date_dt', 
        'Arrival_Date', 'arrival_date_time', 'initial_assessment_date_time'
    ],
    
    "Leakage_Columns": [
        'spell_los_hrs', 'spell_days_elective', 'spell_days_non_elective', 
        'delayed_discharges_no_of_days', 'delayed_discharges_flag', 
        'discharge_delay_reason_description'
    ]
}

# -------------------------------
# 2. Helper Functions
# -------------------------------

def initial_cleaning(df):
    """Drops ID/Text columns and Leakage columns early."""
    # Combine the drop lists
    cols_to_drop = categories['To_Drop_Early'] + categories['Leakage_Columns']
    # Only drop if they exist
    existing_cols = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_cols)
    return df

def fill_na_values(df):
    """Fills categorical and numerical NAs."""
    # Categorical Defaults
    defaults = {
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
    df = df.fillna(defaults)
    return df

def target_transformation(df):
    """Log transform the target."""
    if categories['Target'] in df.columns:
        df[categories['Target']] = np.log1p(df[categories['Target']])
    return df

def plot_histogram(df:pd.DataFrame,cat_to_draw):
    if categories[cat_to_draw] in df.columns:
        sns.histplot(df[categories[cat_to_draw]])
        plt.show()

def split_outliers(df):
    """
    Splits data based on IQR. 
    Assumes 'spell_episode_los' is ALREADY log-transformed.
    """
    target = categories['Target']
    
    Q1 = df[target].quantile(0.25)
    Q3 = df[target].quantile(0.75)
    IQR = Q3 - Q1
    
    upper_limit = Q3 + 1.5 * IQR
    lower_limit = Q1 - 1.5 * IQR # Usually irrelevant for LOS as log(LOS) > 0
    
    mask_outlier = (df[target] > upper_limit) | (df[target] < lower_limit)
    
    df_outliers = df[mask_outlier].copy()
    df_normal = df[~mask_outlier].copy()
    
    print(f"Split Result: Normal={len(df_normal)}, Outliers={len(df_outliers)}")
    return df_normal, df_outliers

def target_encoding(df:pd.DataFrame) -> tuple[pd.DataFrame,TargetEncoder]:
    df_encoded = df.copy() 
    y= df_encoded[categories['Target']]
    print(f"encoding columns: {categories['Target_Encode_Cols']}")

    enc = TargetEncoder(target_type='continuous', smooth='auto', random_state=42)

    transformed_data = enc.fit_transform(df_encoded[categories['Target_Encode_Cols']], y)

    df_encoded[categories['Target_Encode_Cols']] = transformed_data
    return df_encoded,enc

# -------------------------------
# 3. Execution Pipeline
# -------------------------------

# Load Data
cwd = os.getcwd()
try:
    # Adjust path as needed for your environment
    LancData = pd.read_csv(f"wwlLancMsc_data\\wwlLancMsc_data.csv") # Simplified path for example
except:
    LancData = pd.DataFrame() # Fallback

# Step 1: Clean and Fill (Before splitting, to ensure consistency)
df = initial_cleaning(LancData)
df = fill_na_values(df)

#step 2: split dataset into outliers and normal
df_normal,df_outliers = split_outliers(df=df)

#step 3: target encode each
df_normal_encoded,encoder_normal = target_encoding(df=df_normal)


df_outliers_encoded,encoder_normal = target_encoding(df=df_outliers)

df_normal_encoded.to_csv("Normal_Data.csv",index=False)

df_outliers_encoded.to_csv("Outlier_Data.csv",index=False)

