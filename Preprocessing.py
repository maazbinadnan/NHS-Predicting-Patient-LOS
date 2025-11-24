import pandas as pd
import os
import numpy as np
import re
from sklearn.model_selection import train_test_split

# -------------------------------
# 1. Configuration & Categories
# -------------------------------
categories = {
    "Patient Demographics": [
        'ethnic_origin_description', 'patient_age_on_admission', 'patient_age_on_discharge',
        'sex_national_code', 'general_medical_practice_desc', 'Deprivation Decile'
    ],
    "Hospital Demographics": [
        'site_national_code', 'specialty_spec_code', 'ward_code_admission', 'ward_code_discharge',
        'specialty_division', 'specialty_directorate', 'hrg_group', 'hrg_sub_group',
        'ward_type_admission', 'ward_type_discharge', 'location', 'IP_admission', 'IP_discharge',
        'arrival_mode_description', 'source_of_ref_description', 'place_of_incident'
    ],
    "Clinical & Medical": [
        'acuity_code', 'attend_dis_description', 'comorbidity_score', 'frailty_score',
        'inj_or_ail', 'is_NEWS2_flag', 'NEWS2', 'presenting_complaint',
        'spell_primary_diagnosis', 'spell_secondary_diagnosis'
        # Note: 'spell_dominant_proc' excluded based on your previous code, 
        # but usually highly predictive.
    ],
    "Target": "spell_episode_los",
    # Columns to drop immediately (IDs, Descriptions, Dates if not using feature eng)
    "To_Drop_Early": [
        "site_description", "site_local_code", "specialty_local_code", "specialty_spec_desc",
        "ward_name_admission", "ward_name_discharge", "date_of_birth_dt", "date_of_death_dt",
        "discharge_delay_reason_national_code", "social_worker_date_time_referred",
        "discharge_created_datetime_dt", "spell_primary_diagnosis_description",
        "covid19_diagnosis_description", "ID", 
        "discharge_letter_sent_in_24hrs", "discharge_letter_status", "discharge_letter_sent", 
        "sex_description.y", "spell_dominant_proc_description",
        # Dropping date columns as per your original script logic
        'Admission_Date', 'admission_date_dt', 'discharge_date_dt', 
        'Arrival_Date', 'arrival_date_time', 'initial_assessment_date_time'
    ],
    # Columns that constitute Data Leakage (proxies for LOS)
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

# --- Transformation Functions (Updated for Safety) ---

def patient_demographics_transform(df, is_test=False, mappings=None):
    cols = ['general_medical_practice_desc', 'ethnic_origin_description']
    # Filter for cols that actually exist in this dataframe
    cols = [c for c in cols if c in df.columns]
    
    target = categories['Target']
    maps = mappings if is_test else {}
    
    for col in cols:
        if not is_test:
            # Learn the mapping (Median Encoding)
            col_map = df.groupby(col)[target].median()
            maps[col] = col_map
        else:
            col_map = maps.get(col)
        
        # Apply Mapping
        if col_map is not None:
            # .map returns NaN for unseen categories, fill with global median (or 0)
            # Using transform median or a default safe value
            df[f'{col}_encoded'] = df[col].map(col_map).fillna(df[target].median() if not is_test else 0)
            df = df.drop(columns=[col])
            
    return df, maps

def hospital_demographics_transformed(df, is_test=False, mappings=None):
    cols = categories['Hospital Demographics']
    # Filter for cols that exist
    cols = [c for c in cols if c in df.columns]
    
    target = categories['Target']
    maps = mappings if is_test else {}
    
    for col in cols:
        if not is_test:
            col_map = df.groupby(col)[target].median()
            maps[col] = col_map
        else:
            col_map = maps.get(col)
            
        if col_map is not None:
            df[f'{col}_encoded'] = df[col].map(col_map).fillna(df[target].median() if not is_test else 0)
            df = df.drop(columns=[col])
            
    return df, maps

def clinical_medical_transform(df, is_test=False, mappings=None):
    # Pre-processing (String manipulation)
    if 'spell_primary_diagnosis' in df.columns:
        df['spell_primary_diagnosis'] = df['spell_primary_diagnosis'].astype(str).str[0]
    if 'spell_secondary_diagnosis' in df.columns:
        df['spell_secondary_diagnosis'] = df['spell_secondary_diagnosis'].astype(str).str[0]
    if 'frailty_score' in df.columns:
        df['frailty_score'] = df['frailty_score'].astype(str).str.extract(r'^(\d+)').astype(float)
    if 'presenting_complaint' in df.columns:
         df['presenting_complaint'] = df['presenting_complaint'].astype(str).apply(
            lambda x: re.findall(r'\((.*?)\)', x)[0] if '(' in x else x
        )

    cols_to_encode = ['presenting_complaint', 'inj_or_ail', 'attend_dis_description', 
                      'spell_primary_diagnosis', 'spell_secondary_diagnosis']
    
    # Filter existing
    cols_to_encode = [c for c in cols_to_encode if c in df.columns]
    
    target = categories['Target']
    maps = mappings if is_test else {}

    for col in cols_to_encode:
        if not is_test:
            col_map = df.groupby(col)[target].median()
            maps[col] = col_map
        else:
            col_map = maps.get(col)
        
        if col_map is not None:
             df[col] = df[col].map(col_map).fillna(df[target].median() if not is_test else 0)
    
    return df, maps

# -------------------------------
# 3. Execution Pipeline
# -------------------------------

# Load Data
cwd = os.getcwd()
try:
    # Adjust path as needed for your environment
    LancData = pd.read_csv(f"NHS_Data_Final_Cleaned.csv") # Simplified path for example
except:
    LancData = pd.DataFrame() # Fallback

# Step 1: Clean and Fill (Before splitting, to ensure consistency)
df = initial_cleaning(LancData)
df = fill_na_values(df)

# Step 2: Target Transformation
df = target_transformation(df)

# Step 3: Split Outliers (On the transformed target)
df_normal, df_outliers = split_outliers(df)

# --- PROCESSING NORMAL DATA ---
print("\nProcessing Normal Data...")
# Split into Train/Test Dataframes (keeping Target included for now)
# We do this because Target Encoding needs the target in the Training set.
df_train, df_test = train_test_split(df_normal, test_size=0.2, random_state=42)

# Apply Transformations (Learn on Train, Apply to Test)
df_train, map_pat = patient_demographics_transform(df_train, is_test=False)
df_test, _        = patient_demographics_transform(df_test, is_test=True, mappings=map_pat)

df_train, map_hosp = hospital_demographics_transformed(df_train, is_test=False)
df_test, _         = hospital_demographics_transformed(df_test, is_test=True, mappings=map_hosp)

df_train, map_clin = clinical_medical_transform(df_train, is_test=False)
df_test, _         = clinical_medical_transform(df_test, is_test=True, mappings=map_clin)

# FINAL STEP: Separate X and y (and drop target from X)
# This prevents Data Leakage
X_normal_train = df_train.drop(columns=[categories['Target']])
y_normal_train = df_train[categories['Target']]

X_normal_test = df_test.drop(columns=[categories['Target']])
y_normal_test = df_test[categories['Target']]

print(f"Normal Train Shape: {X_normal_train.shape}")
print(f"Normal Test Shape: {X_normal_test.shape}")

# --- PROCESSING OUTLIER DATA (Repeat logic) ---
print("\nProcessing Outlier Data...")
df_out_train, df_out_test = train_test_split(df_outliers, test_size=0.2, random_state=42)

# Reuse the functions (Learns new maps specific to outliers)
df_out_train, map_pat_out = patient_demographics_transform(df_out_train, is_test=False)
df_out_test, _            = patient_demographics_transform(df_out_test, is_test=True, mappings=map_pat_out)

df_out_train, map_hosp_out = hospital_demographics_transformed(df_out_train, is_test=False)
df_out_test, _             = hospital_demographics_transformed(df_out_test, is_test=True, mappings=map_hosp_out)

df_out_train, map_clin_out = clinical_medical_transform(df_out_train, is_test=False)
df_out_test, _             = clinical_medical_transform(df_out_test, is_test=True, mappings=map_clin_out)

X_outliers_train = df_out_train.drop(columns=[categories['Target']])
y_outliers_train = df_out_train[categories['Target']]

X_outliers_test = df_out_test.drop(columns=[categories['Target']])
y_outliers_test = df_out_test[categories['Target']]

print(f"Outlier Train Shape: {X_outliers_train.shape}")
print(f"Outlier Test Shape: {X_outliers_test.shape}")


# Save Normal Data
X_normal_train.to_csv("X_normal_train.csv", index=False)
y_normal_train.to_csv("y_normal_train.csv", index=False)
X_normal_test.to_csv("X_normal_test.csv", index=False)
y_normal_test.to_csv("y_normal_test.csv", index=False)

# Save Outlier Data
X_outliers_train.to_csv("X_outliers_train.csv", index=False)
y_outliers_train.to_csv("y_outliers_train.csv", index=False)
X_outliers_test.to_csv("X_outliers_test.csv", index=False)
y_outliers_test.to_csv("y_outliers_test.csv", index=False)

print("All files saved successfully.")