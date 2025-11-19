import pandas as pd
import os

cwd=os.getcwd()
LancData = pd.read_csv(f"{cwd}/wwlLancMsc_data/wwlLancMsc_data.csv")
# print(LancData.shape)

'''
Function to Drop Columns that aren't useful
'''

def DropColumns(dataframe: pd.DataFrame) -> pd.DataFrame:
    LancData_clone = LancData.copy()
    print(f"Shape of the clone then: {LancData_clone.shape}")

    ### Columns to DROP
    dropping = ["site_description", "site_local_code", "specialty_local_code",
                "specialty_spec_desc", "ward_name_admission", "ward_name_discharge", "date_of_birth_dt",
                "discharge_delay_reason_national_code", "social_worker_date_time_referred", "discharge_created_datetime_dt", "spell_primary_diagnosis_description", 
                "covid19_diagnosis_description", "ID","date_of_death_dt"]


    for i in dropping:
        LancData_clone = LancData_clone.drop(i, axis=1)
    print(f"Shape of the clone now: {LancData_clone.shape}")

    return LancData_clone



def FillNACategorical(df_no_missing: pd.DataFrame) -> pd.DataFrame:
    df_no_missing['place_of_incident'] = df_no_missing['place_of_incident'].fillna("Not Specified")
    df_no_missing['ward_code_discharge'] = df_no_missing['ward_code_discharge'].fillna("GAST")
    df_no_missing['general_medical_practice_desc'] = df_no_missing['general_medical_practice_desc'].fillna("Unknown")
    leakage_columns= ['discharge_letter_sent_in_24hrs',
    'discharge_letter_status',
    'discharge_letter_sent'
    ]
    df_no_missing.drop(leakage_columns,inplace=True,axis=1)
    df_no_missing['location'] = df_no_missing['location'].fillna("Unknown Location")
    df_no_missing = df_no_missing.drop(columns=['sex_description.y'])
    cols_to_fill_NA = [
    'inj_or_ail', 'attendancetype', 'arrival_mode_description', 
    'presenting_complaint', 'source_of_ref_description', 'attend_dis_description'
    ]
    df_no_missing[cols_to_fill_NA] = df_no_missing[cols_to_fill_NA].fillna('Not Applicable')
    return df_no_missing

def FillNANumerical(df_no_missing: pd.DataFrame) -> pd.DataFrame:
    df_no_missing['covid19_diagnosis_flag']=df_no_missing['covid19_diagnosis_flag'].fillna(0)
    df_no_missing['is_NEWS2_flag'] = df_no_missing['NEWS2'].notnull().astype(int)
    df_no_missing['NEWS2'] = df_no_missing['NEWS2'].fillna(-1)
    # Create the binary flag: 1 if the wait duration is known (elective), 0 if NULL (emergency)
    df_no_missing['is_elective_flag'] = df_no_missing['duration_elective_wait'].notnull().astype(int)
    #replace the values with -1 just in case
    df_no_missing['duration_elective_wait'] = df_no_missing['duration_elective_wait'].fillna(-1)
    # If Arrival Date is missing, assume they arrived at the moment of admission.
    df_no_missing['Arrival_Date'] = df_no_missing['Arrival_Date'].fillna(df_no_missing['Admission_Date'])
    # Do the same for the time if you plan to use it
    df_no_missing['arrival_date_time'] = df_no_missing['arrival_date_time'].fillna(df_no_missing['admission_date_dt'])
    # do the same for initial assessment
    df_no_missing['initial_assessment_date_time'] = df_no_missing['arrival_date_time'].fillna(df_no_missing['admission_date_dt'])
    df_no_missing['ae_unplanned_attendance'] = df_no_missing['ae_unplanned_attendance'].fillna(0)
    df_no_missing['acuity_code'] = df_no_missing['acuity_code'].fillna(-1)
    return df_no_missing



def nullcheck(df_no_missing):
    print(df_no_missing.isnull().sum()[df_no_missing.isnull().sum()>0].sort_values(ascending=False))
    print("number of columns with missing values:",len(df_no_missing.isnull().sum()[df_no_missing.isnull().sum()>0].sort_values(ascending=False)))


df= DropColumns(LancData)
print("null values before")
print(nullcheck(df))
df = FillNACategorical(df)
df_final = FillNANumerical(df)
print("null values after")
print(nullcheck(df_final))
print(f"final shape: {df_final.shape}")

