import pandas as pd

LancData = pd.read_csv("wwlLancMsc_data.csv")
# print(LancData.shape)

LancData_clone = LancData.copy()
print(f"Shape of the clone then: {LancData_clone.shape}")

### Columns to DROP
dropping = ["site_description", "site_local_code", "specialty_local_code",
            "specialty_spec_desc", "ward_name_admission", "ward_name_discharge", "date_of_birth_dt",
            "discharge_delay_reason_national_code", "social_worker_date_time_referred", "discharge_created_datetime_dt", "spell_primary_diagnosis_description", 
            "covid19_diagnosis_description", "ID"]


for i in dropping:
    LancData_clone = LancData_clone.drop(i, axis=1)
print(f"Shape of the clone now: {LancData_clone.shape}")