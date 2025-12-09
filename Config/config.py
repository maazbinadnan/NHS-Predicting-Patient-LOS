# -------------------------------
# 1. Configuration Class
# -------------------------------
class Config:
    TARGET = "spell_episode_los"

    CLASS_TARGET ="is_outlier"
    
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
    STD_SCALE_COLS = ["patient_age_on_admission","Deprivation Decile"]
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