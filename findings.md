# Data Cleaning & Modeling Strategy: Patient Length of Stay

## 1. The "Location" Insight
The `location` column is the single most important proxy for patient acuity, despite being missing for ~75% of rows.

* **The Finding:**
    * **Missing Location** $\approx$ **0.6 days LOS** (Planned/Direct admissions).
    * **Majors (ED)** $\approx$ **7.0 days LOS** (Severe/Complex cases).
    * **ECC/Minors** $\approx$ **2.6 days LOS** (Less severe cases).
* **The Action:** Do not drop rows with missing locations. Fill them with **"Direct_Admission"**.

## 2. The "Admission Type" Hierarchy
Using the `location` column combined with `elective_admission_flag` and `non_elective_admission_flag`, we identified three distinct patient flows:

1.  **Emergency via ED:** (High Acuity)
    * *Criteria:* Has ED Location + Non-Elective Flag.
    * *Behavior:* Unpredictable and often long LOS.
2.  **Emergency Direct:** (Variable Acuity)
    * *Criteria:* **No** ED Location + Non-Elective Flag (e.g., GP Urgent Referral).
    * *Context:* These patients skip the queue but are still urgent/sick.
3.  **Planned/Elective:** (Low Acuity)
    * *Criteria:* **No** ED Location + Elective Flag (or no flag).
    * *Behavior:* Predictable LOS (often 0 or 1 day).

**Strategy:** Create a new feature called `Admission_Type` to capture these three groups explicitly.

## 3. Handling Missing Values
We identified that most missing values are "Structurally Missing" (they don't exist because the event didn't happen), not random errors.

| Column Type | Examples | Imputation Strategy | Logic |
| :--- | :--- | :--- | :--- |
| **Categorical** | `inj_or_ail`, `arrival_mode` | **"Not Applicable"** | The patient did not arrive via A&E. |
| **Numerical** | `acuity_code`, `NEWS2` | **-1** | Preserves the distribution of valid scores while creating a distinct "No Score" category. |
| **Flags** | `ae_unplanned_attendance` | **0** | If they didn't attend A&E, it wasn't an unplanned A&E attendance. |
| **Dates** | `Arrival_Date` | **= `Admission_Date`** | For direct admissions, Arrival Time is effectively the Admission Time. |

## 4. Modeling Recommendation: The Hurdle Model
Because the data is zero-inflated (median LOS is 0), a standard regression model will likely fail (predicting ~0.5 days for everyone).

* **Stage 1 (Classification):** Predict "Will LOS > 0?"
    * *Goal:* Separate Day Cases from Inpatients.
    * *Metric:* ROC-AUC.
* **Stage 2 (Regression):** Predict "Log(LOS)" for patients where LOS > 0.
    * *Goal:* Predict recovery time for sick patients.
    * *Metric:* RMSE / MAE.

## 5. Data Leakage Warning
The following columns were identified as **Leakage** (information known only *after* discharge) and must be dropped to prevent the model from "cheating":
* `discharge_letter_sent`
* `discharge_letter_sent_in_24hrs`
* `discharge_letter_status`


Basically divide our dataset into two, planned admissions and unplanned admissions and make different regression models for both. (untested theory but it is one)