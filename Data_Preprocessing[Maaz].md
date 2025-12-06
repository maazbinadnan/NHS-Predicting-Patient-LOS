### Preprocessing Pipeline Summary

Our pipeline implements a **"Mixture of Experts"** strategy designed to handle medical Length of Stay (LOS) data, which typically contains extreme outliers (long-stay patients) that confuse standard models.

**1. Data Cleaning & Leakage Prevention**
* **Leakage Removal:** We explicitly drop columns that are only known *after* discharge (e.g., `discharge_date`, `total_cost`, `medically_optimised`), as using these would cheat the prediction.
* **Formatting:** We clean specific fields like extracting numeric values from the `frailty_score` string.

**2. Missing Data Strategy**
* **Imputation:** Instead of dropping rows with missing data (which would bias our model towards "perfect" records), we fill NAs with domain-specific defaults (e.g., missing `frailty_score` or `comorbidity` is assumed to be 0 or "Not Applicable").

**3. Outlier Handling (The Fork)**
* **Split Strategy:** We use the Interquartile Range (IQR) method to physically split the dataset into two branches: **Normal Patients** and **Outliers (Complex Cases)**.
* **Reasoning:** This allows us to train two separate, specialized models—one optimized for routine cases and another for complex, long-stay cases—rather than a single model that performs poorly on both.

**4. Feature Engineering**
* **Target Encoding:** Categorical variables (like `ward_code` or `specialty`) are converted to numbers based on the average LOS for that category.
* **Scaling:** We apply `StandardScaler` to Age and `RobustScaler` to Wait Times/Comorbidity Scores (since RobustScaler is resilient to the extreme values common in wait lists).
* **Log Transformation:** We apply `log1p` to the target variable (LOS). This compresses the range of days (e.g., turning 100 days into ~4.6), making the data distribution more normal and easier for the model to learn.

**5. Leakage Control**
* We strictly separate the **Learning Phase** (fitting scalers/encoders on Training data only) from the **Application Phase** (transforming Test data using those learned values). This ensures no information from the test set bleeds into the training process.