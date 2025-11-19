import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("wwlLancMsc_data/wwlLancMsc_data.csv")

df_encoded = df.copy()

# Columns to encode
columns_to_encode = [
    "spell_primary_diagnosis",
    "spell_secondary_diagnosis",
    "ward_type_admission",
    "ward_type_discharge"
]

# Handle missing values for all columns to be encoded
for col in columns_to_encode:
    df_encoded[col] = df_encoded[col].fillna("MISSING").astype(str)

# Create label encoders and encode
encoders = {}

for col in columns_to_encode:
    le = LabelEncoder()
    df_encoded[f"{col}_encoded"] = le.fit_transform(df_encoded[col])
    encoders[col] = le
    print(f"✓ {col} encoded: {df_encoded[col].nunique()} unique values → integers 0–{df_encoded[col].nunique() - 1}")


# Save encoded file
output_path = "wwlLancMsc_data/wwlLancMsc_data_encoded.csv"
df_encoded.to_csv(output_path, index=False)
