1. clean_data and remove leakage columns
2. remove na_values
3. if !is_classification then split on outliers else add column for is_outliers and remove the target col
4. train test split
5. scale columns
5. log transformation of target column
6. target encoding