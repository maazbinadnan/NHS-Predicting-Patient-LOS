import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import MedicalDataPreprocessor, Config

def process_branch(df: pd.DataFrame, branch_name: str):
    """
    Helper function to run the pipeline on a specific subset of data (Normal or Outlier).
    """
    print(f"\n--- Processing {branch_name} Data Branch ---")
    
    # 1. Initialize a FRESH processor for this specific branch
    # (Important: The 'Normal' encoder shouldn't learn from 'Outlier' data and vice versa)
    processor = MedicalDataPreprocessor(config=Config)

    # 2. Train/Test Split
    # Stratify is tricky for regression, but random split is usually fine here
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # 3. Fit Processors on TRAIN
    print(f"[{branch_name}] Fitting Scalers & Encoders on {len(train_df)} samples...")
    processor.fit_processors(train_df)

    # 4. Transform Data
    train_processed = processor.transform_data(train_df)
    test_processed = processor.transform_data(test_df)

    # 5. Log Transform Target
    # Log transform is still recommended even for split data to normalize residuals
    train_processed = processor.transform_target_log(train_processed)
    # Note: We usually keep Test Target RAW for easier interpretation (MAE in days), 
    # but if you want to compare Loss, you can log transform it too.
    # test_processed = processor.transform_target_log(test_processed) 

    print(f"[{branch_name}] Ready. X_train shape: {train_processed.shape}")
    
    return train_processed, test_processed

def run_pipeline(file_path: str):
    print("--- 1. Loading Global Data ---")
    df = pd.read_csv(file_path) 
    
    # Global Processor for initial cleaning only
    global_processor = MedicalDataPreprocessor(config=Config)

    print("--- 2. Global Cleaning & Imputation ---")
    df = global_processor.initial_cleaning(df)
    df = global_processor.fill_na_values(df)

    print("--- 3. Splitting Outliers (The Fork) ---")
    # This splits the dataset into two separate physical dataframes
    df_normal, df_outliers = global_processor.split_outliers(df)

    # --- Branch 1: Normal Model ---
    train_norm, test_norm = process_branch(df_normal, "NORMAL")

    # --- Branch 2: Outlier Model ---
    train_out, test_out = process_branch(df_outliers, "OUTLIER")
    
    # You now have 4 dataframes ready for training 2 different models
    return train_norm, test_norm, train_out, test_out

def run_classification_pipeline(file_path:str):
    print("--- 1. Loading Global Data ---")
    df = pd.read_csv(file_path) 
    
    # Global Processor for initial cleaning only
    global_processor = MedicalDataPreprocessor(config=Config)

    print("--- 2. Global Cleaning & Imputation ---")
    df = global_processor.initial_cleaning(df)
    df = global_processor.fill_na_values(df)

    print("--- 3. Tagging Outliers ---")
    df = global_processor.tag_outliers(df=df)

    # --- Branch 2: Classification Model ---
    train_class, test_class = process_branch(df, "Classification")

    return train_class,test_class


import os

if __name__ == "__main__":
    # Update with your actual file path
    f_path = "wwlLancMsc_data\\wwlLancMsc_data.csv"
    
    # Directories to save output CSVs
    outlier_dir = "output_outlier_split"
    class_dir = "output_classification_split"
    
    os.makedirs(outlier_dir, exist_ok=True)
    os.makedirs(class_dir, exist_ok=True)
    
    try:
        # Run pipelines
        train_n, test_n, train_o, test_o = run_pipeline(f_path)
        train_class, test_class = run_classification_pipeline(f_path)

        # --- Save Outlier/Inlier Split ---
        train_n.to_csv(os.path.join(outlier_dir, "train_inlier.csv"), index=False)
        test_n.to_csv(os.path.join(outlier_dir, "test_inlier.csv"), index=False)
        train_o.to_csv(os.path.join(outlier_dir, "train_outlier.csv"), index=False)
        test_o.to_csv(os.path.join(outlier_dir, "test_outlier.csv"), index=False)
        
        # --- Save Classification Split ---
        train_class.to_csv(os.path.join(class_dir, "train_class.csv"), index=False)
        test_class.to_csv(os.path.join(class_dir, "test_class.csv"), index=False)
        
        # --- Print Summary ---
        print("\nPipeline finished successfully for outlier/inlier split.")
        print(f"Files saved in '{outlier_dir}'")
        print(f"Normal Train Set: {train_n.shape}")
        print(f"Outlier Train Set: {train_o.shape}")

        print("\nPipeline finished successfully for classification split.")
        print(f"Files saved in '{class_dir}'")
        print(f"Classification Train Set: {train_class.shape}")
        
    except FileNotFoundError:
        print(f"Error: File '{f_path}' not found. Please ensure the data file is in the directory.")
