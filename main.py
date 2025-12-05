'''

Main Class for Data Handling and Model Training


'''
import os
import pandas as pd
from pipeline import PreprocessingPipeline


def run_preprocessing(save_path: str,
                      file_path: str,
                      is_classification: bool):
    """
    Load data, preprocess, and save.
    Handles file naming automatically for split datasets.
    """
    # Initialize pipeline
    pre = PreprocessingPipeline(filepath=file_path, class_bool=is_classification)

    # Run steps
    pre.cleaning()
    pre.fill_empty()
    pre.outlier_detection()

    # Create directory if it doesn't exist (Professional touch)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if is_classification:
        entire_df = pd.DataFrame(pre.get_dataframes())
        # Save single file
        entire_df.to_csv(save_path, index=False)
        print(f"✔ Classification data saved to: {save_path}")
    else:  
        # Unpack the two dataframes
        inliers_data, outliers_data = pre.get_dataframes()
        
        # ---------------------------------------------------------
        # FIX: Modify filenames so they don't overwrite each other
        # ---------------------------------------------------------
        # Split 'data/output.csv' into 'data/output' and '.csv'
        base_name, ext = os.path.splitext(save_path)
        
        path_inliers = f"{base_name}_inliers{ext}"
        path_outliers = f"{base_name}_outliers{ext}"
        
        # Save to separate paths
        inliers_data.to_csv(path_inliers, index=False)
        outliers_data.to_csv(path_outliers, index=False)
        
        print(f"✔ Normal data saved to: {path_inliers}")
        print(f"✔ Outlier data saved to: {path_outliers}")

# Example Usage:
# run_preprocessing("processed_data/hospital_data.csv", "raw_data.csv", is_classification=False)
# Result: Creates 'processed_data/hospital_data_inliers.csv' and 'processed_data/hospital_data_outliers.csv'


'''
running the pipeline

'''
import os

if __name__ == "__main__":

    # boolean flag
    is_classification = True

    # path to save preprocessed data
    save_path = os.path.join(os.getcwd(), "classifier", "pre_processed_data", "classification_data.csv")
    
    # ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # path to raw data
    filepath = "wwlLancMsc_data\\wwlLancMsc_data.csv"

    # call your preprocessing function
    run_preprocessing(is_classification=is_classification, save_path=save_path, file_path=filepath)
