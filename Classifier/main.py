'''

Main Class for Data Handling and Model Training


'''
import os
import pandas as pd
from pipeline import preprocessing_pipeline


def run_preprocessing(save_path: str,
                      file_path: str,
                      is_classification: bool):
    """
    Load data from file_path, run preprocessing, and optionally save results to save_path.
    Returns (inlier_df, outlier_df).
    """
    pre = preprocessing_pipeline(filepath=file_path)

    pre.cleaning()
    pre.fill_empty()
    pre.outlier_detection()

    if is_classification:
        inliers_data,outliers_data = pre.get_dataframes()
        inliers_data.to_csv(save_path)
        outliers_data.to_csv(save_path)
    else:
        entire_df= pre.get_dataframes()
        entire_df.to_csv(save_path)


'''
running the pipeline

'''
if __name__ == "__main__":
    
    # code to run when this file is executed directly
    is_classification = True
    save_path = os.path.join(os.getcwd(),"pre_processed_data","classification_data.csv")
    filepath = "wwlLancMsc_data\\wwlLancMsc_data.csv"
    run_preprocessing(is_classification=is_classification,save_path=save_path,file_path=filepath)