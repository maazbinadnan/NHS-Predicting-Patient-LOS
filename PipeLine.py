from Preprocessing import MedicalDataPreprocessor
from Regression import ModelTrainer
import pandas as pd
import pickle
import os

def run_training_pipeline(raw_df: pd.DataFrame):
    """
    Orchestrates the cleaning, splitting, and encoding for TRAINING.
    """
    processor = MedicalDataPreprocessor()

    # 1. Clean and Fill
    print("Step 1: Initial Cleaning...")
    df = processor.initial_cleaning(raw_df)
    df = processor.fill_na_values(df)

    # 2. Target Transform (Log)
    print("Step 2: Log Transforming Target...")
    df = processor.transform_target(df)

    # 3. Scale (Fit on full training set first to establish scale)
    print("Step 3: Scaling Features...")
    processor.fit_scaling(df)
    df = processor.transform_scaling(df)

    # 4. Outlier Split (Strategy: Two Models)
    print("Step 4: Splitting Outliers...")
    df_normal, df_outliers = processor.split_outliers(df)

    # 5. Target Encode (Separate encoders for Normal vs Outlier distributions)
    print("Step 5: Target Encoding...")
    df_normal_encoded = processor.handle_target_encoding(df_normal, encoder_key="inlier", train_mode=True)
    df_outliers_encoded = processor.handle_target_encoding(df_outliers, encoder_key="outlier", train_mode=True)
    
    # Save Encoders (Simulation)
    pickle.dump(processor.encoders['inlier'], open("inlier_encoder.pkl", "wb"))
    pickle.dump(processor.encoders['outlier'], open("outlier_encoder.pkl", "wb"))

    return df_normal_encoded, df_outliers_encoded, processor


def run_training_pipeline_single(raw_df: pd.DataFrame):
    """
    Orchestrates the cleaning, splitting, and encoding for TRAINING.
    """
    processor = MedicalDataPreprocessor()

    # 1. Clean and Fill
    print("Step 1: Initial Cleaning...")
    df = processor.initial_cleaning(raw_df)
    df = processor.fill_na_values(df)

    # 2. Target Transform (Log)
    print("Step 2: Log Transforming Target...")
    df = processor.transform_target(df)

    # 3. Scale (Fit on full training set first to establish scale)
    print("Step 3: Scaling Features...")
    processor.fit_scaling(df)
    df = processor.transform_scaling(df)

    # 5. Target Encode (Separate encoders for Normal vs Outlier distributions)
    print("Step 5: Target Encoding...")
    df_final_clean = processor.handle_target_encoding(df, encoder_key="inlier", train_mode=True)
   
    # Save Encoders (Simulation)
    #pickle.dump(processor.encoders['inlier'], open("inlier_encoder.pkl", "wb"))

    return df_final_clean
 



def run_model_trainer(df_normal_encoded, df_outliers_encoded):
    MODEL_DIR = os.path.join(os.getcwd(), "saved_models")
    TARGET = 'spell_episode_los'

    model_trainer = ModelTrainer(save_dir=MODEL_DIR)

    results_normal = model_trainer.train_and_save(df=df_normal_encoded,target_col=TARGET,experiment_name="Inliers")
    results_outliers = model_trainer.train_and_save(df=df_outliers_encoded,target_col=TARGET,experiment_name="Outliers")

    return results_normal,results_outliers

if __name__ == "__main__":
    file =  pd.read_csv(os.path.join(os.getcwd(),"wwlLancMsc_data","wwlLancMsc_data.csv"))
    df_inliers,df_outliers,processor = run_training_pipeline(raw_df=file)
    results_inlier,results_outlier = run_model_trainer(df_normal_encoded=df_inliers,df_outliers_encoded=df_outliers)
    results_inlier.to_markdown("inlier_results.md")
    results_outlier.to_markdown("outlier_results.md")
