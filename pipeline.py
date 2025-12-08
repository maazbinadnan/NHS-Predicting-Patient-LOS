import pandas as pd
from sklearn.model_selection import train_test_split
import os
from preprocessing import MedicalDataPreprocessor, Config
from models.LinearRegression import TrainLinearRegressor
from models.RandomForest import TrainRandomForestRegressor
from models.XGboost import TrainXGBoostRegressor


class PipelineManager:
    def __init__(self) -> None:
        self.model = None
    
    @staticmethod
    def process_branch(df: pd.DataFrame, branch_name: str):
        """
        Helper function to run the pipeline on a specific subset of data (Normal or Outlier).
        """
        print(f"\n--- Processing {branch_name} Data Branch ---")
        
        processor = MedicalDataPreprocessor(config=Config)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        print(f"[{branch_name}] Fitting Scalers & Encoders on {len(train_df)} samples...")
        processor.fit_processors(train_df, task=branch_name.lower())

        train_processed = processor.transform_data(train_df)
        test_processed = processor.transform_data(test_df)
        if branch_name.lower() == "classification":
            pass
        else:
            train_processed = processor.transform_target_log(train_processed)

        print(f"[{branch_name}] Ready. X_train shape: {train_processed.shape}")
        
        return train_processed, test_processed

    @staticmethod
    def run_pipeline(file_path: str):
        print("--- 1. Loading Global Data ---")
        df = pd.read_csv(file_path) 
        
        global_processor = MedicalDataPreprocessor(config=Config)

        print("--- 2. Global Cleaning & Imputation ---")
        df = global_processor.initial_cleaning(df)
        df = global_processor.fill_na_values(df)

        print("--- 3. Splitting Outliers (The Fork) ---")
        df_normal, df_outliers = global_processor.split_outliers(df)

        train_inlier, test_norm = PipelineManager.process_branch(df_normal, "NORMAL")
        train_out, test_out = PipelineManager.process_branch(df_outliers, "OUTLIER")
        
        return train_inlier, test_norm, train_out, test_out

    @staticmethod
    def run_classification_pipeline(file_path: str):
        print("--- 1. Loading Global Data ---")
        df = pd.read_csv(file_path) 
        
        global_processor = MedicalDataPreprocessor(config=Config)

        print("--- 2. Global Cleaning & Imputation ---")
        df = global_processor.initial_cleaning(df)
        df = global_processor.fill_na_values(df)

        print("--- 3. Tagging Outliers ---")
        df = global_processor.tag_outliers(df=df)

        train_class, test_class = PipelineManager.process_branch(df, "Classification")

        return train_class, test_class

    @staticmethod
    def create_datasets(f_path, outlier_dir, class_dir):
        try:
            train_inlier, test_n, train_o, test_o = PipelineManager.run_pipeline(f_path)
            train_class, test_class = PipelineManager.run_classification_pipeline(f_path)

            train_inlier.to_csv(os.path.join(outlier_dir, "train_inlier.csv"), index=False)
            test_n.to_csv(os.path.join(outlier_dir, "test_inlier.csv"), index=False)
            train_o.to_csv(os.path.join(outlier_dir, "train_outlier.csv"), index=False)
            test_o.to_csv(os.path.join(outlier_dir, "test_outlier.csv"), index=False)
            
            train_class.to_csv(os.path.join(class_dir, "train_class.csv"), index=False)
            test_class.to_csv(os.path.join(class_dir, "test_class.csv"), index=False)
            
            print("\nPipeline finished successfully for outlier/inlier split.")
            print(f"Files saved in '{outlier_dir}'")
            print(f"Normal Train Set: {train_inlier.shape}")
            print(f"Outlier Train Set: {train_o.shape}")

            print("\nPipeline finished successfully for classification split.")
            print(f"Files saved in '{class_dir}'")
            print(f"Classification Train Set: {train_class.shape}")
            
        except FileNotFoundError:
            print(f"Error: File '{f_path}' not found. Please ensure the data file is in the directory.")
    
    def fine_tune_regression_models(self,train_df:pd.DataFrame,method:str,output_dir:str,filename:str): 
        x_train = train_df.drop(columns=Config().TARGET)
        y_train = train_df[Config().TARGET]

        print(train_df.shape)
        print(x_train.shape)
        print(y_train.shape)

        if method == "Linear Regression":
            self.model = TrainLinearRegressor()
            self.model.tune_linear_regression(X_train=x_train,y_train=y_train,output_dir=output_dir,filename=filename)
            self.model.tune_linear_regression(X_train=x_train,y_train=y_train,model_type= 'lasso',output_dir=output_dir,filename=filename)
        elif method == "Random Forest":
            self.model = TrainRandomForestRegressor()
        elif method == "XGboost":
            self.model = TrainXGBoostRegressor()
        else:
            ValueError("please enter correct model name")

if __name__ == "__main__":
    # Update with your actual file path
    f_path = "wwlLancMsc_data\\wwlLancMsc_data.csv"
    
    # Directories to save output CSVs
    outlier_dir = "Inlier_Outlier_Split"
    class_dir = "Classification_Layer_Data"
    fine_tune_dir = "fine_tuning_results"

    os.makedirs(outlier_dir, exist_ok=True)
    os.makedirs(class_dir, exist_ok=True)
    os.makedirs(fine_tune_dir, exist_ok=True)
    
    pipeline = PipelineManager()
    pipeline.create_datasets(f_path=f_path,outlier_dir=outlier_dir,class_dir=class_dir)
    # outlier_train_df = pd.read_csv(os.path.join(outlier_dir,"train_outlier.csv"))
    # inlier_train_df = pd.read_csv(os.path.join(outlier_dir,"train_inlier.csv"))

    # pipeline.fine_tune_regression_models(train_df=inlier_train_df,method="Linear Regression",output_dir=fine_tune_dir,filename="inlier.csv")
    # pipeline.fine_tune_regression_models(train_df=outlier_train_df,method="Linear Regression",output_dir=fine_tune_dir,filename="outlier.csv")
    








    
