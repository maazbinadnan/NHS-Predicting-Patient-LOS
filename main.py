import pandas as pd
from sklearn.model_selection import train_test_split
import os
from pipeline import PipelineManager
from models.LinearRegression import TrainLinearRegressor
from models.RandomForest import TrainRandomForestRegressor
from models.XGboost import TrainXGBoostRegressor
from models.IsolationForest import TrainIsolationForest


class PipelineRunner:
    def __init__(self, f_path: str, training: bool):
        self.f_path = f_path
        self.training = training
        self.outlier_dir = "Inlier_Outlier_Split"
        self.class_dir = "Classification_Layer_Data"
        self.no_split_dir = "No_Split_Data"
        self.fine_tune_dir = "fine_tuning_results"
        self.models_dir = "saved_models"
        self.pipeline = None

        self.inlier_train, self.inlier_test = pd.DataFrame(), pd.DataFrame()
        self.outlier_train, self.outlier_test = pd.DataFrame(), pd.DataFrame()
        self.nosplit_train, self.nosplit_test = pd.DataFrame(), pd.DataFrame()
        self.classification_train, self.classification_test = pd.DataFrame(), pd.DataFrame()

        os.makedirs(self.outlier_dir, exist_ok=True)
        os.makedirs(self.class_dir, exist_ok=True)
        os.makedirs(self.no_split_dir, exist_ok=True)
        os.makedirs(self.fine_tune_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

    #run if data isnt created already
    def pre_process_data(self):
        if not self.training:
            return
        self.pipeline = PipelineManager()
        self.pipeline.create_datasets(
            f_path=self.f_path,
            outlier_dir=self.outlier_dir,
            class_dir=self.class_dir,
            no_split_dir=self.no_split_dir
        )

    #run if u wanna run the no_split_pipeline
    def load_no_split_data(self):
        train_path = os.path.join(self.no_split_dir, "train_df_nosplit.csv")
        test_path = os.path.join(self.no_split_dir, "test_df_nosplit.csv")

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError("No-split train/test files not found")

        self.nosplit_train = pd.read_csv(train_path)
        self.nosplit_test = pd.read_csv(test_path)

    def load_inlier_data(self):
        train_path = os.path.join(self.outlier_dir, "train_df_inliers.csv")
        test_path = os.path.join(self.outlier_dir, "test_df_inliers.csv")

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError("Inlier train/test files not found")

        self.inlier_train = pd.read_csv(train_path)
        self.inlier_test = pd.read_csv(test_path)

    def load_outlier_data(self):
        train_path = os.path.join(self.outlier_dir, "train_df_outliers.csv")
        test_path = os.path.join(self.outlier_dir, "test_df_outliers.csv")

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError("Outlier train/test files not found")

        self.outlier_train = pd.read_csv(train_path)
        self.outlier_test = pd.read_csv(test_path)


    def load_classification_data(self):
        train_path = os.path.join(self.class_dir, "train_class.csv")
        test_path = os.path.join(self.class_dir, "test_class.csv")

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError("Classification train/test files not found")

        self.classification_train = pd.read_csv(train_path)
        self.classification_test = pd.read_csv(test_path)


if __name__ == "__main__":
    f_path = "wwlLancMsc_data\\wwlLancMsc_data.csv"
    
    runner = PipelineRunner(f_path=f_path,training=True)
    runner.pre_process_data()
    runner.load_no_split_data()

    models_dir = "saved_models"
    fine_tune_dir = "fine_tuning_results"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(fine_tune_dir, exist_ok=True)

    runner = PipelineRunner(f_path=f_path, training=True)

    # Load no-split data only
    runner.load_no_split_data()
    
    #linear_regression on entire data
    linear_regression = TrainLinearRegressor()
    linear_regression.train_model_with_params(train=runner.nosplit_train,test=runner.nosplit_test)
    linear_regression.save_model(os.path.join(models_dir,"linear_regression_no_split.pkl"))


    

