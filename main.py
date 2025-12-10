import pandas as pd
from sklearn.model_selection import train_test_split
import os
from pipeline import PipelineManager
from models.LinearRegression import TrainLinearRegressor
from models.RandomForest import TrainRandomForestRegressor
from models.XGboost import TrainXGBoostRegressor
from models.IsolationForest import TrainIsolationForest
from models.RandomForestClassifier import TrainRandomFClassifier
from models.Logistic_Regression import TrainLogisticRegression
import numpy as np

class PipelineRunner:
    def __init__(self, f_path: str):
        self.f_path = f_path
        self.outlier_dir = "Flow 2\\Split_Layer_Data"
        self.class_dir = "Flow 2\\Classification_Layer_Data"
        self.no_split_dir = "Flow 1 - Entire PreProcessed"
        self.fine_tune_dir = "fine_tuning_results"
        self.models_dir = "pre_trained_models"
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
        train_path = os.path.join(self.outlier_dir, "train_inlier.csv")
        test_path = os.path.join(self.outlier_dir, "test_inlier.csv")

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError("Inlier train/test files not found")

        self.inlier_train = pd.read_csv(train_path)
        self.inlier_test = pd.read_csv(test_path)

    def load_outlier_data(self):
        train_path = os.path.join(self.outlier_dir, "train_outlier.csv")
        test_path = os.path.join(self.outlier_dir, "test_outlier.csv")

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

    def unpack_target_log(self, data: pd.DataFrame | pd.Series):
        return pd.DataFrame(np.floor(data.apply(np.expm1)))


def train_evaluate_model(
    regressor, 
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    utils:PipelineRunner,  # your object containing unpack, model dir, etc.
    fine_tune_filepath: str | None = None,
    save_filename: str | None = None
):
    """
    Trains, predicts, unpacks, and evaluates a regression model.
    
    Args:
        regressor: an instance of your Train*Regressor class
        train_df: training dataframe
        test_df: test dataframe
        flow1_training: object with models_dir, unpack_target_log, etc.
        fine_tune: whether to run fine-tuning (grid search)
        fine_tune_filepath: filepath for saving fine-tune results (if any)
        save_filename: name to save the trained model
    """

   
    tune_params = regressor.fine_tune_model(train=train_df, test=test_df, filepath=fine_tune_filepath)
    tune_params = {}

    # 2️⃣ Train the model
    regressor.train_model_with_params(train=train_df, test=test_df, **tune_params)
    regressor.save_model(filepath=utils.models_dir, filename=save_filename)

    # 4️⃣ Run predictions
    predictions = regressor.run(val=test_df)

    # 5️⃣ Prepare dataframe with actual & predicted values
    final_df = pd.DataFrame({
        "actual_los": test_df["spell_episode_los"],
        "predicted_los": predictions
    })

    # 6️⃣ Unpack the target if necessary (log or other transforms)
    unpacked_df = utils.unpack_target_log(final_df)

    # 7️⃣ Evaluate
    regressor.evaluate_model(
        original_values=unpacked_df["actual_los"], 
        predictions=unpacked_df["predicted_los"]
    )

    return unpacked_df

def train_evaluate_classifier(
    classifier, 
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    utils:PipelineRunner,  # your object containing unpack, model dir, etc.
    fine_tune_filepath: str | None = None,
    save_filename: str | None = None
):
    """
    Trains, predicts, unpacks, and evaluates a regression model.
    
    Args:
        regressor: an instance of your Train*Regressor class
        train_df: training dataframe
        test_df: test dataframe
        flow1_training: object with models_dir, unpack_target_log, etc.
        fine_tune: whether to run fine-tuning (grid search)
        fine_tune_filepath: filepath for saving fine-tune results (if any)
        save_filename: name to save the trained model
    """

   
    tune_params = classifier.fine_tune_model(train=train_df, test=test_df, filepath=fine_tune_filepath)
    tune_params = {}

    # 2️⃣ Train the model
    classifier.train_model_with_params(train=train_df, test=test_df, **tune_params)
    classifier.save_model(filepath=utils.models_dir, filename=save_filename)

    # 4️⃣ Run predictions
    predictions = classifier.run(val=test_df)

    # 5️⃣ Prepare dataframe with actual & predicted values
    final_df = pd.DataFrame({
        "actual_outlier": test_df["is_outlier"],
        "predicted_outlier": predictions
    })


    return final_df


if __name__ == "__main__":
    f_path = "wwlLancMsc_data\\wwlLancMsc_data.csv"
    utility_funcs = PipelineRunner(f_path=f_path)
    # utility_funcs.pre_process_data()


    # -------------------------
    # Flow 1 - Entire Training
    # -------------------------
    #utility_funcs.load_no_split_data()

    # # Define models to train: (Regressor class, model_type, model_filename)
    # no_split_models = [
    #     (TrainLinearRegressor, "linear", "flow1_linear_reg.pkl"),
    #     (TrainLinearRegressor, "lasso", "flow1_lasso_reg.pkl"),
    #     (TrainLinearRegressor, "ridge", "flow1_ridge_reg.pkl"),
    #     (TrainRandomForestRegressor, None, "flow1_random_forest_reg.pkl"),
    #     (TrainXGBoostRegressor, None, "flow1_xgboost_reg.pkl")
    # ]
    # # Define models to train: (Regressor class, model_type, model_filename)
    # no_split_models = [
    #     (TrainLinearRegressor, "linear", "flow1_linear_reg.pkl"),
    #     (TrainLinearRegressor, "lasso", "flow1_lasso_reg.pkl"),
    #     (TrainLinearRegressor, "ridge", "flow1_ridge_reg.pkl"),
    #     (TrainRandomForestRegressor, None, "flow1_random_forest_reg.pkl"),
    #     (TrainXGBoostRegressor, None, "flow1_xgboost_reg.pkl")
    # ]

    # for reg_class, model_type, model_filename in no_split_models:
    #     regressor = reg_class() if not model_type else reg_class(model_type=model_type)
    #     fine_tune_dir = os.path.join(utility_funcs.fine_tune_dir, model_filename.replace(".pkl", ".md"))
    # for reg_class, model_type, model_filename in no_split_models:
    #     regressor = reg_class() if not model_type else reg_class(model_type=model_type)
    #     fine_tune_dir = os.path.join(utility_funcs.fine_tune_dir, model_filename.replace(".pkl", ".md"))

    #     train_evaluate_model(
    #         regressor=regressor,
    #         train_df=utility_funcs.nosplit_train,
    #         test_df=utility_funcs.nosplit_test,
    #         utils=utility_funcs,
    #         fine_tune_filepath=fine_tune_dir,
    #         save_filename=os.path.join(utility_funcs.models_dir, model_filename)
    #     )
    #     train_evaluate_model(
    #         regressor=regressor,
    #         train_df=utility_funcs.nosplit_train,
    #         test_df=utility_funcs.nosplit_test,
    #         utils=utility_funcs,
    #         fine_tune_filepath=fine_tune_dir,
    #         save_filename=os.path.join(utility_funcs.models_dir, model_filename)
    #     )




    # -------------------------
    # Flow 2 - Classifier
    # -------------------------
    utility_funcs.load_classification_data()

    # Define models to train: (Regressor class, model_type, model_filename)
    classification_models = [
        #(TrainRandomFClassifier, "flow2_random_forest_classifier.pkl"),
        (TrainLogisticRegression,"flow2_logistic_regression.pkl")
    ]
    for classifier_class, model_filename in classification_models:
        classifier = classifier_class()
        fine_tune_dir = os.path.join(utility_funcs.fine_tune_dir, model_filename.replace(".pkl", ".md"))

        train_evaluate_classifier(
            classifier=classifier,
            train_df=utility_funcs.classification_train,
            test_df=utility_funcs.classification_test,
            utils=utility_funcs,
            fine_tune_filepath=fine_tune_dir,
            save_filename=os.path.join(utility_funcs.models_dir, model_filename)
        )

    # -------------------------
    # Flow 2 - Inliers Training
    # -------------------------
    # utility_funcs.load_inlier_data()

    # # Define models to train: (Regressor class, model_type, model_filename)
    # inlier_models = [
    #     (TrainLinearRegressor, "linear", "flow2_linear_reg_inliers.pkl"),
    #     (TrainLinearRegressor, "lasso", "flow2_lasso_reg_inliers.pkl"),
    #     (TrainLinearRegressor, "ridge", "flow2_ridge_reg_inliers.pkl"),
    #     (TrainRandomForestRegressor, None, "flow2_random_forest_reg_inliers.pkl"),
    #     (TrainXGBoostRegressor, None, "flow2_xgboost_reg_inliers.pkl")
    # ]
    # # Define models to train: (Regressor class, model_type, model_filename)
    # inlier_models = [
    #     (TrainLinearRegressor, "linear", "flow2_linear_reg_inliers.pkl"),
    #     (TrainLinearRegressor, "lasso", "flow2_lasso_reg_inliers.pkl"),
    #     (TrainLinearRegressor, "ridge", "flow2_ridge_reg_inliers.pkl"),
    #     (TrainRandomForestRegressor, None, "flow2_random_forest_reg_inliers.pkl"),
    #     (TrainXGBoostRegressor, None, "flow2_xgboost_reg_inliers.pkl")
    # ]

    # for reg_class, model_type, model_filename in inlier_models:
    #     regressor = reg_class() if not model_type else reg_class(model_type=model_type)
    #     fine_tune_dir = os.path.join(utility_funcs.fine_tune_dir, model_filename.replace(".pkl", ".md"))
    # for reg_class, model_type, model_filename in inlier_models:
    #     regressor = reg_class() if not model_type else reg_class(model_type=model_type)
    #     fine_tune_dir = os.path.join(utility_funcs.fine_tune_dir, model_filename.replace(".pkl", ".md"))

    #     train_evaluate_model(
    #         regressor=regressor,
    #         train_df=utility_funcs.inlier_train,
    #         test_df=utility_funcs.inlier_test,
    #         utils=utility_funcs,
    #         fine_tune_filepath=fine_tune_dir,
    #         save_filename=os.path.join(utility_funcs.models_dir, model_filename)
    #     )
    #     train_evaluate_model(
    #         regressor=regressor,
    #         train_df=utility_funcs.inlier_train,
    #         test_df=utility_funcs.inlier_test,
    #         utils=utility_funcs,
    #         fine_tune_filepath=fine_tune_dir,
    #         save_filename=os.path.join(utility_funcs.models_dir, model_filename)
    #     )

    # # -------------------------
    # # Flow 2 - Outliers Training
    # # -------------------------
    # utility_funcs.load_outlier_data()
    # # -------------------------
    # # Flow 2 - Outliers Training
    # # -------------------------
    # utility_funcs.load_outlier_data()

    # # Use the same model setup for outliers (just change the filenames)
    # outlier_models = [
    #     (TrainLinearRegressor, "linear", "flow2_linear_reg_outliers.pkl"),
    #     (TrainLinearRegressor, "lasso", "flow2_lasso_reg_outliers.pkl"),
    #     (TrainLinearRegressor, "ridge", "flow2_ridge_reg_outliers.pkl"),
    #     (TrainRandomForestRegressor, None, "flow2_random_forest_reg_outliers.pkl"),
    #     (TrainXGBoostRegressor, None, "flow2_xgboost_reg_outliers.pkl")
    # ]
    # # Use the same model setup for outliers (just change the filenames)
    # outlier_models = [
    #     (TrainLinearRegressor, "linear", "flow2_linear_reg_outliers.pkl"),
    #     (TrainLinearRegressor, "lasso", "flow2_lasso_reg_outliers.pkl"),
    #     (TrainLinearRegressor, "ridge", "flow2_ridge_reg_outliers.pkl"),
    #     (TrainRandomForestRegressor, None, "flow2_random_forest_reg_outliers.pkl"),
    #     (TrainXGBoostRegressor, None, "flow2_xgboost_reg_outliers.pkl")
    # ]

    # for reg_class, model_type, model_filename in outlier_models:
    #     regressor = reg_class() if not model_type else reg_class(model_type=model_type)
    #     fine_tune_dir = os.path.join(utility_funcs.fine_tune_dir, model_filename.replace(".pkl", ".md"))
    # for reg_class, model_type, model_filename in outlier_models:
    #     regressor = reg_class() if not model_type else reg_class(model_type=model_type)
    #     fine_tune_dir = os.path.join(utility_funcs.fine_tune_dir, model_filename.replace(".pkl", ".md"))

    #     train_evaluate_model(
    #         regressor=regressor,
    #         train_df=utility_funcs.outlier_train,
    #         test_df=utility_funcs.outlier_test,
    #         utils=utility_funcs,
    #         fine_tune_filepath=fine_tune_dir,
    #         save_filename=os.path.join(utility_funcs.models_dir, model_filename)
    #     )
    #     train_evaluate_model(
    #         regressor=regressor,
    #         train_df=utility_funcs.outlier_train,
    #         test_df=utility_funcs.outlier_test,
    #         utils=utility_funcs,
    #         fine_tune_filepath=fine_tune_dir,
    #         save_filename=os.path.join(utility_funcs.models_dir, model_filename)
    #     )
