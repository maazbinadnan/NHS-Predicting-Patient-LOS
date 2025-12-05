import pandas as pd
from typing import Union, Tuple, Optional
from PreProcessing.preprocessing import MedicalDataPreprocessor



class PreprocessingPipeline:
    def __init__(self, filepath: Optional[str] = None, df: Optional[pd.DataFrame] = None, class_bool: bool = False):
        self.processor = MedicalDataPreprocessor()
        self.is_classification = class_bool
        
        # Initialize internal storage
        # Use single underscore '_' for protected attributes, not double '__'
        self._inlier_data: Optional[pd.DataFrame] = None
        self._outlier_data: Optional[pd.DataFrame] = None
        
        if df is not None:
            self.dataframe = df.copy() # Good practice to copy input data
        elif filepath is not None:
            self.dataframe = pd.read_csv(filepath)
        else:
            raise ValueError("Provide either a filepath or a dataframe.")

    def cleaning(self):
        print("Step 1: Cleaning data...")
        self.dataframe = self.processor.initial_cleaning(self.dataframe)
        return self.dataframe
    
    def fill_empty(self):
        print("Step 2: Filling empty values...")
        self.dataframe = self.processor.fill_na_values(self.dataframe)

    def outlier_detection(self):
        if self.is_classification:
            print("Step 3: Outlier detection (Classification mode - Flagging only)")
            # Assuming classify_outliers returns a DF with an 'is_outlier' column
            self.dataframe = self.processor.classify_outliers(df=self.dataframe)
        else:
            print("Step 3: Outlier detection (Regression mode - Splitting data)")
            self._inlier_data, self._outlier_data = self.processor.split_outliers(self.dataframe)

    def get_dataframes(self) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Returns the processed data.
        Raises an error if outlier_detection hasn't been run yet.
        """
        if self.is_classification:
            print("Returning single dataframe (Classification)")
            return self.dataframe
        else:
            # SAFETY CHECK: Ensure data actually exists before returning
            if self._inlier_data is None or self._outlier_data is None:
                raise RuntimeError("Pipeline incomplete! You must run 'outlier_detection()' before getting dataframes.")
            
            print("Returning inliers and outliers (Regression)")
            return self._inlier_data, self._outlier_data