'''
Preprocessing Pipeline is as follows: 
1. Load data
2. drop columns
3. fill na values (doesnt use any mean/median imputing only simple logical filling)
4. split into outliers and inliers OR add is_outlier column depending on what type of training
'''
import pandas as pd
from ..preprocessing import MedicalDataPreprocessor
from typing import Union, Tuple

class preprocessing_pipeline:

    def __init__(self, filepath: str | None = None, df: pd.DataFrame | None = None,class_bool = False):
        self.processor = MedicalDataPreprocessor()
        self.is_classification = class_bool
        self.__inlier_data: pd.DataFrame | None = None
        self.__outlier_data: pd.DataFrame | None = None
        if df is not None:
            self.dataframe = df
        elif filepath is not None:
            self.dataframe = pd.read_csv(filepath)
        else:
            raise ValueError("Provide either a filepath or a dataframe.")

        
    def cleaning(self):
        print("cleaning data and removing columns")
        self.dataframe = self.processor.initial_cleaning(self.dataframe)
        return self.dataframe
    
    def fill_empty(self):
        print("filling empty values")
        self.dataframe = self.processor.fill_na_values(self.dataframe)

    def outlier_detection(self):
        if (self.is_classification):
            print("outlier detection for classification so 1 df returned")
            self.dataframe = self.processor.classify_outliers(df=self.dataframe)
        else:
            print("outlier detection for classification so 2 df's returned")
            self.__inlier_data,self.__outlier_data=self.processor.split_outliers(self.dataframe)

    def get_dataframes(self) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        if (self.is_classification):
            print("returning single dataframe with is_outlier column")
            return self.dataframe
        else:
            print("returning 2 dataframs with inliers and outliers data")
            return self.__inlier_data,self.__outlier_data
        
    
    