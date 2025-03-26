import sys  # Module for system-specific parameters and functions
from dataclasses import dataclass  # Decorator for creating data classes
import numpy as np  # Library for numerical operations
import pandas as pd  # Library for data manipulation and analysis
from sklearn.compose import ColumnTransformer  # For creating a composite transformer for preprocessing
from sklearn.impute import SimpleImputer  # For handling missing data
from sklearn.pipeline import Pipeline  # For creating a sequence of data processing steps
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # For categorical and numerical data preprocessing
from src.exception import CustomException  # Custom exception handling module
from src.logger import logging  # Logging module for tracking events
import os  # Module for interacting with the operating system

from src.utils import save_object  # Utility function to save objects


@dataclass
class DataTransformationConfig:
    """
    Configuration class for data transformation.
    Defines the file path where the preprocessor object will be saved.
    """
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')  # Path to save the preprocessor object


def get_data_transformer_object():
    """
    This function creates and returns a preprocessor object
    that applies transformations to numerical and categorical features.
    """

    try:
        # Define numerical and categorical columns
        numerical_columns = ["writing_score", "reading_score"]
        categorical_columns = [
            "gender",
            "race_ethnicity",
            "parental_level_of_education",
            "lunch",
            "test_preparation_course",
        ]

        # Create a pipeline for numerical features: imputation followed by scaling
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),  # Replace missing values with median
                ("scaler", StandardScaler())  # Standardize numerical features
            ]
        )

        # Create a pipeline for categorical features: imputation, one-hot encoding, and scaling
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                # Replace missing values with mode (most frequent)
                ("one_hot_encoder", OneHotEncoder()),  # Convert categorical variables into dummy/indicator variables
                ("scaler", StandardScaler(with_mean=False))  # Scale categorical features (without centering)
            ]
        )

        logging.info(f"Categorical Columns: {categorical_columns}")  # Log categorical columns
        logging.info(f"Numerical Columns: {numerical_columns}")  # Log numerical columns

        # Combine both pipelines into a single column transformer
        preprocessor = ColumnTransformer(
            [
                ("num_pipeline", num_pipeline, numerical_columns),  # Apply num_pipeline to numerical columns
                ("cat_pipeline", cat_pipeline, categorical_columns)  # Apply cat_pipeline to categorical columns
            ]
        )

        return preprocessor  # Return the combined preprocessor object

    except Exception as e:
        raise CustomException(e, sys)  # Raise a custom exception if an error occurs


class DataTransformation:
    def __init__(self):
        """Initialize the DataTransformation class and set up configuration."""
        self.data_transformation_config = DataTransformationConfig()  # Create an instance of the configuration class

    def initiate_data_transformation(self, train_path, test_path):
        """
        Method to perform data transformation on training and testing datasets.

        Parameters:
            train_path (str): Path to the training dataset.
            test_path (str): Path to the testing dataset.

        Returns:
            Tuple: Transformed training array, transformed testing array, and path to the preprocessor object.
        """

        try:
            train_df = pd.read_csv(train_path)  # Read training data into a DataFrame
            test_df = pd.read_csv(test_path)  # Read testing data into a DataFrame

            logging.info("Read train and test data completed")  # Log successful reading of datasets
            logging.info("Obtaining preprocessing object")  # Log initiation of preprocessing object retrieval

            preprocessing_obj = get_data_transformer_object()  # Get the preprocessing object

            target_column_name = "math_score"  # Define target column name
            numerical_columns = ["writing_score", "reading_score"]  # Define numerical columns

            input_feature_train_df = train_df.drop(columns=[target_column_name],
                                                   axis=1)  # Drop target column from training data
            target_feature_train_df = train_df[target_column_name]  # Extract target column from training data

            input_feature_test_df = test_df.drop(columns=[target_column_name],
                                                 axis=1)  # Drop target column from testing data
            target_feature_test_df = test_df[target_column_name]  # Extract target column from testing data

            logging.info(f"Applying preprocessing object on training and testing dataframe")  # Log preprocessing step

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df)  # Fit and transform training features
            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df)  # Transform testing features using fitted preprocessor

            train_arr = np.c_[input_feature_train_arr, np.array(
                target_feature_train_df)]  # Combine transformed features with target variable for training set
            test_arr = np.c_[input_feature_test_arr, np.array(
                target_feature_test_df)]  # Combine transformed features with target variable for testing set

            logging.info(f"Saved preprocessing object")  # Log saving of preprocessing object

            save_object(  # Save the preprocessing object to specified file path
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path  # Return transformed arrays and file path of preprocessor object

        except Exception as e:
            raise CustomException(e, sys)  # Raise a custom exception if an error occurs
