import os  # Module for interacting with the operating system
import sys  # Module for system-specific parameters and functions
from dataclasses import dataclass  # Decorator for creating data classes
from catboost import CatBoostRegressor  # CatBoost regressor for gradient boosting
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)  # Ensemble methods
from sklearn.linear_model import LinearRegression  # Linear regression model
from sklearn.metrics import r2_score  # Function to calculate R² score for model evaluation
from sklearn.tree import DecisionTreeRegressor  # Decision tree regressor
from xgboost import XGBRegressor  # XGBoost regressor for gradient boosting
from src.exception import CustomException  # Custom exception handling module
from src.logger import logging  # Logging module for tracking events
from src.utils import save_object, evaluate_models  # Utility functions for saving objects and evaluating models


@dataclass
class ModelTrainerConfig:
    """
    Configuration class for the model trainer.
    Defines the file path where the trained model will be saved.
    """
    trained_model_file_path = os.path.join("artifacts", "model.pkl")  # Path to save the trained model


class ModelTrainer:
    def __init__(self):
        """Initialize the ModelTrainer class and set up configuration."""
        self.model_trainer_config = ModelTrainerConfig()  # Create an instance of the configuration class

    def initiate_model_trainer(self, train_array, test_array):
        """
        Method to train multiple models and select the best one based on R² score.

        Parameters:
            train_array (np.ndarray): Array containing training data.
            test_array (np.ndarray): Array containing testing data.

        Returns:
            float: R² score of the best performing model on the test data.
        """

        try:
            logging.info("Split training and test input data")  # Log the start of data splitting

            # Split the input features and target variable from training and testing arrays
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # Features for training
                train_array[:, -1],  # Target variable for training
                test_array[:, :-1],  # Features for testing
                test_array[:, -1]  # Target variable for testing
            )

            # Define a dictionary of models to evaluate
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Define hyperparameters for each model to tune during evaluation
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # Additional parameters can be added here if needed
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]  # Number of trees in the forest
                },
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},  # No hyperparameters to tune for linear regression
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, .5, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            # Evaluate models using provided training and testing data along with hyperparameters
            model_report: dict = evaluate_models(X_train=X_train,
                                                 y_train=y_train,
                                                 X_test=X_test,
                                                 y_test=y_test,
                                                 models=models,
                                                 param=params)

            # Get the best model score from the report dictionary
            best_model_score = max(sorted(model_report.values()))

            # Identify the name of the best model based on its score
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]  # Retrieve the best model from the models dictionary

            if best_model_score < 0.6:  # Check if the best model score is below threshold
                raise CustomException("No best model found")  # Raise an exception if no suitable model is found

            logging.info(
                f"Best found model on training and testing dataset")  # Log successful identification of best model

            save_object(  # Save the trained best model to specified file path
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)  # Make predictions using the best model on test data

            r2_square = r2_score(y_test, predicted)  # Calculate R² score to evaluate performance of predictions

            return r2_square  # Return R² score of the best performing model

        except Exception as e:
            raise CustomException(e, sys)  # Raise a custom exception if an error occurs during processing
