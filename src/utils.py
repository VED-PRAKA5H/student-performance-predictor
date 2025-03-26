import os  # For file/directory operations
import sys  # For system-specific parameters
import dill  # Extended pickle module for object serialization
from sklearn.model_selection import GridSearchCV  # For hyperparameter tuning
from src.exception import CustomException  # Custom exception class
from sklearn.metrics import r2_score  # Regression metric
from src.logger import logging  # Logging setup


def save_object(file_path: str, obj: object) -> None:
    """
    Save Python objects to file using dill serialization.

    Parameters:
        file_path (str): Path to save the object
        obj (object): Python object to serialize
    """
    try:
        dir_path = os.path.dirname(file_path)
        # Create directory structure if missing
        os.makedirs(dir_path, exist_ok=True)

        # Serialize and save object
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models: dict, param: dict) -> dict:
    """
    Evaluate multiple models with hyperparameter tuning and return performance report.

    Parameters:
        X_train: Training features
        y_train: Training target
        X_test: Testing features
        y_test: Testing target
        models (dict): Dictionary of model instances
        param (dict): Hyperparameters for each model

    Returns:
        dict: Model names with their test R² scores
    """
    try:
        report = {}

        # Iterate through each model
        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            params = param[model_name]

            # Hyperparameter tuning with Grid Search
            gs = GridSearchCV(model, params, cv=3)  # 3-fold cross-validation
            gs.fit(X_train, y_train)

            # Set best parameters and retrain
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R² scores
            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            logging.info(
                f"{model_name} - Train score: {train_score:.4f}, Test score: {test_score:.4f}")

            # Store test score in report
            report[model_name] = test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str) -> object:
    """
    Load serialized Python object from file.

    Parameters:
        file_path (str): Path to serialized object file

    Returns:
        object: Deserialized Python object
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
