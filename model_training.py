# model_training.py

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib  # For saving the model
import os
import logging

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Paths ---
# Input: Path to the preprocessed data file from feature_engineering.py
PROCESSED_DATA_PATH = './processed_data/final_features.parquet'
# Output: Path to save the trained model
MODEL_SAVE_PATH = './models/gdelt_stock_model.joblib'
MODEL_DIR = './models'

# --- Model & Training Parameters ---
TARGET_VARIABLE = 'Target'  # Name of the target column created in feature_engineering.py
TEST_SET_SIZE_RATIO = 0.2  # Use the last 20% of data for testing
N_SPLITS_CV = 5  # Number of splits for TimeSeriesSplit (for potential hyperparameter tuning, not fully implemented here)

# LightGBM parameters (example, tune these)
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'seed': 42,
    'n_jobs': -1,
    'verbose': -1,  # Suppress LightGBM verbosity
    'colsample_bytree': 0.8,  # Feature fraction
    'subsample': 0.8,  # Bagging fraction
    'reg_alpha': 0.1,  # L1 regularization
    'reg_lambda': 0.1,  # L2 regularization
}


# --- Helper Functions ---

def load_processed_data(filepath):
    """Loads the processed feature data."""
    logging.info(f"Loading processed data from {filepath}...")
    try:
        if filepath.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath, parse_dates=['Date'], index_col=['Date', 'Ticker'])
        else:
            raise ValueError("Unsupported file format. Please use .parquet or .csv")

        # Ensure Date is parsed if it wasn't set as index correctly during saving/loading
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index(['Date', 'Ticker'], inplace=True)
            df.sort_index(inplace=True)

        logging.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logging.error(f"Processed data file not found: {filepath}")
        return None
    except Exception as e:
        logging.error(f"Error loading processed data: {e}", exc_info=True)
        return None


def split_data_chronological(df, target_column, test_size=0.2):
    """
    Splits the data chronologically into training and testing sets.
    Assumes the DataFrame is sorted by Date.

    Args:
        df (pd.DataFrame): The input DataFrame with features and target.
        target_column (str): The name of the target variable column.
        test_size (float): The proportion of the data to use for the test set.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    logging.info(f"Splitting data chronologically. Test size ratio: {test_size}")
    split_index = int(len(df) * (1 - test_size))

    # Ensure split happens at a date boundary if multi-index is flattened
    # For simplicity with multi-index, we split by position after sorting
    df_sorted = df.sort_index(level='Date')  # Ensure chronological order overall

    train_df = df_sorted.iloc[:split_index]
    test_df = df_sorted.iloc[split_index:]

    feature_columns = [col for col in df.columns if
                       col != target_column and col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Ticker',
                                                            'SQLDATE', 'GLOBALEVENTID', 'Actor1Name', 'Actor2Name',
                                                            'SOURCEURL', 'Title From URL', 'CompanyName', 'Sector',
                                                            'Industry']]  # Adapt if needed

    X_train = train_df[feature_columns]
    y_train = train_df[target_column]
    X_test = test_df[feature_columns]
    y_test = test_df[target_column]

    logging.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test, feature_columns


# --- Main Training Logic ---

def train_and_evaluate(X_train, y_train, X_test, y_test, model_params, feature_columns):
    """
    Trains the model using a pipeline and evaluates it on the test set.

    Args:
        X_train, X_test, y_train, y_test: Data splits.
        model_params (dict): Parameters for the LightGBM classifier.
        feature_columns (list): List of feature column names.

    Returns:
        Pipeline: The trained scikit-learn pipeline object.
    """
    logging.info("Setting up model pipeline...")

    # Define preprocessing steps (handle potential NaNs from lagging/merging)
    # LightGBM can handle NaNs internally, but explicit imputation is safer for pipelines
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()

    # Check if there are actually numeric features to process
    if not numeric_features:
        logging.error("No numeric features found for preprocessing.")
        return None

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Use median for robustness to outliers
        ('scaler', StandardScaler())  # Scale features
    ])

    # Create the preprocessor using ColumnTransformer
    # This setup assumes all features passed are numeric. If categorical features exist,
    # they would need separate handling (e.g., OneHotEncoder or TargetEncoder).
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
            # Add transformers for categorical features if needed
            # ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'  # Keep any columns not specified (should ideally be none)
    )

    # Define the model
    lgbm = lgb.LGBMClassifier(**model_params)

    # Create the full pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', lgbm)])

    logging.info("Starting model training...")
    pipeline.fit(X_train, y_train)
    logging.info("Model training complete.")

    # --- Evaluation ---
    logging.info("Evaluating model on the test set...")
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]  # Probability of class 1

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print("\n--- Model Evaluation Results (Test Set) ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("-------------------------------------------")

    # Feature Importance (specific to tree-based models like LightGBM)
    try:
        feature_importances = pipeline.named_steps['classifier'].feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_columns,  # Use original feature names before potential scaling
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        print("\nFeature Importances (Top 20):")
        print(importance_df.head(20))
    except Exception as e:
        logging.warning(f"Could not retrieve feature importances: {e}")

    return pipeline


def save_model(model_pipeline, file_path):
    """Saves the trained model pipeline."""
    logging.info(f"Saving model to {file_path}...")
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(model_pipeline, file_path)
        logging.info("Model saved successfully.")
    except Exception as e:
        logging.error(f"Error saving model: {e}", exc_info=True)


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load Processed Data
    data = load_processed_data(PROCESSED_DATA_PATH)

    if data is not None and not data.empty:
        # 2. Prepare Data for Modeling & Split
        # Ensure data is sorted by date before splitting
        data = data.sort_index(level='Date')
        X_train, X_test, y_train, y_test, feature_cols = split_data_chronological(
            data, TARGET_VARIABLE, test_size=TEST_SET_SIZE_RATIO
        )

        # 3. Train and Evaluate Model
        trained_pipeline = train_and_evaluate(X_train, y_train, X_test, y_test, LGBM_PARAMS, feature_cols)

        # 4. Save the Trained Model
        if trained_pipeline:
            save_model(trained_pipeline, MODEL_SAVE_PATH)
        else:
            logging.error("Model training failed, model not saved.")
    else:
        logging.error("Could not load processed data. Training aborted.")

    print("\nModel training script finished.")