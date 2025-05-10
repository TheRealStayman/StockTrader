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
import joblib
import os
import logging

# Import from local modules
from utils import load_processed_data, save_model_artefacts  # Use utils for loading data and saving model

logger = logging.getLogger(__name__)

# --- Model & Training Parameters ---
# These could be moved to a central config file or passed to main training function
TARGET_VARIABLE = 'Target'
TEST_SET_SIZE_RATIO = 0.2
# N_SPLITS_CV = 5 # For hyperparameter tuning, not fully implemented here

# LightGBM parameters (example, tune these)
LGBM_PARAMS = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'n_estimators': 200, 'learning_rate': 0.05,  # Reduced n_estimators for faster example
    'num_leaves': 31, 'max_depth': -1, 'seed': 42, 'n_jobs': -1,
    'verbose': -1, 'colsample_bytree': 0.8, 'subsample': 0.8,
    'reg_alpha': 0.1, 'reg_lambda': 0.1,
}


def split_data_chronological(df, target_column_name, test_size=0.2,
                             cols_to_exclude_from_features=None):
    """
    Splits data chronologically. Assumes df is sorted by Date.
    Identifies feature columns by excluding target and other specified columns.
    """
    logger.info(f"Splitting data chronologically. Test size: {test_size}")

    # Ensure DataFrame is sorted by the Date level of the MultiIndex
    if isinstance(df.index, pd.MultiIndex) and 'Date' in df.index.names:
        df_sorted = df.sort_index(level='Date')
    else:  # Fallback if not MultiIndex or Date is not a level, sort by Date column if exists
        if 'Date' in df.columns:
            df_sorted = df.sort_values(by='Date')
        elif 'Date' in df.index.names:  # Single index named Date
            df_sorted = df.sort_index()
        else:  # Default to sorting by index if Date is not clearly identified
            logger.warning(
                "Date column/index level not explicitly found for sorting. Assuming pre-sorted or using existing index sort.")
            df_sorted = df.sort_index()

    split_idx = int(len(df_sorted) * (1 - test_size))

    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]

    if cols_to_exclude_from_features is None:
        cols_to_exclude_from_features = [
            target_column_name, 'Open', 'High', 'Low', 'Close', 'Volume',  # Base OHLCV
            'Ticker',  # Identifier
            # Raw GDELT fields that might have been merged but not directly used as features
            'GLOBALEVENTID', 'SQLDATE', 'Actor1Name', 'Actor2Name',
            'SOURCEURL', 'Title From URL',
            # Stock info fields that might have been merged
            'CompanyName', 'Sector', 'Industry', 'CompanyName_Clean',
            # Backtrader specific columns if they sneak in
            'openinterest'
        ]

    feature_columns = [col for col in df.columns if
                       col not in cols_to_exclude_from_features and col != target_column_name]

    # Ensure target is not in features
    if target_column_name in feature_columns:
        feature_columns.remove(target_column_name)

    # Check for empty feature list
    if not feature_columns:
        logger.error("No feature columns identified after exclusions. Check `cols_to_exclude_from_features`.")
        return None, None, None, None, []

    X_train = train_df[feature_columns]
    y_train = train_df[target_column_name]
    X_test = test_df[feature_columns]
    y_test = test_df[target_column_name]

    logger.info(
        f"Train set shape: X-{X_train.shape}, y-{y_train.shape}. Test set shape: X-{X_test.shape}, y-{y_test.shape}")
    logger.info(f"Identified {len(feature_columns)} feature columns: {feature_columns[:5]}...")  # Log first 5
    return X_train, X_test, y_train, y_test, feature_columns


def train_evaluate_and_save_model(processed_data_path, target_variable_name,
                                  model_save_path, features_save_path,
                                  lgbm_params=None, test_ratio=0.2):
    """
    Main function to load data, train, evaluate, and save the model and feature list.
    """
    if lgbm_params is None:
        lgbm_params = LGBM_PARAMS

    data = load_processed_data(processed_data_path)
    if data is None or data.empty:
        logger.error("Model training aborted: No data loaded.")
        return None, []

    # Drop rows with NaN in target (can happen due to target shifting)
    initial_rows = len(data)
    data.dropna(subset=[target_variable_name], inplace=True)
    if initial_rows > len(data):
        logger.info(f"Dropped {initial_rows - len(data)} rows with NaN in target variable '{target_variable_name}'.")

    if data.empty:
        logger.error("Model training aborted: Data became empty after dropping NaN targets.")
        return None, []

    X_train, X_test, y_train, y_test, feature_cols = split_data_chronological(
        data, target_variable_name, test_size=test_ratio
    )

    if X_train is None or X_train.empty:
        logger.error("Model training aborted: Training data is empty after split.")
        return None, []
    if not feature_cols:
        logger.error("Model training aborted: No feature columns identified.")
        return None, []

    logger.info("Setting up model pipeline...")
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()

    # Ensure that numeric_features are a subset of feature_cols
    # (or all feature_cols are numeric after previous steps)
    numeric_features = [f for f in numeric_features if f in feature_cols]

    if not numeric_features:
        logger.warning(
            "No numeric features identified for scaling/imputation within the selected feature_cols. Model will use raw features.")
        # If preprocessor expects numeric features, this will be an issue.
        # For LightGBM, it might be okay if it handles NaNs and scale is not critical.
        # However, a robust pipeline usually defines this.
        # Create pipeline without numeric_transformer if numeric_features is empty
        preprocessor = ColumnTransformer(transformers=[], remainder='passthrough')

    else:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        preprocessor = ColumnTransformer(
            transformers=[('num', numeric_transformer, numeric_features)],
            remainder='passthrough'  # Important: Keeps columns not in numeric_features
        )

    lgbm_classifier = lgb.LGBMClassifier(**lgbm_params)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', lgbm_classifier)])

    logger.info("Starting model training...")
    pipeline.fit(X_train, y_train)
    logger.info("Model training complete.")

    logger.info("Evaluating model on the test set...")
    if X_test.empty:
        logger.warning("Test set is empty. Skipping evaluation.")
    else:
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        logger.info(f"\n--- Model Evaluation Results (Test Set) ---\n"
                    f"Accuracy: {accuracy:.4f}\n"
                    f"ROC AUC Score: {roc_auc:.4f}\n"
                    f"Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}\n"
                    f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n"
                    f"-------------------------------------------")

        # Feature Importance
        try:
            # Get feature names after preprocessing (if ColumnTransformer was used effectively)
            # This can be complex depending on the preprocessor structure.
            # For simplicity, we use the original feature_cols assuming the order is maintained
            # or the classifier directly provides importances matching them.
            if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
                importances = pipeline.named_steps['classifier'].feature_importances_

                # If preprocessor changes number of features (e.g. OHE), `feature_cols` won't match.
                # A robust way is to get names from preprocessor.get_feature_names_out()
                # For now, let's assume feature_cols map directly to importances if length matches.
                if len(importances) == len(feature_cols):
                    importance_df = pd.DataFrame({
                        'Feature': feature_cols,
                        'Importance': importances
                    }).sort_values(by='Importance', ascending=False)
                    logger.info(f"\nFeature Importances (Top 20):\n{importance_df.head(20)}")
                else:
                    logger.warning(
                        f"Mismatch in number of feature importances ({len(importances)}) and original feature columns ({len(feature_cols)}). Cannot reliably map names.")
            else:
                logger.warning("Classifier does not have 'feature_importances_' attribute.")
        except Exception as e:
            logger.warning(f"Could not retrieve/display feature importances: {e}")

    # Save model and feature list
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    save_model_artefacts(pipeline, feature_cols, model_save_path, features_save_path)

    return pipeline, feature_cols


if __name__ == "__main__":
    logger.info("--- Running Model Training Script (Example Usage) ---")

    # Define paths for the example run
    # These would typically be sourced from a config in the main script
    processed_data_input_path = os.path.join('processed_data', 'final_features.parquet')  # Example path
    model_output_path = os.path.join('models', 'test_gdelt_stock_model.joblib')
    features_output_path = os.path.join('models', 'test_gdelt_stock_model_features.joblib')

    # Create dummy processed data if it doesn't exist for the test run
    if not os.path.exists(processed_data_input_path):
        logger.warning(f"Dummy processed data {processed_data_input_path} not found, creating a minimal version.")
        os.makedirs('processed_data', exist_ok=True)
        # Create a very simple DataFrame
        dates = pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02', '2023-01-03', '2023-01-03'])
        tickers = ['AAPL', 'MSFT'] * 3
        dummy_processed_df = pd.DataFrame({
            'Date': dates,
            'Ticker': tickers,
            'Close': np.random.rand(6) * 100 + 100,
            'Volume': np.random.rand(6) * 10000,
            'SMA_20': np.random.rand(6) * 100 + 90,
            'RSI_14': np.random.rand(6) * 50 + 25,
            'gdelt_tone_mean': np.random.rand(6) * 2 - 1,
            'Target': np.random.randint(0, 2, 6)
        }).set_index(['Date', 'Ticker'])
        dummy_processed_df.to_parquet(processed_data_input_path)

    # Run the training and evaluation
    trained_model, trained_features = train_evaluate_and_save_model(
        processed_data_path=processed_data_input_path,
        target_variable_name=TARGET_VARIABLE,
        model_save_path=model_output_path,
        features_save_path=features_output_path,
        lgbm_params=LGBM_PARAMS,  # Use default or pass custom
        test_ratio=TEST_SET_SIZE_RATIO
    )

    if trained_model:
        logger.info("Example model training script finished successfully.")
    else:
        logger.error("Example model training script failed.")