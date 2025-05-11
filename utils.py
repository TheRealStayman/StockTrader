# utils.py
import pandas as pd
import numpy as np
import joblib
import logging
import backtrader as bt
import re
import gc

logger = logging.getLogger(__name__)


def clean_company_name(name):
    """Removes common suffixes like Inc., Corp., Ltd. for better matching."""
    if pd.isna(name):
        return None
    name = str(name)
    # Expanded list of suffixes and cleaning steps
    name = name.lower()
    name = re.sub(r'[,.\(\)-]', '', name)  # Remove punctuation
    suffixes = [
        ' inc', ' incorporated', ' corp', ' corporation', ' ltd', ' limited',
        ' plc', ' co', ' company', ' lp', ' llp', ' llc', ' ag', ' sa',
        ' sas', ' nv', ' bv', ' ab', ' oyj', ' gmbh', ' spa'
    ]
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name.strip().upper()  # Standardize to uppercase for matching


def load_processed_data(filepath):
    """Loads the processed feature data."""
    logger.info(f"Loading processed data from {filepath}...")
    try:
        if filepath.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        elif filepath.endswith('.csv'):
            # Assuming if it's CSV, Date and Ticker might not be index yet
            df = pd.read_csv(filepath, parse_dates=['Date'])
        else:
            logger.error(f"Unsupported file format: {filepath}. Please use .parquet or .csv")
            raise ValueError("Unsupported file format. Please use .parquet or .csv")

        if 'Date' in df.columns and 'Ticker' in df.columns:
            if not isinstance(df.index, pd.MultiIndex) or df.index.names != ['Date', 'Ticker']:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index(['Date', 'Ticker'], inplace=True)
        df.sort_index(inplace=True)

        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Processed data file not found: {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error loading processed data from {filepath}: {e}", exc_info=True)
        return None


def save_model_artefacts(pipeline, feature_columns, model_path, features_path):
    """Saves the trained model pipeline and its feature columns."""
    logger.info(f"Saving model pipeline to {model_path}")
    try:
        joblib.dump(pipeline, model_path)
        logger.info("Model pipeline saved successfully.")

        logger.info(f"Saving feature columns to {features_path}")
        joblib.dump(feature_columns, features_path)  # Save as a list
        logger.info("Feature columns saved successfully.")

    except Exception as e:
        logger.error(f"Error saving model artefacts: {e}", exc_info=True)


def load_model_artefacts(model_path, features_path):
    """Loads the trained model pipeline and its feature columns."""
    pipeline = None
    feature_columns = None

    try:
        logger.info(f"Loading model pipeline from {model_path}")
        pipeline = joblib.load(model_path)
        logger.info("Model pipeline loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Model pipeline file not found: {model_path}")
    except Exception as e:
        logger.error(f"Error loading model pipeline from {model_path}: {e}", exc_info=True)

    try:
        logger.info(f"Loading feature columns from {features_path}")
        feature_columns = joblib.load(features_path)
        logger.info(f"Feature columns loaded successfully: {len(feature_columns)} features.")
    except FileNotFoundError:
        logger.error(
            f"Feature columns file not found: {features_path}. Model may require feature_cols to be inferred or manually set.")
    except Exception as e:
        logger.error(f"Error loading feature columns from {features_path}: {e}", exc_info=True)

    return pipeline, feature_columns


def downcast_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcasts numeric columns in a DataFrame to more memory-efficient types.
    Iterates column by column to reduce peak memory usage during the operation.
    Provides verbose logging.
    """
    logger.info("Attempting to downcast numeric columns (verbose method)...")
    if df.empty:
        logger.warning("Input DataFrame is empty, skipping downcasting.")
        return df

    original_mem_usage = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logger.info(f"Original DataFrame memory usage: {original_mem_usage:.2f} MB")
    cols_changed_count = 0

    for col in df.columns:
        original_dtype = df[col].dtype

        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(original_dtype) and not pd.api.types.is_bool_dtype(original_dtype):
            if pd.api.types.is_float_dtype(original_dtype):
                # Downcast float columns (e.g., float64 to float32)
                df[col] = pd.to_numeric(df[col], downcast='float')
            elif pd.api.types.is_integer_dtype(original_dtype):
                # Downcast integer columns (e.g., int64 to int32, int16, or int8)
                df[col] = pd.to_numeric(df[col], downcast='integer')

            if df[col].dtype != original_dtype:
                logger.info(f"Downcasted column '{col}': {original_dtype} -> {df[col].dtype}")
                cols_changed_count += 1
            # else:
            # logger.debug(f"Column '{col}' (dtype: {original_dtype}) not downcasted further (already optimal or type not targeted).")
        # else:
        # logger.debug(f"Column '{col}' is not a standard numeric type targeted for downcasting (dtype: {original_dtype}), skipping.")

    if cols_changed_count == 0:
        logger.warning(
            "No columns were actually downcasted. Data might already be in optimal types (e.g., float32) or non-standard numeric types not handled by this basic downcaster.")
    else:
        logger.info(f"Successfully downcasted {cols_changed_count} numeric columns.")

    new_mem_usage = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logger.info(f"DataFrame memory usage after downcasting attempt: {new_mem_usage:.2f} MB")
    if original_mem_usage > 0:
        mem_saved = original_mem_usage - new_mem_usage
        if abs(mem_saved) > 0.01:  # Log if there's a noticeable change
            logger.info(f"Memory change after downcasting: {mem_saved:.2f} MB (Positive means saved)")
    return df


class PandasDataWithFeatures(bt.feeds.PandasData):
    """
    Custom Backtrader data feed to include feature columns alongside standard OHLCV.
    It expects 'dataname' to be a pandas DataFrame.
    - The 'datetime' parameter (from kwargs) maps to a column with pandas datetime objects.
    - 'open', 'high', 'low', 'close', 'volume', 'openinterest' (from kwargs) map to their respective columns.
    - Each string in 'feature_cols' (a direct argument to __init__)
      corresponds to a column name in the DataFrame and will be available as a line.
    """

    def __init__(self, feature_cols=None, **kwargs):
        _custom_feature_cols = feature_cols if feature_cols is not None else []

        # Define the standard data lines explicitly.
        # These are the lines typically accessed like self.datas[0].close, self.datas[0].open, etc.
        # 'datetime' is handled by the 'datetime' parameter to __init__ and is the primary "line 0".
        _standard_lines_list = ['open', 'high', 'low', 'close', 'volume', 'openinterest']

        # Combine standard lines with custom feature lines
        _all_lines_list = list(_standard_lines_list)  # Start with a mutable copy of standard lines

        for f_col in _custom_feature_cols:
            if f_col not in _all_lines_list:  # Avoid duplicates if a feature is named like a standard line
                _all_lines_list.append(f_col)

        # Set the 'lines' attribute for this feed instance.
        # This must be done BEFORE calling super().__init__ if the superclass's __init__
        # inspects self.lines during its setup.
        self.lines = tuple(_all_lines_list)

        # Prepare kwargs for the superclass.
        # Standard mappings (e.g., open='OpenCol', high='HighCol') are already in kwargs
        # from the call in main.py.
        # For custom feature lines, we need to ensure Backtrader knows which DataFrame column
        # corresponds to each new line. If a line is named 'MyFeature' and a column named
        # 'MyFeature' exists in the DataFrame, PandasData will typically map it if 'MyFeature'
        # is a declared line and a parameter self.p.MyFeature = 'MyFeature' exists.
        # We add such mappings to kwargs if not already present.

        # Get default parameter names from the base PandasData class
        # (e.g., 'datetime', 'open', 'high', etc.)
        _base_param_names = dict(bt.feeds.PandasData.params._getdefaults()).keys()

        for f_col in _custom_feature_cols:
            # If the feature column name is not already a standard parameter
            # and not already explicitly passed in kwargs (which is unlikely for custom features).
            if f_col not in _base_param_names and f_col not in kwargs:
                # Add mapping: line_name (f_col) should use DataFrame column_name (f_col)
                kwargs[f_col] = f_col

        # Call the superclass __init__ to complete the feed setup.
        # It will use self.lines and its parameters (populated from the processed kwargs)
        # to connect to the data in the 'dataname' DataFrame.
        super(PandasDataWithFeatures, self).__init__(**kwargs)


class FixedFractionPortfolioSizer(bt.Sizer):
    """
    Sizes trades based on a fixed fraction of the current portfolio value.
    """
    params = (('perc', 0.05),)  # Default 5% of portfolio value per trade

    def _getsizing(self, comminfo, cash, data, isbuy):
        if self.broker.get_value() == 0:  # Avoid division by zero if portfolio value is zero
            return 0
        if data.close[0] == 0:  # Avoid division by zero if price is zero
            return 0

        size = (self.broker.get_value() * self.p.perc) / data.close[0]
        return int(size)