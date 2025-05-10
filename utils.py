# utils.py
import pandas as pd
import joblib
import logging
import backtrader as bt
import re

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


class PandasDataWithFeatures(bt.feeds.PandasData):
    """
    Custom Backtrader data feed to include feature columns.
    """

    # `lines` will be set dynamically based on feature_cols passed to the instance
    # `params` will also be set dynamically

    def __init__(self, feature_cols=None, **kwargs):
        if feature_cols is None:
            feature_cols = []

        self.lines = tuple(feature_cols)
        self.params = tuple(
            [(col, -1) for col in feature_cols]  # Map each feature column name to a line
            + [('datetime', None),  # Use the DataFrame index for datetime
               ('open', 'Open'),  # Map DataFrame 'Open' column to 'open' line
               ('high', 'High'),  # Map DataFrame 'High' column to 'high' line
               ('low', 'Low'),  # Map DataFrame 'Low' column to 'low' line
               ('close', 'Close'),  # Map DataFrame 'Close' column to 'close' line
               ('volume', 'Volume'),  # Map DataFrame 'Volume' column to 'volume' line
               ('openinterest', -1)]  # Use -1 if 'openinterest' column doesn't exist
        )
        super(PandasDataWithFeatures, self).__init__(**kwargs)
        # Dynamically add lines for features
        for col in feature_cols:
            self.lines.append(col)  # This might need adjustment based on Backtrader's internals
            # for adding lines post __init__. A common pattern is to define all possible lines
            # and then map them in params.
            # For dynamic features, ensure the data feed creation is robust.
            # Backtrader expects lines to be class attributes.
            # A more robust approach is to pre-define a max number of feature lines
            # or structure the data feed to accept a list of features that it maps.


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