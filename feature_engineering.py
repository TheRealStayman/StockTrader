# feature_engineering.py
import pandas as pd
import numpy as np
import pandas_ta as ta
from fuzzywuzzy import process, fuzz
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import logging
import re
import os  # For main test block
import json  # For main test block

# Import from local modules
from utils import clean_company_name  # Use the centralized version
# For the test block, we'd import data_loader functions if needed
from data_loader import load_stock_info as dl_load_stock_info
from data_loader import load_gdelt_data as dl_load_gdelt_data
from data_loader import load_stock_data as dl_load_stock_data

logger = logging.getLogger(__name__)

# --- Download NLTK data if not present ---
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    logger.info("Downloading VADER lexicon for NLTK...")
    nltk.download('vader_lexicon', quiet=True)


# --- Feature Engineering Functions ---

def add_technical_indicators(stock_df, macd_config=None, bbands_config=None, aroc_config=None, adx_config=None):
    logger.info("Calculating technical indicators...")
    try:
        if not isinstance(stock_df.index, pd.MultiIndex) or stock_df.index.nlevels != 2:
            logger.error(f"stock_df.index is not a 2-level MultiIndex at the start. Index: {stock_df.index[:5]}")
            return stock_df
        if not stock_df.index.is_monotonic_increasing:
            logger.warning("Stock data index is not sorted. Sorting for TA calculations...")
            stock_df = stock_df.sort_index()

        grouped_by_ticker = stock_df.groupby(level='Ticker')

        # --- Single-series input indicators (SMA, EMA, RSI, MACD components, BBands components) ---
        logger.debug("Calculating SMA, EMA, RSI...")
        stock_df['SMA_20'] = grouped_by_ticker['Close'].transform(lambda x: ta.sma(x, length=20))
        stock_df['EMA_20'] = grouped_by_ticker['Close'].transform(lambda x: ta.ema(x, length=20))
        stock_df['RSI'] = grouped_by_ticker['Close'].transform(lambda x: ta.rsi(x, length=14))

        logger.debug("Calculating MACD...")
        macd_cfg = macd_config if macd_config else {}
        fast = macd_cfg.get('fast', 12);
        slow = macd_cfg.get('slow', 26);
        signal = macd_cfg.get('signal', 9)
        col_macd_line = f'MACD_{fast}_{slow}_{signal}';
        col_macd_signal = f'MACDs_{fast}_{slow}_{signal}'
        stock_df['MACD_line'] = grouped_by_ticker['Close'].transform(
            lambda x: ta.macd(x, fast=fast, slow=slow, signal=signal)[col_macd_line])
        stock_df['MACD_signal'] = grouped_by_ticker['Close'].transform(
            lambda x: ta.macd(x, fast=fast, slow=slow, signal=signal)[col_macd_signal])

        logger.debug("Calculating Bollinger Bands...")
        bbands_cfg = bbands_config if bbands_config else {}
        bb_len = bbands_cfg.get('length', 20);
        bb_std = float(bbands_cfg.get('std', 2.0))
        col_bbl = f'BBL_{bb_len}_{bb_std}';
        col_bbm = f'BBM_{bb_len}_{bb_std}';
        col_bbu = f'BBU_{bb_len}_{bb_std}'
        stock_df['BB_lower'] = grouped_by_ticker['Close'].transform(
            lambda x: ta.bbands(x, length=bb_len, std=bb_std)[col_bbl])
        stock_df['BB_middle'] = grouped_by_ticker['Close'].transform(
            lambda x: ta.bbands(x, length=bb_len, std=bb_std)[col_bbm])
        stock_df['BB_upper'] = grouped_by_ticker['Close'].transform(
            lambda x: ta.bbands(x, length=bb_len, std=bb_std)[col_bbu])

        # --- Aroon Oscillator ---
        logger.debug("Calculating Aroon Oscillator...")
        aroon_cfg = aroc_config if aroc_config else {}
        aroon_len = aroon_cfg.get('length', 14)
        aroon_osc_col = f'AROONOSC_{aroon_len}'

        aroon_values_raw = grouped_by_ticker.apply(
            lambda group: ta.aroon(high=group['High'], low=group['Low'], length=aroon_len)[aroon_osc_col].rename(None)
        )
        logger.debug(
            f"Aroon: Raw aroon_values_raw. Empty: {aroon_values_raw.empty}, Index type: {type(aroon_values_raw.index)}, nlevels: {getattr(aroon_values_raw.index, 'nlevels', 'N/A')}, names: {list(aroon_values_raw.index.names)}, head:\n{aroon_values_raw.head()}")

        processed_aroon_values = None
        if not aroon_values_raw.empty and isinstance(aroon_values_raw.index, pd.MultiIndex):
            actual_nlevels = aroon_values_raw.index.nlevels
            actual_names = list(aroon_values_raw.index.names)

            if actual_nlevels == 3 and actual_names == ['Ticker', 'Date', 'Ticker']:
                logger.warning(
                    f"Aroon: apply() result has 3 levels with names {actual_names}. Attempting to drop the last level (level 2).")
                try:
                    # The levels are (Ticker_group, Date_series, Ticker_series_index_if_present)
                    # We want to keep the first Ticker (grouping key) and Date. Drop the 3rd.
                    processed_aroon_values = aroon_values_raw.droplevel(2)
                    logger.debug(
                        f"Aroon: After dropping level 2. Index type: {type(processed_aroon_values.index)}, nlevels: {getattr(processed_aroon_values.index, 'nlevels', 'N/A')}, names: {list(processed_aroon_values.index.names)}")
                except Exception as e_drop:
                    logger.error(f"Aroon: Failed to drop level 2 from 3-level index: {e_drop}")
            elif actual_nlevels == 2:
                logger.debug("Aroon: apply() result has 2 levels as expected.")
                processed_aroon_values = aroon_values_raw
            else:
                logger.error(f"Aroon: apply() result has unexpected nlevels: {actual_nlevels} or names: {actual_names}")
        elif not aroon_values_raw.empty:
            logger.error(
                f"Aroon: apply() result is not a MultiIndex but not empty. Type: {type(aroon_values_raw.index)}")

        if processed_aroon_values is not None and not processed_aroon_values.empty and \
                isinstance(processed_aroon_values.index, pd.MultiIndex) and processed_aroon_values.index.nlevels == 2:

            # At this point, processed_aroon_values.index should be (Ticker, Date)
            # Set names explicitly to ensure they are correct before swapping
            processed_aroon_values.index.set_names(['Ticker', 'Date'], inplace=True)

            # Swap levels to get (Date, Ticker) for assignment
            series_to_assign = processed_aroon_values.swaplevel('Ticker', 'Date').sort_index()
            # series_to_assign.index is now (Date, Ticker) with names ['Date', 'Ticker']

            logger.debug(
                f"Aroon: After processing and swap. series_to_assign.index type: {type(series_to_assign.index)}, nlevels: {getattr(series_to_assign.index, 'nlevels', 'N/A')}, names: {list(series_to_assign.index.names)}")
            stock_df[aroon_osc_col] = series_to_assign
            logger.debug(f"Aroon oscillator '{aroon_osc_col}' assigned.")
        else:
            logger.error(
                f"Aroon: Could not process aroon_values to a 2-level MI. Assigning NA. Final processed_aroon_values details: empty={processed_aroon_values is None or processed_aroon_values.empty}, nlevels={getattr(getattr(processed_aroon_values, 'index', None), 'nlevels', 'N/A')}")
            stock_df[aroon_osc_col] = pd.NA

        # --- ADX --- (Apply similar logic)
        logger.debug("Calculating ADX...")
        adx_cfg = adx_config if adx_config else {}
        adx_len = adx_cfg.get('length', 14)
        adx_col = f'ADX_{adx_len}'

        adx_values_raw = grouped_by_ticker.apply(
            lambda group: ta.adx(high=group['High'], low=group['Low'], close=group['Close'], length=adx_len)[
                adx_col].rename(None)
        )
        logger.debug(
            f"ADX: Raw adx_values_raw. Empty: {adx_values_raw.empty}, Index type: {type(adx_values_raw.index)}, nlevels: {getattr(adx_values_raw.index, 'nlevels', 'N/A')}, names: {list(adx_values_raw.index.names)}, head:\n{adx_values_raw.head()}")

        processed_adx_values = None
        if not adx_values_raw.empty and isinstance(adx_values_raw.index, pd.MultiIndex):
            actual_nlevels_adx = adx_values_raw.index.nlevels
            actual_names_adx = list(adx_values_raw.index.names)

            if actual_nlevels_adx == 3 and actual_names_adx == ['Ticker', 'Date', 'Ticker']:
                logger.warning(
                    f"ADX: apply() result has 3 levels with names {actual_names_adx}. Attempting to drop the last level (level 2).")
                try:
                    processed_adx_values = adx_values_raw.droplevel(2)
                    logger.debug(
                        f"ADX: After dropping level 2. Index type: {type(processed_adx_values.index)}, nlevels: {getattr(processed_adx_values.index, 'nlevels', 'N/A')}, names: {list(processed_adx_values.index.names)}")
                except Exception as e_drop_adx:
                    logger.error(f"ADX: Failed to drop level 2 from 3-level index: {e_drop_adx}")
            elif actual_nlevels_adx == 2:
                logger.debug("ADX: apply() result has 2 levels as expected.")
                processed_adx_values = adx_values_raw
            else:
                logger.error(
                    f"ADX: apply() result has unexpected nlevels: {actual_nlevels_adx} or names: {actual_names_adx}")
        elif not adx_values_raw.empty:
            logger.error(f"ADX: apply() result is not a MultiIndex but not empty. Type: {type(adx_values_raw.index)}")

        if processed_adx_values is not None and not processed_adx_values.empty and \
                isinstance(processed_adx_values.index, pd.MultiIndex) and processed_adx_values.index.nlevels == 2:
            processed_adx_values.index.set_names(['Ticker', 'Date'], inplace=True)
            series_to_assign_adx = processed_adx_values.swaplevel('Ticker', 'Date').sort_index()
            logger.debug(
                f"ADX: After processing and swap. series_to_assign_adx.index type: {type(series_to_assign_adx.index)}, nlevels: {getattr(series_to_assign_adx.index, 'nlevels', 'N/A')}, names: {list(series_to_assign_adx.index.names)}")
            stock_df[adx_col] = series_to_assign_adx
            logger.debug(f"ADX '{adx_col}' assigned.")
        else:
            logger.error(
                f"ADX: Could not process adx_values to a 2-level MI. Assigning NA. Final processed_adx_values details: empty={processed_adx_values is None or processed_adx_values.empty}, nlevels={getattr(getattr(processed_adx_values, 'index', None), 'nlevels', 'N/A')}")
            stock_df[adx_col] = pd.NA

        logger.info("Technical indicators calculation process completed.")
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}", exc_info=True)
    return stock_df


def link_events_to_stocks(gdelt_df, stock_info_df, threshold=85):
    logger.info("Linking GDELT events to stock tickers...")
    if 'CompanyName_Clean' not in stock_info_df.columns or 'Ticker' not in stock_info_df.columns:
        logger.error("Stock info DataFrame missing 'CompanyName_Clean' or 'Ticker'.")
        return None  # Or gdelt_df with Ticker=None
    if 'Actor1Name_Clean' not in gdelt_df.columns and 'Actor2Name_Clean' not in gdelt_df.columns:
        logger.error("GDELT DataFrame missing cleaned Actor names.")
        return None  # Or gdelt_df with Ticker=None

    # Create a mapping from cleaned company name to Ticker
    # Handle potential duplicate CompanyName_Clean if they map to different tickers (less likely but possible)
    # For simplicity, using the first ticker found for a clean name if duplicates exist.
    company_mapping = stock_info_df.groupby('CompanyName_Clean')['Ticker'].first().to_dict()
    company_choices = list(company_mapping.keys())

    if not company_choices:
        logger.warning("No company names available in stock_info_df for matching.")
        gdelt_df['Ticker'] = pd.NA  # Use pandas NA for consistency
        return gdelt_df

    match_results = []
    for index, row in gdelt_df.iterrows():
        matched_tickers = set()
        for actor_col in ['Actor1Name_Clean', 'Actor2Name_Clean']:
            if pd.notna(row.get(actor_col)) and row[actor_col]:  # Check if column exists and has value
                # Using fuzz.token_set_ratio which is good for matching strings with different word order
                match = process.extractOne(row[actor_col], company_choices, scorer=fuzz.token_set_ratio,
                                           score_cutoff=threshold)
                if match:
                    matched_tickers.add(company_mapping[match[0]])

        if matched_tickers:
            for ticker in matched_tickers:
                new_row = row.to_dict()  # Convert Series to dict before updating
                new_row['Ticker'] = ticker
                match_results.append(new_row)
        # else: # Optionally keep events with no match
        #     new_row = row.to_dict()
        #     new_row['Ticker'] = pd.NA
        #     match_results.append(new_row)

    if not match_results:
        logger.warning("No GDELT events could be linked to stocks based on current settings.")
        gdelt_df['Ticker'] = pd.NA
        return gdelt_df  # Return original with NA Ticker column

    linked_gdelt_df = pd.DataFrame(match_results)
    logger.info(f"GDELT events linked. Original: {len(gdelt_df)}, Linked instances: {len(linked_gdelt_df)}")

    # Drop temporary clean name columns if they were added by this specific function and not needed elsewhere
    # Assuming Actor1Name_Clean, Actor2Name_Clean came from data_loader.py, so keep them.
    return linked_gdelt_df


def add_nlp_features(gdelt_df):
    logger.info("Adding NLP sentiment features (VADER)...")
    if 'Title From URL' not in gdelt_df.columns:
        logger.warning("Column 'Title From URL' not found. Skipping NLP sentiment.")
        gdelt_df['title_sentiment_vader'] = 0.0
        return gdelt_df

    analyzer = SentimentIntensityAnalyzer()

    # Ensure 'Title From URL' is string type, fill NA with empty string for apply
    gdelt_df['Title From URL'] = gdelt_df['Title From URL'].astype(str).fillna('')

    def get_vader_sentiment(text):
        if text and text.strip():  # Check if text is not empty or just whitespace
            try:
                return analyzer.polarity_scores(text)['compound']
            except Exception as e:
                logger.warning(f"VADER failed for text: '{text[:50]}...'. Error: {e}")
                return 0.0
        return 0.0

    gdelt_df['title_sentiment_vader'] = gdelt_df['Title From URL'].apply(get_vader_sentiment)
    logger.info("Finished adding VADER sentiment scores.")
    return gdelt_df


def aggregate_gdelt_features(gdelt_df):
    logger.info("Aggregating GDELT features per stock per day...")
    if gdelt_df is None or gdelt_df.empty or 'Ticker' not in gdelt_df.columns or gdelt_df['Ticker'].isna().all():
        logger.warning("Cannot aggregate GDELT: DataFrame is empty or 'Ticker' column missing/all NA.")
        # Return empty DataFrame with expected columns for graceful merge
        return pd.DataFrame(columns=['Date', 'Ticker', 'gdelt_event_count',
                                     'gdelt_goldstein_mean', 'gdelt_tone_mean',
                                     'gdelt_title_sentiment_mean'])  # Add more as defined below

    # Ensure Date is datetime and normalized
    gdelt_df['Date'] = pd.to_datetime(gdelt_df['Date']).dt.normalize()

    agg_funcs = {
        'GLOBALEVENTID': 'count',
        'GoldsteinScale': ['mean', 'sum', 'min', 'max'],
        'AvgTone': ['mean', 'sum'],
        'NumArticles': 'sum',
    }
    if 'title_sentiment_vader' in gdelt_df.columns:
        agg_funcs['title_sentiment_vader'] = 'mean'

    try:
        aggregated_gdelt = gdelt_df.groupby(['Date', 'Ticker']).agg(agg_funcs)
        aggregated_gdelt.columns = ['_'.join(col).strip('_') for col in aggregated_gdelt.columns.values]

        # Standardize column names
        rename_map = {
            'GLOBALEVENTID_count': 'gdelt_event_count',
            'GoldsteinScale_mean': 'gdelt_goldstein_mean', 'GoldsteinScale_sum': 'gdelt_goldstein_sum',
            'GoldsteinScale_min': 'gdelt_goldstein_min', 'GoldsteinScale_max': 'gdelt_goldstein_max',
            'AvgTone_mean': 'gdelt_tone_mean', 'AvgTone_sum': 'gdelt_tone_sum',
            'NumArticles_sum': 'gdelt_num_articles_sum',
        }
        if 'title_sentiment_vader_mean' in aggregated_gdelt.columns:
            rename_map['title_sentiment_vader_mean'] = 'gdelt_title_sentiment_mean'

        aggregated_gdelt = aggregated_gdelt.rename(columns=rename_map).reset_index()
        logger.info(f"Aggregated GDELT features. Shape: {aggregated_gdelt.shape}")
        return aggregated_gdelt
    except Exception as e:
        logger.error(f"Error aggregating GDELT features: {e}", exc_info=True)
        # Return empty structure if aggregation fails
        return pd.DataFrame(columns=['Date', 'Ticker'] + list(rename_map.values()))


def merge_data(stock_df, gdelt_features_df):
    logger.info("Merging stock data with GDELT features...")
    if stock_df is None or stock_df.empty:
        logger.error("Stock DataFrame is empty, cannot merge.")
        return None  # Or return gdelt_features_df if that's desired

    # Ensure stock_df has Date and Ticker as index or columns
    if isinstance(stock_df.index, pd.MultiIndex) and {'Date', 'Ticker'}.issubset(stock_df.index.names):
        stock_df_reset = stock_df.reset_index()
    elif {'Date', 'Ticker'}.issubset(stock_df.columns):
        stock_df_reset = stock_df.copy()
    else:
        logger.error("Stock DataFrame for merge must contain 'Date' and 'Ticker' as index or columns.")
        return None

    stock_df_reset['Date'] = pd.to_datetime(stock_df_reset['Date'])

    if gdelt_features_df is None or gdelt_features_df.empty:
        logger.warning("GDELT features DataFrame is empty. Returning stock data only.")
        # Set index for consistency if it was reset
        return stock_df_reset.set_index(['Date', 'Ticker']).sort_index()

    gdelt_features_df['Date'] = pd.to_datetime(gdelt_features_df['Date'])

    try:
        merged_df = pd.merge(stock_df_reset, gdelt_features_df, on=['Date', 'Ticker'], how='left')

        # Fill NaNs for GDELT columns that resulted from the left merge
        gdelt_cols = [col for col in gdelt_features_df.columns if col not in ['Date', 'Ticker']]
        fill_values = {col: 0 for col in gdelt_cols}  # Assuming 0 is a neutral fill for GDELT features
        merged_df.fillna(fill_values, inplace=True)

        # Set index
        merged_df.set_index(['Date', 'Ticker'], inplace=True)
        merged_df.sort_index(inplace=True)

        logger.info(f"Successfully merged data. Shape: {merged_df.shape}")
        return merged_df
    except Exception as e:
        logger.error(f"Error merging dataframes: {e}", exc_info=True)
        return None  # Or stock_df_reset.set_index(['Date', 'Ticker'])


def create_lagged_features(df, feature_cols_to_lag, lag_periods):
    logger.info(f"Creating lagged features for periods: {lag_periods}")
    if not isinstance(df.index, pd.MultiIndex) or not {'Date', 'Ticker'}.issubset(df.index.names):
        logger.error("DataFrame must have a MultiIndex with 'Date' and 'Ticker' for lagging.")
        return df

    df_copy = df.sort_index()  # Ensure correct sort order within groups

    for col in feature_cols_to_lag:
        if col in df_copy.columns:
            for lag in lag_periods:
                df_copy[f'{col}_lag_{lag}'] = df_copy.groupby(level='Ticker')[col].shift(lag)
        else:
            logger.warning(f"Column '{col}' not found for lagging.")

    logger.info("Lagged features created.")
    return df_copy


def define_target_variable(df, target_col_name='Target', horizon=1):
    logger.info(f"Defining target variable '{target_col_name}' (price increase in {horizon} day(s)).")
    if 'Close' not in df.columns:
        logger.error("Column 'Close' not found, cannot define target variable.")
        return df
    if not isinstance(df.index, pd.MultiIndex) or not {'Date', 'Ticker'}.issubset(df.index.names):
        logger.error("DataFrame must have a MultiIndex with 'Date' and 'Ticker' for target definition.")
        return df

    df_copy = df.sort_index()
    df_copy['Future_Close'] = df_copy.groupby(level='Ticker')['Close'].shift(-horizon)
    df_copy[target_col_name] = (df_copy['Future_Close'] > df_copy['Close']).astype(int)

    # Rows where Future_Close is NaN (end of series for each ticker) will have NaN Target.
    # These should be dropped before training.
    # df_copy.dropna(subset=[target_col_name], inplace=True) # Drop them here or later
    df_copy.drop(columns=['Future_Close'], inplace=True)

    logger.info(f"Target variable '{target_col_name}' created.")
    return df_copy


if __name__ == "__main__":
    # This block is for basic testing of the feature engineering pipeline.
    # Uses data_loader functions directly.
    logger.info("--- Testing Feature Engineering Pipeline (Example Usage) ---")

    # Define paths for test data (ensure data_loader.py's test block creates these or use actual small sample files)
    test_stock_path = 'data/stock_data.csv'
    test_gdelt_path = 'data/events.json'
    test_info_path = 'data/Stock Information.csv'
    output_dir = 'processed_data_test'  # Use a test output directory
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Data (using functions from data_loader.py)
    # Ensure dummy files are created by data_loader.py's __main__ or place sample files
    stock_info = dl_load_stock_info(test_info_path)
    gdelt_data = dl_load_gdelt_data(test_gdelt_path)
    stock_data = dl_load_stock_data(test_stock_path)  # This is OHLCV

    if stock_data is not None and gdelt_data is not None and stock_info is not None:
        # 2. Link GDELT Events
        gdelt_linked = link_events_to_stocks(gdelt_data, stock_info)
        if gdelt_linked is None or gdelt_linked.empty:
            logger.warning("Test: GDELT linking failed or produced no results.")
            gdelt_linked = pd.DataFrame()  # Ensure it's an empty df for next steps

        # 3. Add NLP Features
        gdelt_nlp = add_nlp_features(gdelt_linked.copy() if not gdelt_linked.empty else pd.DataFrame())

        # 4. Aggregate GDELT
        gdelt_agg = aggregate_gdelt_features(gdelt_nlp.copy() if not gdelt_nlp.empty else pd.DataFrame())

        # 5. Add Technical Indicators
        stock_data_with_ta = add_technical_indicators(stock_data.copy())

        # 6. Merge
        merged_data = merge_data(stock_data_with_ta, gdelt_agg)

        if merged_data is not None and not merged_data.empty:
            # 7. Define Target
            data_with_target = define_target_variable(merged_data.copy(), horizon=1)

            # 8. Lagged Features
            # Define a small, fixed list of features to lag for testing
            features_to_lag_example = ['Close', 'Volume', 'SMA_20', 'RSI_14']
            if 'gdelt_tone_mean' in data_with_target.columns:
                features_to_lag_example.append('gdelt_tone_mean')

            final_features_df = create_lagged_features(data_with_target, features_to_lag_example, [1, 3])
            final_features_df.dropna(inplace=True)  # Drop NaNs from lagging

            logger.info(f"\n--- Test: Final Features DataFrame Sample ---\n{final_features_df.head()}")
            logger.info(f"Test: Final Features Shape: {final_features_df.shape}")

            # Optional: Save test output
            # final_features_df.reset_index().to_parquet(os.path.join(output_dir, 'test_final_features.parquet'))
            # logger.info(f"Test: Final features saved to {os.path.join(output_dir, 'test_final_features.parquet')}")
        else:
            logger.error("Test: Merging failed or resulted in empty data.")
    else:
        logger.error("Test: Failed to load one or more initial data sources for feature engineering test.")