# feature_engineering.py

import pandas as pd
import numpy as np
import pandas_ta as ta # Requires installation: pip install pandas_ta
from fuzzywuzzy import process, fuzz # Requires installation: pip install fuzzywuzzy python-Levenshtein
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import logging
import re

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Download NLTK data if not present ---
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    logging.info("Downloading VADER lexicon for NLTK...")
    nltk.download('vader_lexicon')

# --- Helper Functions ---
def clean_company_name_for_matching(name):
    """ Cleans company names for better fuzzy matching. """
    if pd.isna(name):
        return ""
    name = str(name).lower()
    # Remove common suffixes and punctuation
    name = re.sub(r'[,.\(\)]', '', name)
    name = re.sub(r'\b(inc|corp|corporation|ltd|llc|plc)\b', '', name, flags=re.IGNORECASE)
    return name.strip()

# --- Feature Engineering Functions ---

def add_technical_indicators(stock_df):
    """
    Calculates and adds technical indicators to the stock DataFrame.

    Args:
        stock_df (pd.DataFrame): DataFrame with columns 'Open', 'High', 'Low', 'Close', 'Volume',
                                 and indexed by Date and Ticker (MultiIndex).

    Returns:
        pd.DataFrame: DataFrame with added technical indicator columns.
    """
    logging.info("Calculating technical indicators...")
    # Ensure data is sorted by date within each ticker group for TA calculations
    stock_df = stock_df.sort_index(level=['Ticker', 'Date'])

    # Use pandas_ta library
    try:
        stock_df.groupby(level='Ticker').agg(
            {'Close': lambda x: ta.sma(x, length=20),
             'High': lambda x: ta.sma(x, length=50)} # Use dummy High col just to pass data
        ) # Pre-run to ensure group-by context is understood by pandas_ta

        # Apply indicators using groupby().apply() to handle multi-index correctly
        stock_df['SMA_20'] = stock_df.groupby(level='Ticker')['Close'].apply(lambda x: ta.sma(x, length=20))
        stock_df['SMA_50'] = stock_df.groupby(level='Ticker')['Close'].apply(lambda x: ta.sma(x, length=50))
        stock_df['RSI_14'] = stock_df.groupby(level='Ticker')['Close'].apply(lambda x: ta.rsi(x, length=14))

        # MACD requires applying to the group and merging results
        macd = stock_df.groupby(level='Ticker')['Close'].apply(lambda x: ta.macd(x, fast=12, slow=26, signal=9))
        stock_df = stock_df.join(macd) # Joins MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9

        # Bollinger Bands
        bbands = stock_df.groupby(level='Ticker')['Close'].apply(lambda x: ta.bbands(x, length=20, std=2))
        stock_df = stock_df.join(bbands) # Joins BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0

        # Average True Range (ATR) - needs High, Low, Close
        stock_df['ATR_14'] = stock_df.groupby(level='Ticker').apply(
            lambda x: ta.atr(x['High'], x['Low'], x['Close'], length=14)
        ).reset_index(level=0, drop=True) # Reset index to align results properly

        # On-Balance Volume (OBV) - needs Close and Volume
        stock_df['OBV'] = stock_df.groupby(level='Ticker').apply(
            lambda x: ta.obv(x['Close'], x['Volume'])
        ).reset_index(level=0, drop=True)

        # Add more indicators as desired...

        logging.info(f"Added technical indicators. Columns: {stock_df.columns.tolist()}")

    except Exception as e:
        logging.error(f"Error calculating technical indicators: {e}", exc_info=True)
        # Decide how to handle errors, e.g., return original df or raise exception
        # For now, we'll continue with potentially fewer indicators

    return stock_df

def link_events_to_stocks(gdelt_df, stock_info_df, threshold=85):
    """
    Links GDELT events to stock tickers based on fuzzy matching of Actor names
    against company names. Creates a 'Ticker' column in gdelt_df.

    Args:
        gdelt_df (pd.DataFrame): DataFrame with GDELT event data, including
                                'Actor1Name_Clean' and 'Actor2Name_Clean'.
        stock_info_df (pd.DataFrame): DataFrame with stock info, including
                                      'Ticker' and 'CompanyName_Clean'.
        threshold (int): The minimum fuzzy match score (0-100) to consider a match.

    Returns:
        pd.DataFrame: The gdelt_df with an added 'Ticker' column (or None if fails).
                      An event might be linked to multiple tickers if actors match different companies.
                      This function currently explodes the DataFrame to handle multiple matches.
    """
    logging.info("Linking GDELT events to stock tickers...")
    if 'CompanyName_Clean' not in stock_info_df.columns or 'Ticker' not in stock_info_df.columns:
        logging.error("Stock info DataFrame missing 'CompanyName_Clean' or 'Ticker' column.")
        return None
    if 'Actor1Name_Clean' not in gdelt_df.columns or 'Actor2Name_Clean' not in gdelt_df.columns:
        logging.error("GDELT DataFrame missing 'Actor1Name_Clean' or 'Actor2Name_Clean' column.")
        return None

    company_mapping = stock_info_df.set_index('CompanyName_Clean')['Ticker'].to_dict()
    company_choices = list(company_mapping.keys())

    if not company_choices:
        logging.warning("No company names available in stock_info_df for matching.")
        gdelt_df['Ticker'] = None
        return gdelt_df

    match_results = []

    # Iterate through GDELT events (can be slow for large datasets)
    for index, row in gdelt_df.iterrows():
        matched_tickers = set() # Use a set to avoid duplicate tickers for the same event

        # Check Actor1Name
        if pd.notna(row['Actor1Name_Clean']):
            match1 = process.extractOne(row['Actor1Name_Clean'], company_choices, scorer=fuzz.token_set_ratio, score_cutoff=threshold)
            if match1:
                matched_tickers.add(company_mapping[match1[0]])

        # Check Actor2Name
        if pd.notna(row['Actor2Name_Clean']):
            match2 = process.extractOne(row['Actor2Name_Clean'], company_choices, scorer=fuzz.token_set_ratio, score_cutoff=threshold)
            if match2:
                matched_tickers.add(company_mapping[match2[0]])

        if matched_tickers:
            for ticker in matched_tickers:
                new_row = row.copy()
                new_row['Ticker'] = ticker
                match_results.append(new_row)
        # else:
            # Optionally keep events with no match but with Ticker=None
            # new_row = row.copy()
            # new_row['Ticker'] = None
            # match_results.append(new_row)

    if not match_results:
        logging.warning("No GDELT events could be linked to stocks.")
        # Return the original DataFrame with an empty Ticker column if needed downstream
        gdelt_df['Ticker'] = None
        return gdelt_df


    linked_gdelt_df = pd.DataFrame(match_results)
    logging.info(f"GDELT events linked. Original events: {len(gdelt_df)}, Linked event instances: {len(linked_gdelt_df)}")

    # Drop temporary clean name columns if no longer needed
    linked_gdelt_df = linked_gdelt_df.drop(columns=['Actor1Name_Clean', 'Actor2Name_Clean'], errors='ignore')

    return linked_gdelt_df


def add_nlp_features(gdelt_df):
    """
    Adds sentiment scores to GDELT events based on 'Title From URL'.
    Uses VADER for basic sentiment analysis.

    Args:
        gdelt_df (pd.DataFrame): DataFrame with GDELT event data, including 'Title From URL'.

    Returns:
        pd.DataFrame: DataFrame with added 'title_sentiment_vader' column.
    """
    logging.info("Adding NLP sentiment features (VADER)...")
    if 'Title From URL' not in gdelt_df.columns:
        logging.warning("Column 'Title From URL' not found in GDELT data. Skipping NLP sentiment.")
        gdelt_df['title_sentiment_vader'] = 0.0 # Default neutral value
        return gdelt_df

    analyzer = SentimentIntensityAnalyzer()

    def get_vader_sentiment(text):
        if isinstance(text, str) and text.strip():
            try:
                # Handle potential errors during analysis for specific texts
                return analyzer.polarity_scores(text)['compound']
            except Exception as e:
                logging.warning(f"VADER failed for text: '{text}'. Error: {e}")
                return 0.0 # Return neutral on error
        return 0.0 # Return neutral for empty or non-string titles

    # Apply sentiment analysis only to rows after April 1st, 2013 where titles might exist
    # Note: GDELT data might have empty titles even after this date.
    mask_has_title = (gdelt_df['Date'] >= '2013-04-01') & (gdelt_df['Title From URL'].str.strip() != '')
    gdelt_df['title_sentiment_vader'] = 0.0 # Initialize column with neutral
    gdelt_df.loc[mask_has_title, 'title_sentiment_vader'] = gdelt_df.loc[mask_has_title, 'Title From URL'].apply(get_vader_sentiment)

    logging.info("Finished adding VADER sentiment scores.")
    # Placeholder for more advanced NLP (e.g., FinBERT, NER)
    # Consider adding features like:
    # - FinBERT sentiment (positive, negative, neutral probabilities)
    # - Entity counts (e.g., number of ORG mentions related to the stock)
    # - Topic modeling features

    return gdelt_df

def aggregate_gdelt_features(gdelt_df):
    """
    Aggregates linked GDELT event features per day per stock.

    Args:
        gdelt_df (pd.DataFrame): GDELT DataFrame with 'Date' and 'Ticker' columns
                                 (after linking).

    Returns:
        pd.DataFrame: DataFrame with aggregated GDELT features per Date and Ticker.
                      Returns an empty DataFrame if input is empty or lacks 'Ticker'.
    """
    logging.info("Aggregating GDELT features per stock per day...")
    if gdelt_df is None or gdelt_df.empty or 'Ticker' not in gdelt_df.columns or gdelt_df['Ticker'].isna().all():
        logging.warning("Cannot aggregate GDELT features: Input DataFrame is empty or missing 'Ticker' column.")
        # Return an empty DataFrame with expected columns for graceful merging later
        return pd.DataFrame(columns=['Date', 'Ticker', 'gdelt_event_count', 'gdelt_goldstein_mean',
                                    'gdelt_goldstein_sum', 'gdelt_tone_mean', 'gdelt_tone_sum',
                                    'gdelt_num_articles_sum', 'gdelt_title_sentiment_mean'])

    # Ensure Date is suitable for grouping
    gdelt_df['Date'] = pd.to_datetime(gdelt_df['Date']).dt.normalize()

    # Define aggregation functions
    agg_funcs = {
        'GLOBALEVENTID': 'count', # Count of events
        'GoldsteinScale': ['mean', 'sum', 'min', 'max'],
        'AvgTone': ['mean', 'sum'],
        'NumArticles': 'sum',
    }
    # Add sentiment aggregation if the column exists
    if 'title_sentiment_vader' in gdelt_df.columns:
        agg_funcs['title_sentiment_vader'] = 'mean'

    # Group by Date and Ticker, then aggregate
    try:
        aggregated_gdelt = gdelt_df.groupby(['Date', 'Ticker']).agg(agg_funcs)

        # Flatten multi-level column index
        aggregated_gdelt.columns = ['_'.join(col).strip('_') for col in aggregated_gdelt.columns.values]

        # Rename columns for clarity
        aggregated_gdelt = aggregated_gdelt.rename(columns={
            'GLOBALEVENTID_count': 'gdelt_event_count',
            'GoldsteinScale_mean': 'gdelt_goldstein_mean',
            'GoldsteinScale_sum': 'gdelt_goldstein_sum',
            'GoldsteinScale_min': 'gdelt_goldstein_min',
            'GoldsteinScale_max': 'gdelt_goldstein_max',
            'AvgTone_mean': 'gdelt_tone_mean',
            'AvgTone_sum': 'gdelt_tone_sum',
            'NumArticles_sum': 'gdelt_num_articles_sum',
            'title_sentiment_vader_mean': 'gdelt_title_sentiment_mean' # Adjust if using different sentiment column
        })

        aggregated_gdelt = aggregated_gdelt.reset_index()
        logging.info(f"Aggregated GDELT features. Shape: {aggregated_gdelt.shape}")
        return aggregated_gdelt

    except Exception as e:
        logging.error(f"Error aggregating GDELT features: {e}", exc_info=True)
        return pd.DataFrame(columns=['Date', 'Ticker'] + list(agg_funcs.keys())) # Return empty structure


def merge_data(stock_df, gdelt_features_df):
    """
    Merges stock data (with technical indicators) and aggregated GDELT features.

    Args:
        stock_df (pd.DataFrame): DataFrame with stock prices and technical indicators,
                                 indexed by Date and Ticker.
        gdelt_features_df (pd.DataFrame): DataFrame with aggregated GDELT features,
                                          with 'Date' and 'Ticker' columns.

    Returns:
        pd.DataFrame: Merged DataFrame containing stock data and GDELT features.
    """
    logging.info("Merging stock data with GDELT features...")
    if stock_df is None or stock_df.empty:
        logging.error("Stock DataFrame is empty, cannot merge.")
        return None
    if gdelt_features_df is None or gdelt_features_df.empty:
        logging.warning("GDELT features DataFrame is empty. Merging stock data only.")
        # Ensure stock_df index is reset if it's not already
        if isinstance(stock_df.index, pd.MultiIndex):
             stock_df = stock_df.reset_index()
        return stock_df

    try:
        # Ensure date columns are datetime objects and Ticker is present
        stock_df_reset = stock_df.reset_index()
        stock_df_reset['Date'] = pd.to_datetime(stock_df_reset['Date'])
        gdelt_features_df['Date'] = pd.to_datetime(gdelt_features_df['Date'])

        # Perform a left merge to keep all stock data rows
        merged_df = pd.merge(stock_df_reset, gdelt_features_df, on=['Date', 'Ticker'], how='left')

        # Fill NaN values created by the merge (for days/stocks with no GDELT features)
        # Identify GDELT feature columns dynamically
        gdelt_cols = [col for col in gdelt_features_df.columns if col not in ['Date', 'Ticker']]
        fill_values = {col: 0 for col in gdelt_cols} # Fill with 0, assuming 0 represents neutral/no event impact
        merged_df.fillna(fill_values, inplace=True)

        # Set index back to Date (or keep as columns if preferred for ML)
        merged_df.set_index('Date', inplace=True)
        merged_df.sort_index(inplace=True)

        logging.info(f"Successfully merged data. Shape: {merged_df.shape}")
        return merged_df

    except Exception as e:
        logging.error(f"Error merging dataframes: {e}", exc_info=True)
        return None


def create_lagged_features(df, feature_cols, lag_periods):
    """
    Creates lagged features for specified columns within each ticker group.

    Args:
        df (pd.DataFrame): DataFrame containing the features and a 'Ticker' column.
                           Must be sorted by Ticker and Date.
        feature_cols (list): List of column names to create lags for.
        lag_periods (list): List of integers representing the lag periods (e.g., [1, 2, 3, 5]).

    Returns:
        pd.DataFrame: DataFrame with original and new lagged features.
    """
    logging.info(f"Creating lagged features for periods: {lag_periods}")
    df_copy = df.sort_index(level=['Ticker', 'Date']) # Ensure correct sorting for shift

    for col in feature_cols:
        if col in df_copy.columns:
            for lag in lag_periods:
                df_copy[f'{col}_lag_{lag}'] = df_copy.groupby(level='Ticker')[col].shift(lag)
        else:
            logging.warning(f"Column '{col}' not found for lagging.")

    logging.info("Lagged features created.")
    return df_copy


def define_target_variable(df, horizon=1):
    """
    Defines the target variable for prediction.
    Example: Binary classification - predicts if the price will go up tomorrow.

    Args:
        df (pd.DataFrame): DataFrame with 'Close' prices, indexed by Date and Ticker.
        horizon (int): Number of days ahead to predict (default: 1).

    Returns:
        pandas.DataFrame: DataFrame with the added 'Target' column.
                          Rows with NaN targets (end of series) are dropped.
    """
    logging.info(f"Defining target variable (price increase in {horizon} day(s)).")
    df_copy = df.sort_index(level=['Ticker', 'Date']) # Ensure correct sorting

    # Calculate future close price
    df_copy['Future_Close'] = df_copy.groupby(level='Ticker')['Close'].shift(-horizon)

    # Define target: 1 if Future_Close > Close, 0 otherwise
    df_copy['Target'] = (df_copy['Future_Close'] > df_copy['Close']).astype(int)

    # Remove rows where target cannot be calculated (last 'horizon' rows per ticker)
    df_copy.dropna(subset=['Target'], inplace=True)
    df_copy.drop(columns=['Future_Close'], inplace=True) # Remove intermediate column

    logging.info("Target variable 'Target' created.")
    return df_copy

# --- Main Execution Block (Example Usage) ---
if __name__ == "__main__":
    # NOTE: Replace with actual file paths
    STOCK_PRICES_FILE = 'path/to/your/stock_data.csv' # NEED TO PROVIDE
    GDELT_FILE = 'path/to/your/gdelt_events.json'       # NEED TO PROVIDE
    STOCK_INFO_FILE = 'path/to/your/stock_info.csv'   # NEED TO PROVIDE

    if not all([os.path.exists(p) for p in [STOCK_PRICES_FILE, GDELT_FILE, STOCK_INFO_FILE]]):
         print("-" * 50)
         print("ERROR: One or more input files not found.")
         print("Please update the placeholder paths in feature_engineering.py:")
         print(f"STOCK_PRICES_FILE = '{STOCK_PRICES_FILE}'")
         print(f"GDELT_FILE = '{GDELT_FILE}'")
         print(f"STOCK_INFO_FILE = '{STOCK_INFO_FILE}'")
         print("-" * 50)
         # Create dummy files for demonstration if they don't exist
         # In a real scenario, you would stop execution here or handle appropriately
         print("Creating dummy data files for demonstration purposes...")
         # Create dummy stock info
         stock_info_data = {'Ticker': ['AAPL', 'MSFT'], 'Company': ['Apple Inc', 'Microsoft Corp'], 'GICS Sector': ['Technology', 'Technology'], 'GICS Sub-Industry': ['Hardware', 'Software']}
         pd.DataFrame(stock_info_data).to_csv(STOCK_INFO_FILE, index=False)

         # Create dummy GDELT data
         gdelt_dummy_data = {
             "2023-01-01": {
                 "Event 1": { "GLOBALEVENTID": 1, "SQLDATE": 20230101, "Actor1Name": "APPLE", "Actor2Name": "CHINA", "EventCode": "042", "GoldsteinScale": 5.0, "NumArticles": 10, "AvgTone": 2.5, "SOURCEURL": "http://example.com/1", "Title From URL": "Apple discusses production in China"},
                 "Event 2": { "GLOBALEVENTID": 2, "SQLDATE": 20230101, "Actor1Name": "MICROSOFT", "Actor2Name": None, "EventCode": "010", "GoldsteinScale": 1.0, "NumArticles": 5, "AvgTone": -1.0, "SOURCEURL": "http://example.com/2", "Title From URL": "Microsoft announces new software"}
             },
             "2023-01-02": {
                  "Event 1": { "GLOBALEVENTID": 3, "SQLDATE": 20230102, "Actor1Name": "USA", "Actor2Name": "APPLE INC", "EventCode": "051", "GoldsteinScale": -2.0, "NumArticles": 20, "AvgTone": -3.5, "SOURCEURL": "http://example.com/3", "Title From URL": "US government questions Apple policies"}
              }
         }
         with open(GDELT_FILE, 'w') as f:
             json.dump(gdelt_dummy_data, f, indent=4)

         # Create dummy stock data (needs the complex header format described)
         stock_data_content = """Price,Close,Close,High,High,Low,Low,Open,Open,Volume,Volume
Ticker,AAPL,MSFT,AAPL,MSFT,AAPL,MSFT,AAPL,MSFT,AAPL,MSFT
Date,,,,,,,,,,
2023-01-01,170.0,280.0,172.0,282.0,169.0,279.0,171.0,281.0,100000,200000
2023-01-02,171.0,281.0,173.0,283.0,170.0,280.0,170.5,280.5,110000,210000
2023-01-03,172.0,280.5,172.5,282.5,170.5,279.5,171.5,281.5,105000,190000
2023-01-04,171.5,282.0,172.0,283.0,171.0,280.0,172.0,281.0,120000,220000
2023-01-05,173.0,283.0,174.0,284.0,171.0,281.5,171.5,282.0,130000,230000
"""
         with open(STOCK_PRICES_FILE, 'w') as f:
             f.write(stock_data_content)
         print("Dummy files created. Please replace with your actual data.")


    # --- Pipeline Execution ---
    # 1. Load Data
    stock_info = load_stock_info(STOCK_INFO_FILE)
    gdelt_data = load_gdelt_data(GDELT_FILE)
    stock_data = load_stock_data(STOCK_PRICES_FILE)

    if stock_data is not None and gdelt_data is not None and stock_info is not None:
        # 2. Link GDELT Events to Stocks
        gdelt_linked = link_events_to_stocks(gdelt_data, stock_info)

        # 3. Add NLP Features to GDELT
        gdelt_nlp = add_nlp_features(gdelt_linked)

        # 4. Aggregate GDELT Features
        gdelt_agg = aggregate_gdelt_features(gdelt_nlp)

        # 5. Add Technical Indicators to Stock Data
        stock_data_with_ta = add_technical_indicators(stock_data.copy()) # Use copy to avoid modifying original

        # 6. Merge Stock Data with GDELT Features
        # Reset index for stock_data_with_ta if it's still multi-index before merge
        if isinstance(stock_data_with_ta.index, pd.MultiIndex):
             stock_data_with_ta = stock_data_with_ta.reset_index()

        merged_data = merge_data(stock_data_with_ta, gdelt_agg)

        if merged_data is not None:
             # Reset index again if merge_data set it back
             if 'Date' in merged_data.columns:
                  merged_data.set_index(['Date', 'Ticker'], inplace=True)

             # 7. Define Target Variable
             data_with_target = define_target_variable(merged_data.copy(), horizon=1) # Use copy

             # 8. Create Lagged Features
             features_to_lag = [col for col in data_with_target.columns if col not in ['Open', 'High', 'Low', 'Close', 'Ticker', 'Target', 'Future_Close']]
             # Limit the number of features to lag initially for performance
             features_to_lag = ['Close', 'Volume', 'SMA_20', 'RSI_14', 'gdelt_tone_mean', 'gdelt_event_count']
             lag_periods = [1, 3, 5]
             final_features_df = create_lagged_features(data_with_target, features_to_lag, lag_periods)

             # Drop rows with NaNs created by lagging
             final_features_df.dropna(inplace=True)

             print("\n--- Final Features DataFrame Sample ---")
             print(final_features_df.head())
             print(f"\nFinal Features DataFrame shape: {final_features_df.shape}")
             print("\nFinal Features Columns:")
             print(final_features_df.columns.tolist())

             # Save the final features (optional)
             # final_features_df.reset_index().to_parquet(os.path.join(OUTPUT_DIR, 'final_features.parquet'))
             # logging.info(f"Final feature DataFrame saved to {os.path.join(OUTPUT_DIR, 'final_features.parquet')}")

        else:
            logging.error("Data merging failed.")
    else:
        logging.error("One or more data files failed to load. Feature engineering aborted.")