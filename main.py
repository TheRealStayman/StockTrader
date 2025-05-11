# main.py
import gc

import pandas as pd
import numpy as np
import os
import logging
import joblib  # For loading model if utils.load_model_artefacts is not used directly here

# Import custom modules
import data_loader
import feature_engineering
import model_training  # Contains training and evaluation logic
from trading_strategy import MLTradingStrategy
from utils import (clean_company_name, load_processed_data,
                   PandasDataWithFeatures, FixedFractionPortfolioSizer,
                   load_model_artefacts, downcast_numeric_df)  # Centralized utilities
# For backtesting, import Backtrader and analyzers if not calling backtest.py's function
import backtrader as bt
import backtrader.analyzers as btanalyzers

# --- Configuration ---
# Setup basic logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])  # Add FileHandler for persistence if needed
logger = logging.getLogger(__name__)

# 1. File Paths (Centralized)
# Corrected paths as per user's request
STOCK_PRICES_CSV = 'data/stock_data.csv'
GDELT_JSON = 'data/events.json'
STOCK_INFO_CSV = 'data/Stock Information.csv'  # Corrected path

BASE_OUTPUT_DIR = 'output'  # Main output directory
PROCESSED_DATA_DIR = os.path.join(BASE_OUTPUT_DIR, 'processed_data')
MODEL_DIR = os.path.join(BASE_OUTPUT_DIR, 'models')

PROCESSED_FEATURES_FILE = os.path.join(PROCESSED_DATA_DIR, 'final_features.parquet')
MODEL_FILE = os.path.join(MODEL_DIR, 'gdelt_stock_model.joblib')
MODEL_FEATURES_FILE = os.path.join(MODEL_DIR, 'gdelt_stock_model_features.joblib')  # For saving feature list

# Create directories if they don't exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# 2. Feature Engineering Parameters
LAG_PERIODS = [1, 3, 5, 10]
TARGET_HORIZON = 1
FUZZY_MATCH_THRESHOLD = 85

# 3. Model Training Parameters (LGBM_PARAMS are in model_training.py)
TRAIN_TEST_RATIO = 0.2
MODEL_TARGET_VARIABLE = model_training.TARGET_VARIABLE  # From model_training module

# 4. Backtesting Parameters
BT_INITIAL_CASH = 1000.0
BT_COMMISSION_PER_TRADE = 0.001
BT_SLIPPAGE_PERCENT = 0.0005
BT_POSITION_SIZE_PERCENT = 5.0
BT_BUY_THRESHOLD = 0.55  # Default for MLTradingStrategy
BT_SELL_THRESHOLD = 0.45  # Default for MLTradingStrategy
BT_START_DATE = None  # e.g., '2022-01-01'
BT_END_DATE = None  # e.g., '2023-12-31'

# 5. Control Flags
FORCE_DATA_PROCESSING = False
FORCE_MODEL_TRAINING = False


# --- Main Workflow Function ---
def run_trading_pipeline():
    logger.info("--- Starting Algorithmic Trading Pipeline ---")
    final_features_df = None  # Initialize

    # --- Step 1 & 2: Data Loading and Feature Engineering ---
    final_features_df = None
    if FORCE_DATA_PROCESSING or not os.path.exists(PROCESSED_FEATURES_FILE):
        logger.info("--- Phase: Data Loading & Feature Engineering ---")

        # 1a. Load Raw Data
        stock_info_df = data_loader.load_stock_info(STOCK_INFO_CSV)
        gdelt_df = data_loader.load_gdelt_data(GDELT_JSON)
        stock_ohlcv_df = data_loader.load_stock_data(STOCK_PRICES_CSV)

        if stock_info_df is None or stock_ohlcv_df is None:  # GDELT can be optional initially
            logger.error("Failed to load essential stock or info data. Exiting.")
            return
        if gdelt_df is None:
            logger.warning("GDELT data failed to load. Proceeding without GDELT features initially.")
            gdelt_df = pd.DataFrame()  # Empty df

        # 2a. Link GDELT Events (if GDELT data is available)
        gdelt_linked_df = pd.DataFrame()
        if not gdelt_df.empty and stock_info_df is not None:
            # Ensure CompanyName_Clean is present if not already added by load_stock_info
            if 'CompanyName_Clean' not in stock_info_df.columns and 'CompanyName' in stock_info_df.columns:
                stock_info_df['CompanyName_Clean'] = stock_info_df['CompanyName'].apply(clean_company_name)

            gdelt_linked_df = feature_engineering.link_events_to_stocks(
                gdelt_df, stock_info_df, threshold=FUZZY_MATCH_THRESHOLD
            )
        if gdelt_linked_df is None or gdelt_linked_df.empty:
            logger.warning("GDELT linking failed or produced no matches. No GDELT features will be generated.")
            gdelt_agg_features = pd.DataFrame()  # Empty df for merge
        else:
            # 2b. Add NLP Features
            gdelt_nlp_df = feature_engineering.add_nlp_features(gdelt_linked_df.copy())
            # 2c. Aggregate GDELT Features
            gdelt_agg_features = feature_engineering.aggregate_gdelt_features(gdelt_nlp_df.copy())

        # 2d. Add Technical Indicators to stock_ohlcv_df
        # Ensure stock_ohlcv_df has Date/Ticker index for TA functions
        if not isinstance(stock_ohlcv_df.index, pd.MultiIndex) and {'Date', 'Ticker'}.issubset(stock_ohlcv_df.columns):
            stock_ohlcv_df = stock_ohlcv_df.set_index(['Date', 'Ticker'])

        stock_with_ta_df = feature_engineering.add_technical_indicators(stock_ohlcv_df.copy())

        # 2e. Merge Data
        merged_data = feature_engineering.merge_data(stock_with_ta_df, gdelt_agg_features)
        if merged_data is None or merged_data.empty:
            logger.error("Data merging failed or resulted in empty data. Exiting.")
            return

        # Ensure index is Date, Ticker after merge for subsequent steps
        if not (isinstance(merged_data.index, pd.MultiIndex) and
                {'Date', 'Ticker'}.issubset(merged_data.index.names)):
            if {'Date', 'Ticker'}.issubset(merged_data.columns):
                merged_data.set_index(['Date', 'Ticker'], inplace=True)
                merged_data.sort_index(inplace=True)
            else:
                logger.error("Merged data missing Date/Ticker for index. Exiting.")
                return

        # 2f. Define Target Variable
        data_with_target_df = feature_engineering.define_target_variable(
            merged_data.copy(), target_col_name=MODEL_TARGET_VARIABLE, horizon=TARGET_HORIZON
        )

        # 2g. Create Lagged Features
        # Define features to lag (exclude identifiers, current OHLCV, target, raw GDELT text etc.)
        # Feature columns should be numeric or categorical that make sense to lag.
        cols_to_exclude_lag = [
            MODEL_TARGET_VARIABLE, 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker',
            'CompanyName', 'Sector', 'Industry', 'CompanyName_Clean',
            'GLOBALEVENTID', 'SQLDATE', 'Actor1Name', 'Actor2Name', 'SOURCEURL', 'Title From URL'
        ]
        feature_cols_for_lagging = [col for col in data_with_target_df.columns if col not in cols_to_exclude_lag]

        final_features_df = feature_engineering.create_lagged_features(
            data_with_target_df.copy(), feature_cols_for_lagging, LAG_PERIODS
        )

        # 2h. Final Cleanup (Drop NaNs from lagging and target creation)
        initial_rows = len(final_features_df)
        final_features_df.dropna(subset=[MODEL_TARGET_VARIABLE], inplace=True)  # Drop rows where target is NaN
        final_features_df.dropna(inplace=True)  # Drop any other NaNs (e.g. from lags)
        rows_dropped = initial_rows - len(final_features_df)
        logger.info(f"Dropped {rows_dropped} rows due to NaNs from target creation & lagging.")

        if final_features_df.empty:
            logger.error("Data processing resulted in an empty DataFrame. Exiting.")
            return

        # 2i. Save Processed Data
        logger.info(f"Saving processed data to {PROCESSED_FEATURES_FILE}")
        try:
            # Parquet needs columns, not multi-index, for broadest compatibility if reset_index()
            final_features_df.reset_index().to_parquet(PROCESSED_FEATURES_FILE, index=False)
            logger.info("Processed data saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save processed data: {e}", exc_info=True)
            # Decide if to proceed without saving or exit

        if final_features_df is not None and not final_features_df.empty:
            logger.info("Downcasting 'final_features_df' immediately after creation/processing.")
            final_features_df = downcast_numeric_df(final_features_df)
            gc.collect()
        else:
            logger.error("Data processing resulted in an empty DataFrame. Exiting.")
            return

    else:
        logger.info(f"--- Phase: Loading Pre-processed Data from {PROCESSED_FEATURES_FILE} ---")
        final_features_df = load_processed_data(PROCESSED_FEATURES_FILE)  # From utils

        if final_features_df is not None and not final_features_df.empty:
            logger.info("Downcasting 'final_features_df' immediately after loading from file.")
            final_features_df = downcast_numeric_df(final_features_df)
            gc.collect()
        else:
            logger.error(f"Failed to load pre-processed data from {PROCESSED_FEATURES_FILE}. Exiting.")
            return

    # At this point, final_features_df should be loaded and downcasted, regardless of the path taken.
    if final_features_df is None or final_features_df.empty:
        logger.error("Critical error: final_features_df is not available after data loading/processing. Exiting.")
        return

    # --- Step 3: Model Training ---
    trained_model = None
    model_feature_columns = None

    if FORCE_MODEL_TRAINING or not os.path.exists(MODEL_FILE) or not os.path.exists(MODEL_FEATURES_FILE):
        logger.info("--- Phase: Model Training ---")
        # final_features_df is already loaded and should be downcasted.
        trained_model, model_feature_columns = model_training.train_evaluate_and_save_model(
            input_data=final_features_df,  # Pass the (already downcasted) DataFrame
            target_variable_name=MODEL_TARGET_VARIABLE,
            model_save_path=MODEL_FILE,
            features_save_path=MODEL_FEATURES_FILE,
            lgbm_params=model_training.LGBM_PARAMS,
            test_ratio=TRAIN_TEST_RATIO
        )
        if trained_model is None:
            logger.error("Model training failed. Cannot proceed to backtesting.")
            return
    else:
        logger.info(f"--- Phase: Loading Existing Model from {MODEL_FILE} and features from {MODEL_FEATURES_FILE} ---")
        trained_model, model_feature_columns = load_model_artefacts(MODEL_FILE, MODEL_FEATURES_FILE)  # From utils
        if trained_model is None:
            logger.error("Failed to load existing model. Consider re-running with FORCE_MODEL_TRAINING=True.")
            return
        if model_feature_columns is None:
            logger.warning("Model features list not loaded. Backtesting may be unreliable if features are inferred.")
            # Fallback: Try to infer from final_features_df if needed by backtest logic, but this is risky
            # The backtest function should ideally require model_feature_columns

    # --- Step 4: Backtesting ---
    logger.info("--- Phase: Backtesting ---")
    if trained_model is None:
        logger.error("Model not available for backtesting. Exiting.")
        return
    if final_features_df is None or final_features_df.empty:  # Should have been caught earlier
        logger.error("Processed data (final_features_df) is not available for backtesting. Exiting.")
        return
    if model_feature_columns is None:
        logger.error("Feature columns used for training are not available. Cannot reliably run backtest. Exiting.")
        return

    # final_features_df should be the downcasted version now.
    logger.info(
        f"Preparing to copy 'final_features_df' for backtesting. Current memory usage: {final_features_df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB")
    try:
        backtest_run_data = final_features_df.copy()  # This is where it crashed previously
        logger.info("Successfully copied 'final_features_df' into 'backtest_run_data'.")
    except MemoryError as e:
        logger.error(f"MemoryError during final_features_df.copy() even after downcasting attempts: {e}", exc_info=True)
        logger.error(
            f"Memory usage of final_features_df before copy: {final_features_df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB")
        logger.error(
            "If downcasting was not effective, you may need to reduce features or use a machine with more RAM.")
        return

    # Optional: If memory is extremely tight, and final_features_df is not needed anymore
    # logger.info("Deleting 'final_features_df' after copy to free up memory before Cerebro run.")
    # del final_features_df
    # gc.collect()

    # Filter data for the backtest period (BT_START_DATE, BT_END_DATE)
    if BT_START_DATE:
        backtest_run_data = backtest_run_data[
            backtest_run_data.index.get_level_values('Date') >= pd.to_datetime(BT_START_DATE)]
    if BT_END_DATE:
        backtest_run_data = backtest_run_data[
            backtest_run_data.index.get_level_values('Date') <= pd.to_datetime(BT_END_DATE)]

    if backtest_run_data.empty:
        logger.error("No data available for the specified backtest period in main.py.")
        return

    # Using the run_backtest function from backtest.py (or embed logic here)
    # For tighter integration, the backtesting logic from backtest.py's run_backtest can be moved here.
    # Let's integrate it directly for streamlining:

    cerebro = bt.Cerebro(stdstats=False, cheat_on_open=True)
    cerebro.addstrategy(MLTradingStrategy,
                        model=trained_model,
                        feature_cols=model_feature_columns,  # Use loaded/trained feature columns
                        buy_threshold=BT_BUY_THRESHOLD,
                        sell_threshold=BT_SELL_THRESHOLD,
                        printlog=True)

    # Prepare data for the backtest period
    # The `final_features_df` loaded earlier should be used.
    # Ensure it has Date/Ticker index. load_processed_data should handle this.

    backtest_run_data = final_features_df.copy()
    if BT_START_DATE:
        backtest_run_data = backtest_run_data[
            backtest_run_data.index.get_level_values('Date') >= pd.to_datetime(BT_START_DATE)]
    if BT_END_DATE:
        backtest_run_data = backtest_run_data[
            backtest_run_data.index.get_level_values('Date') <= pd.to_datetime(BT_END_DATE)]

    if backtest_run_data.empty:
        logger.error("No data available for the specified backtest period in main.py.")
        return

    tickers_to_run_bt = backtest_run_data.index.get_level_values('Ticker').unique().tolist()
    logger.info(f"Backtesting on {len(tickers_to_run_bt)} tickers.")
    added_tickers_bt_count = 0

    for ticker in tickers_to_run_bt:
        df_ticker_bt = backtest_run_data.xs(ticker, level='Ticker').copy()
        if not df_ticker_bt.empty:
            # Ensure OHLCV columns, add 'openinterest'
            required_bt_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df_ticker_bt.columns for col in required_bt_cols):
                logger.warning(f"Ticker {ticker} missing one of {required_bt_cols}. Skipping.")
                continue
            if 'openinterest' not in df_ticker_bt.columns:
                df_ticker_bt['openinterest'] = 0.0

            # Create the custom data feed using the class from utils
            # PandasDataWithFeatures expects datetime in a column if index is not datetime.
            # Here, df_ticker_bt has 'Date' as index.

            # Pass feature_cols to PandasDataWithFeatures constructor
            data_feed_bt = PandasDataWithFeatures(
                dataname=df_ticker_bt.reset_index(),  # PandasData often prefers datetime as a column
                feature_cols=model_feature_columns,  # Pass the correct feature list
                # dtformat=('%Y-%m-%d %H:%M:%S'),  # <--- REMOVE THIS LINE
                datetime='Date',  # This tells PandasData to look for a 'Date' column
                open='Open',
                high='High',
                low='Low',
                close='Close',
                volume='Volume',
                openinterest='openinterest'  # Make sure 'openinterest' column exists in df_ticker_bt
            )
            cerebro.adddata(data_feed_bt, name=ticker)
            added_tickers_bt_count += 1

    if added_tickers_bt_count == 0:
        logger.error("No data feeds added to Cerebro in main.py backtest. Aborting.")
        return

    cerebro.broker.set_cash(BT_INITIAL_CASH)
    cerebro.broker.setcommission(commission=BT_COMMISSION_PER_TRADE)
    cerebro.broker.set_slippage_perc(perc=BT_SLIPPAGE_PERCENT)
    cerebro.addsizer(FixedFractionPortfolioSizer, perc=BT_POSITION_SIZE_PERCENT / 100.0)  # From utils

    # Add Analyzers
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, compression=252,
                        riskfreerate=0.0)
    cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='tradeanalyzer')
    cerebro.addanalyzer(btanalyzers.SQN, _name='sqn')

    logger.info(f"Starting Backtrader run from main.py. Initial Portfolio: {cerebro.broker.getvalue():,.2f}")
    results = cerebro.run()
    logger.info("Backtrader run complete.")

    # Print Analysis Results (simplified, can be expanded)
    if results and results[0]:
        final_value = cerebro.broker.getvalue()
        logger.info(f"\n--- Main.py Backtest Results ---")
        logger.info(f"Initial Portfolio Value: ${BT_INITIAL_CASH:,.2f}")
        logger.info(f"Final Portfolio Value:   ${final_value:,.2f}")
        logger.info(f"Total Return:            {((final_value / BT_INITIAL_CASH) - 1) * 100:.2f}%")
        # Add more detailed analyzer outputs here from results[0].analyzers...
        try:
            sharpe_ratio = results[0].analyzers.sharpe.get_analysis().get('sharperatio', 'N/A')
            logger.info(
                f"Annualized Sharpe Ratio: {sharpe_ratio if isinstance(sharpe_ratio, str) else f'{sharpe_ratio:.3f}'}")
        except Exception as e:
            logger.warning(f"Could not get Sharpe: {e}")
        # ... (similar for drawdown, SQN, tradeanalyzer)

        # Plotting
        try:
            figure = cerebro.plot(style='candlestick', barup='green', bardown='red', volume=False, iplot=False)[0][0]
            figure.set_size_inches(18, 10)
            plot_file = os.path.join(BASE_OUTPUT_DIR, 'main_backtest_plot.png')
            figure.savefig(plot_file, dpi=300)
            logger.info(f"Plot saved to {plot_file}")
        except Exception as e:
            logger.error(f"Error generating plot: {e}", exc_info=True)
    else:
        logger.error("Backtest in main.py did not produce results.")

    logger.info("--- Algorithmic Trading Pipeline Finished ---")


if __name__ == "__main__":
    run_trading_pipeline()