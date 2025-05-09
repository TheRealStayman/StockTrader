# main.py

import pandas as pd
import numpy as np
import os
import logging
import joblib
import backtrader as bt
import backtrader.analyzers as btanalyzers

# Import custom modules
import data_loader
import feature_engineering
import model_training  # Contains training and evaluation logic
from trading_strategy import MLTradingStrategy  # Import the strategy class

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. File Paths
STOCK_PRICES_CSV = 'data/stock_data.csv'  # Replace with your actual path
GDELT_JSON = 'data/gdelt_events.json'  # Replace with your actual path
STOCK_INFO_CSV = 'data/stock_info.csv'  # Replace with your actual path
OUTPUT_DIR = 'processed_data'
MODEL_DIR = 'models'
PROCESSED_FEATURES_FILE = os.path.join(OUTPUT_DIR, 'final_features.parquet')
MODEL_FILE = os.path.join(MODEL_DIR, 'gdelt_stock_model.joblib')

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# 2. Feature Engineering Parameters
LAG_PERIODS = [1, 3, 5, 10]  # Example lag periods for features
TARGET_HORIZON = 1  # Predict next day's movement
FUZZY_MATCH_THRESHOLD = 85  # Confidence threshold for linking companies

# 3. Model Training Parameters
TEST_SET_RATIO = 0.2  # Proportion of data for the final test set (used in training script)
# Model parameters are defined within model_training.py but could be loaded from a config file

# 4. Backtesting Parameters
INITIAL_CASH = 100000.0
COMMISSION_PER_TRADE = 0.001  # 0.1%
SLIPPAGE_PERCENT = 0.0005  # 0.05%
POSITION_SIZE_PERCENT = 5.0  # Invest 5% of portfolio value per trade
BACKTEST_START_DATE = None  # Use None to start from the beginning of the test set
BACKTEST_END_DATE = None  # Use None to end at the end of the test set

# 5. Control Flags
FORCE_DATA_PROCESSING = False  # Set to True to re-run data loading and feature engineering
FORCE_MODEL_TRAINING = False  # Set to True to re-train the model even if a saved one exists


# --- Main Workflow ---

def run_trading_system():
    """Orchestrates the entire data processing, model training, and backtesting pipeline."""

    # --- Step 1 & 2: Data Loading and Feature Engineering ---
    if FORCE_DATA_PROCESSING or not os.path.exists(PROCESSED_FEATURES_FILE):
        logging.info("--- Starting Data Loading and Feature Engineering ---")

        # 1a. Load Raw Data
        stock_info_df = data_loader.load_stock_info(STOCK_INFO_CSV)
        gdelt_df = data_loader.load_gdelt_data(GDELT_JSON)
        stock_ohlcv_df = data_loader.load_stock_data(STOCK_PRICES_CSV)  # Assumes long format output

        if stock_info_df is None or gdelt_df is None or stock_ohlcv_df is None:
            logging.error("Failed to load one or more data sources. Exiting.")
            return

        # 2a. Link GDELT Events to Stocks
        # Ensure 'CompanyName_Clean' exists in stock_info_df from data_loader
        if 'CompanyName_Clean' not in stock_info_df.columns:
            stock_info_df['CompanyName_Clean'] = stock_info_df['CompanyName'].apply(data_loader.clean_company_name)

        gdelt_linked_df = feature_engineering.link_events_to_stocks(gdelt_df, stock_info_df,
                                                                    threshold=FUZZY_MATCH_THRESHOLD)
        if gdelt_linked_df is None:
            logging.warning("GDELT linking failed or produced no matches. Proceeding without GDELT NLP/aggregation.")
            # Create an empty DataFrame with expected columns if needed downstream
            gdelt_agg = pd.DataFrame(columns=['Date', 'Ticker'])
        else:
            # 2b. Add NLP Features (Basic Sentiment)
            gdelt_nlp_df = feature_engineering.add_nlp_features(gdelt_linked_df)

            # 2c. Aggregate GDELT Features
            gdelt_agg = feature_engineering.aggregate_gdelt_features(gdelt_nlp_df)
            if gdelt_agg.empty and not gdelt_linked_df.empty:
                logging.warning(
                    "GDELT aggregation resulted in an empty DataFrame. Check linking and aggregation logic.")
                # Reset to an empty frame with expected columns if needed
                gdelt_agg = pd.DataFrame(
                    columns=['Date', 'Ticker', 'gdelt_event_count', 'gdelt_goldstein_mean', 'gdelt_tone_mean',
                             'gdelt_title_sentiment_mean'])  # Add other cols as needed

        # 2d. Add Technical Indicators
        stock_data_with_ta = feature_engineering.add_technical_indicators(stock_ohlcv_df.copy())  # Use copy

        # 2e. Merge Data
        # Ensure stock_data_with_ta has Date and Ticker in columns for merging if it's indexed
        if isinstance(stock_data_with_ta.index, pd.MultiIndex):
            stock_data_with_ta.reset_index(inplace=True)

        # Make sure gdelt_agg has Date and Ticker as columns
        if isinstance(gdelt_agg.index, pd.MultiIndex):
            gdelt_agg.reset_index(inplace=True)
        elif 'Date' not in gdelt_agg.columns and 'Date' in gdelt_agg.index.names:
            gdelt_agg.reset_index(inplace=True)

        merged_data = feature_engineering.merge_data(stock_data_with_ta, gdelt_agg)
        if merged_data is None:
            logging.error("Data merging failed. Exiting.")
            return

        # Set MultiIndex for lagging and target calculation
        if not isinstance(merged_data.index, pd.MultiIndex) and {'Date', 'Ticker'}.issubset(merged_data.columns):
            merged_data['Date'] = pd.to_datetime(merged_data['Date'])
            merged_data.set_index(['Date', 'Ticker'], inplace=True)
            merged_data.sort_index(inplace=True)
        elif not isinstance(merged_data.index, pd.MultiIndex):
            logging.error("Merged data does not have 'Date' and 'Ticker' columns for indexing.")
            return

        # 2f. Define Target Variable
        data_with_target = feature_engineering.define_target_variable(merged_data.copy(), horizon=TARGET_HORIZON)

        # 2g. Create Lagged Features
        # Define features to lag (exclude identifiers, future info, and non-numeric where necessary)
        cols_to_exclude_lag = ['Open', 'High', 'Low', 'Close', 'Volume', 'Target',  # Base OHLCV + Target
                               'Ticker', 'CompanyName', 'Sector', 'Industry',  # Identifiers/Info from merges
                               'GLOBALEVENTID', 'SQLDATE', 'Actor1Name', 'Actor2Name', 'SOURCEURL',
                               'Title From URL']  # Raw GDELT fields if present
        feature_cols_for_lagging = [col for col in data_with_target.columns if col not in cols_to_exclude_lag]
        final_features_df = feature_engineering.create_lagged_features(data_with_target, feature_cols_for_lagging,
                                                                       LAG_PERIODS)

        # 2h. Final Cleanup (Dropping NaNs introduced by lagging/target shifting)
        initial_rows = len(final_features_df)
        final_features_df.dropna(inplace=True)
        rows_dropped = initial_rows - len(final_features_df)
        logging.info(f"Dropped {rows_dropped} rows due to NaNs from lagging/target creation.")

        # 2i. Save Processed Data
        logging.info(f"Saving processed data to {PROCESSED_FEATURES_FILE}")
        try:
            # Reset index to save Ticker and Date as columns for easier loading later
            final_features_df.reset_index().to_parquet(PROCESSED_FEATURES_FILE, index=False)
            logging.info("Processed data saved successfully.")
        except Exception as e:
            logging.error(f"Failed to save processed data: {e}", exc_info=True)
            # Continue without saving if preferred, but it's recommended for large datasets

    else:
        logging.info(f"Loading pre-processed data from {PROCESSED_FEATURES_FILE}...")
        final_features_df = feature_engineering.load_processed_data(PROCESSED_FEATURES_FILE)
        if final_features_df is None:
            logging.error("Failed to load pre-processed data. Consider running with FORCE_DATA_PROCESSING=True.")
            return
        # Ensure index is set correctly after loading
        if not isinstance(final_features_df.index, pd.MultiIndex):
            if {'Date', 'Ticker'}.issubset(final_features_df.columns):
                final_features_df['Date'] = pd.to_datetime(final_features_df['Date'])
                final_features_df.set_index(['Date', 'Ticker'], inplace=True)
                final_features_df.sort_index(inplace=True)
            else:
                logging.error("Loaded data missing 'Date' or 'Ticker' columns for index.")
                return

    # --- Step 3: Model Training ---
    if FORCE_MODEL_TRAINING or not os.path.exists(MODEL_FILE):
        logging.info("--- Starting Model Training ---")
        # Split data (using the same logic as in model_training.py for consistency)
        X_train, X_test, y_train, y_test, feature_cols = model_training.split_data_chronological(
            final_features_df, model_training.TARGET_VARIABLE, test_size=model_training.TEST_SET_SIZE_RATIO
        )

        if feature_cols is None:  # Attempt to define if not inferred in training script
            feature_cols = [col for col in final_features_df.columns if col not in
                            ['Open', 'High', 'Low', 'Close', 'Volume', 'Ticker', 'Target',
                             'SQLDATE', 'GLOBALEVENTID', 'Actor1Name', 'Actor2Name',
                             'SOURCEURL', 'Title From URL', 'CompanyName', 'Sector', 'Industry'] and col not in ['open',
                                                                                                                 'high',
                                                                                                                 'low',
                                                                                                                 'close',
                                                                                                                 'volume',
                                                                                                                 'openinterest']]
            logging.warning(f"Manually defining feature columns in main.py: {len(feature_cols)} features.")

        # Train, evaluate, and save the model
        trained_pipeline = model_training.train_and_evaluate(
            X_train, y_train, X_test, y_test, model_training.LGBM_PARAMS, feature_cols
        )

        if trained_pipeline:
            model_training.save_model(trained_pipeline, MODEL_FILE)
            model_to_backtest = trained_pipeline
            features_for_backtest = feature_cols  # Use the columns defined during training
        else:
            logging.error("Model training failed. Cannot proceed to backtesting.")
            return
    else:
        logging.info(f"--- Loading Existing Model from {MODEL_FILE} ---")
        try:
            model_to_backtest = joblib.load(MODEL_FILE)
            logging.info("Model loaded successfully.")
            # Attempt to get feature names from the loaded pipeline
            try:
                preprocessor_step = model_to_backtest.named_steps['preprocessor']
                # This part depends heavily on how the pipeline was constructed in model_training.py
                # Assuming ColumnTransformer with named 'num' step
                numeric_transformer = next((item[1] for item in preprocessor_step.transformers_ if item[0] == 'num'),
                                           None)
                if numeric_transformer and hasattr(numeric_transformer, 'feature_names_in_'):
                    features_for_backtest = numeric_transformer.feature_names_in_.tolist()
                    logging.info(f"Extracted feature names from loaded preprocessor: {len(features_for_backtest)}")
                elif hasattr(model_to_backtest.named_steps['classifier'], 'feature_name_'):
                    features_for_backtest = model_to_backtest.named_steps['classifier'].feature_name_
                    logging.info(f"Extracted feature names from loaded classifier: {len(features_for_backtest)}")
                else:
                    logging.warning("Could not extract features from loaded model pipeline. Re-inferring from data.")
                    # Fallback: Infer from the loaded dataframe
                    features_for_backtest = [col for col in final_features_df.columns if col not in
                                             ['Open', 'High', 'Low', 'Close', 'Volume', 'Ticker', 'Target',
                                              'SQLDATE', 'GLOBALEVENTID', 'Actor1Name', 'Actor2Name',
                                              'SOURCEURL', 'Title From URL', 'CompanyName', 'Sector',
                                              'Industry'] and col not in ['open', 'high', 'low', 'close', 'volume',
                                                                          'openinterest']]

            except Exception as e:
                logging.warning(f"Error extracting features from model, re-inferring: {e}")
                features_for_backtest = [col for col in final_features_df.columns if col not in
                                         ['Open', 'High', 'Low', 'Close', 'Volume', 'Ticker', 'Target',
                                          'SQLDATE', 'GLOBALEVENTID', 'Actor1Name', 'Actor2Name',
                                          'SOURCEURL', 'Title From URL', 'CompanyName', 'Sector',
                                          'Industry'] and col not in ['open', 'high', 'low', 'close', 'volume',
                                                                      'openinterest']]

        except FileNotFoundError:
            logging.error(f"Model file not found at {MODEL_FILE}, and FORCE_MODEL_TRAINING is False. Exiting.")
            return
        except Exception as e:
            logging.error(f"Error loading model: {e}", exc_info=True)
            return

    # --- Step 4: Backtesting ---
    logging.info("--- Starting Backtesting ---")

    # Define the custom data feed class again here or ensure it's importable if defined elsewhere
    class PandasDataWithFeatures(bt.feeds.PandasData):
        lines = tuple(features_for_backtest)  # Use features identified/loaded
        params = tuple(
            [(col, -1) for col in features_for_backtest] +
            [('datetime', None), ('open', 'Open'), ('high', 'High'), ('low', 'Low'),
             ('close', 'Close'), ('volume', 'Volume'), ('openinterest', -1)]
        )

    cerebro = bt.Cerebro(stdstats=False, cheat_on_open=True)  # cheat_on_open=True can use Open price for orders

    # Add Strategy
    cerebro.addstrategy(MLTradingStrategy,
                        model=model_to_backtest,
                        feature_cols=features_for_backtest,
                        printlog=True)  # Enable logging for backtest details

    # Prepare data for the backtest period
    backtest_data = final_features_df.copy()  # Use the full processed data or filter if needed
    if BACKTEST_START_DATE:
        backtest_data = backtest_data[
            backtest_data.index.get_level_values('Date') >= pd.to_datetime(BACKTEST_START_DATE)]
    if BACKTEST_END_DATE:
        backtest_data = backtest_data[backtest_data.index.get_level_values('Date') <= pd.to_datetime(BACKTEST_END_DATE)]

    if backtest_data.empty:
        logging.error("No data available for the specified backtest period.")
        return

    tickers_to_backtest = backtest_data.index.get_level_values('Ticker').unique().tolist()
    logging.info(f"Backtesting on {len(tickers_to_backtest)} tickers.")

    # Add Data Feeds to Cerebro
    for ticker in tickers_to_backtest:
        df_ticker = backtest_data.xs(ticker, level='Ticker').copy()
        if not df_ticker.empty:
            # Add 'openinterest' if it doesn't exist
            if 'openinterest' not in df_ticker.columns:
                df_ticker['openinterest'] = 0.0

            # Ensure columns match required names by PandasDataFeed (lowercase standard)
            # We map them in PandasDataWithFeatures params, so explicit renaming might not be needed if keys match
            # df_ticker.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)

            data_feed = PandasDataWithFeatures(dataname=df_ticker)
            cerebro.adddata(data_feed, name=ticker)
            logging.debug(
                f"Added data feed for {ticker} from {df_ticker.index.min().date()} to {df_ticker.index.max().date()}")

    if len(cerebro.datas) == 0:
        logging.error("No data feeds were successfully added for backtesting.")
        return

    # Configure Broker and Sizer
    cerebro.broker.set_cash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=COMMISSION_PER_TRADE)
    cerebro.broker.set_slippage_perc(perc=SLIPPAGE_PERCENT)
    cerebro.addsizer(FixedFractionPortfolioSizer, perc=POSITION_SIZE_PERCENT / 100.0)  # Use the custom sizer

    # Add Analyzers
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, compression=252,
                        riskfreerate=0.0)
    cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='tradeanalyzer')
    cerebro.addanalyzer(btanalyzers.SQN, _name='sqn')
    cerebro.addanalyzer(btanalyzers.Returns, _name='returns')

    # Run the backtest
    logging.info(f"Starting Backtrader run. Initial Portfolio Value: {cerebro.broker.getvalue():,.2f}")
    results = cerebro.run()
    logging.info("Backtrader run complete.")

    # Print Analysis Results
    strategy_instance = results[0]  # Get the first strategy instance
    final_value = cerebro.broker.getvalue()

    print("\n" + "=" * 40)
    print("          BACKTEST RESULTS")
    print("=" * 40)
    print(f"Initial Portfolio Value: ${INITIAL_CASH:,.2f}")
    print(f"Final Portfolio Value:   ${final_value:,.2f}")
    print(f"Total Return:            {((final_value / INITIAL_CASH) - 1) * 100:.2f}%")

    print("-" * 40)
    # Access analysis results safely
    try:
        sharpe_analysis = strategy_instance.analyzers.sharpe.get_analysis()
        print(
            f"Annualized Sharpe Ratio: {sharpe_analysis.get('sharperatio', 'N/A'):.3f}" if sharpe_analysis and sharpe_analysis.get(
                'sharperatio') is not None else "Sharpe Ratio: N/A")
    except KeyError:
        print("Sharpe Ratio: Analyzer not found or no trades.")

    try:
        drawdown_analysis = strategy_instance.analyzers.drawdown.get_analysis()
        print(f"Maximum Drawdown:        {drawdown_analysis.max.drawdown:.2f}%")
        print(f"Max Money Drawdown:      ${drawdown_analysis.max.moneydown:,.2f}")
    except KeyError:
        print("Drawdown: Analyzer not found or no trades.")

    try:
        sqn_analysis = strategy_instance.analyzers.sqn.get_analysis()
        print(f"System Quality Number:   {sqn_analysis.get('sqn', 'N/A'):.2f}" if sqn_analysis else "SQN: N/A")
    except KeyError:
        print("SQN: Analyzer not found or no trades.")

    try:
        trade_analysis = strategy_instance.analyzers.tradeanalyzer.get_analysis()
        if trade_analysis and trade_analysis.total and trade_analysis.total.total > 0:
            print("-" * 40)
            print("Trade Analysis:")
            print(f"  Total Trades:          {trade_analysis.total.total}")
            print(f"  Total Closed Trades:   {trade_analysis.total.closed}")
            print(f"  Winning Trades:        {trade_analysis.won.total}")
            print(f"  Losing Trades:         {trade_analysis.lost.total}")
            if trade_analysis.total.closed > 0:
                print(f"  Win Rate:              {trade_analysis.won.total / trade_analysis.total.closed * 100:.2f}%")
                print(f"  Avg Winning Trade ($): {trade_analysis.won.pnl.average:.2f}")
                print(f"  Avg Losing Trade ($):  {trade_analysis.lost.pnl.average:.2f}")
                profit_factor = abs(
                    trade_analysis.won.pnl.total / trade_analysis.lost.pnl.total) if trade_analysis.lost.pnl.total != 0 else float(
                    'inf')
                print(f"  Profit Factor:         {profit_factor:.2f}")
                print(f"  Avg Trade PnL ($):     {trade_analysis.pnl.net.average:.2f}")
            else:
                print("  No trades were closed during the backtest.")
        else:
            print("\n--- No Trades Executed ---")
    except KeyError:
        print("Trade Analysis: Analyzer not found or no trades.")
    print("=" * 40)

    # Optional: Plotting
    try:
        import matplotlib.pyplot as plt  # Import here to avoid dependency if not plotting
        logging.info("Generating plot...")
        # Ensure figure size is reasonable for potentially many data feeds
        figure = cerebro.plot(style='candlestick', barup='green', bardown='red', volume=False, iplot=False)[0][
            0]  # Get the figure object
        figure.set_size_inches(18, 10)  # Adjust size as needed
        figure.savefig('backtest_results_plot.png', dpi=300)
        logging.info("Plot saved to backtest_results_plot.png")
        # plt.show() # Uncomment to display plot interactively if needed
    except ImportError:
        logging.warning("Matplotlib not found. Skipping plot generation. Install with: pip install matplotlib")
    except Exception as e:
        logging.error(f"Error generating plot: {e}", exc_info=True)


if __name__ == "__main__":
    run_trading_system()