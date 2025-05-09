# trading_strategy.py

import backtrader as bt
import pandas as pd
import numpy as np
import logging

# --- Setup Logging ---
# Configure logging if needed within the strategy, though often handled by the main backtest script
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
# Example handler (adjust as needed, might be configured globally)
# handler = logging.StreamHandler()
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# log.addHandler(handler)

class MLTradingStrategy(bt.Strategy):
    """
    A Backtrader strategy that uses a pre-trained machine learning model
    (including preprocessing steps via a pipeline) and GDELT features
    to generate trading signals.

    Parameters:
        model: The trained scikit-learn pipeline object (preprocessor + classifier/regressor).
        feature_cols (list): A list of column names expected by the model's preprocessor.
        buy_threshold (float): Probability threshold to trigger a BUY signal.
        sell_threshold (float): Probability threshold to trigger a SELL signal.
        atr_period (int): Period for calculating ATR for stop-loss.
        atr_multiplier (float): Multiplier for ATR to set stop-loss distance.
        position_size (float): Fixed size or percentage of portfolio for each trade.
    """
    params = (
        ('model', None),
        ('feature_cols', None),
        ('buy_threshold', 0.55),  # Example: Buy if prob(Up) > 55%
        ('sell_threshold', 0.45), # Example: Sell if prob(Up) < 45%
        ('atr_period', 14),      # For potential ATR-based stop-loss
        ('atr_multiplier', 2.0), # Stop-loss distance = ATR * multiplier
        ('position_size', 0.05),  # Example: Invest 5% of portfolio cash per trade
        ('printlog', True),      # Control logging output
    )

    def __init__(self):
        """Initializes the strategy."""
        if not self.params.model:
            raise ValueError("A trained model must be provided via parameters.")
        if not self.params.feature_cols:
            raise ValueError("Feature columns must be provided via parameters.")

        self.model = self.params.model
        self.feature_cols = self.params.feature_cols

        # Keep references to easily access data lines for each stock
        self.d_close = {d._name: d.close for d in self.datas}
        self.d_open = {d._name: d.open for d in self.datas}
        self.d_high = {d._name: d.high for d in self.datas}
        self.d_low = {d._name: d.low for d in self.datas}
        self.d_volume = {d._name: d.volume for d in self.datas}

        # Store references to custom feature lines (assuming they are added to the data feed)
        self.d_features = {}
        for d in self.datas:
            self.d_features[d._name] = {col: d.lines.getlinealias(col) for col in self.feature_cols if hasattr(d.lines, col)}

        # Keep track of pending orders and stop-loss orders
        self.orders = {d._name: None for d in self.datas}
        self.stop_orders = {d._name: None for d in self.datas}

        # Initialize ATR indicator for stop-loss calculation (optional)
        self.atrs = {d._name: bt.indicators.ATR(d, period=self.p.atr_period) for d in self.datas}

        log.info('Strategy Initialized')

    def log(self, txt, dt=None, doprint=False):
        """Logging function for the strategy"""
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}, {txt}')

    def notify_order(self, order):
        """Handles order notifications."""
        stock_name = order.data._name if order.data else 'Unknown' # Handle potential metadata issue

        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted - Nothing to do
            self.log(f'{stock_name}: Order {order.Status[order.status]}')
            self.orders[stock_name] = order # Store the order reference
            return

        # Check if an order has been completed
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f'{stock_name}: BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}'
                )
                # --- Stop-Loss Order Placement (Example) ---
                # If implementing ATR-based stops, place the stop order after entry
                # stop_price = order.executed.price - self.atrs[stock_name][0] * self.p.atr_multiplier
                # self.stop_orders[stock_name] = self.sell(data=order.data, size=order.executed.size, exectype=bt.Order.Stop, price=stop_price)
                # self.log(f'{stock_name}: Stop-loss order placed at {stop_price:.2f}')
                # --- End Stop-Loss Example ---

            elif order.issell():
                self.log(
                    f'{stock_name}: SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}'
                )
                # Cancel any associated stop-loss order if it exists and is active
                if self.stop_orders.get(stock_name):
                    self.cancel(self.stop_orders[stock_name])
                    self.stop_orders[stock_name] = None

            self.orders[stock_name] = None # Clear pending order ref

        elif order.status in [order.Canceled, order.Margin, order.Rejected, order.Expired]:
            self.log(f'{stock_name}: Order {order.Status[order.status]}')
            # Ensure any associated stop is also cancelled if the main order fails
            if order.isbuy() and self.stop_orders.get(stock_name):
                 self.cancel(self.stop_orders[stock_name])
                 self.stop_orders[stock_name] = None
            self.orders[stock_name] = None # Clear pending order ref


    def notify_trade(self, trade):
        """Handles trade notifications."""
        if not trade.isclosed:
            return
        self.log(f'{trade.data._name}: OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')

    def next(self):
        """Main logic executed for each bar/day."""
        current_date = self.datas[0].datetime.date(0) # Use the first data feed for the current date

        for i, d in enumerate(self.datas):
            stock_name = d._name
            position = self.getposition(d).size

            # Skip if order is pending for this stock
            if self.orders.get(stock_name):
                continue

            # --- Feature Preparation ---
            # Ensure we have enough data points for lags and indicators
            # The exact number depends on the largest lag period used in feature engineering
            min_required_bars = 30 # Example, adjust based on longest lookback (e.g., SMA50, lags)
            if len(d) < min_required_bars:
                 continue

            # Collect features for the current time step
            try:
                current_values = {col: self.d_features[stock_name][col][0] for col in self.feature_cols if col in self.d_features[stock_name]}
                # Ensure all expected features are present
                if len(current_values) != len(self.feature_cols):
                     missing_features = set(self.feature_cols) - set(current_values.keys())
                     # log.warning(f"Missing features for {stock_name} on {current_date}: {missing_features}. Skipping prediction.")
                     continue # Skip if not all features are available

                # Create a DataFrame row in the correct order for the pipeline
                features_df = pd.DataFrame([current_values], columns=self.feature_cols)

            except Exception as e:
                log.error(f"Error preparing features for {stock_name} on {current_date}: {e}")
                continue # Skip this stock for this bar if features can't be prepared

            # --- Prediction ---
            try:
                # Model expects numpy array or DataFrame matching training structure
                # The pipeline handles imputation and scaling if it was saved correctly
                prediction_proba = self.model.predict_proba(features_df)
                prob_buy = prediction_proba[0][1] # Assuming class 1 is 'Buy'/'Up'
                prob_sell = prediction_proba[0][0] # Assuming class 0 is 'Sell'/'Down'
            except Exception as e:
                log.error(f"Error during prediction for {stock_name} on {current_date}: {e}")
                continue # Skip if prediction fails

            # --- Trading Logic ---
            # Simple example: Buy if probability > threshold, Sell if opposite is true (or below another threshold)
            # Ensure we have cash for new positions
            cash = self.broker.get_cash()
            value_per_position = self.broker.get_value() * self.p.position_size # Example sizing

            if not position: # No current position
                if prob_buy > self.p.buy_threshold:
                    size_to_buy = int(value_per_position / self.d_close[stock_name][0])
                    if size_to_buy > 0 and cash >= (size_to_buy * self.d_close[stock_name][0]): # Check affordability
                        self.log(f'{stock_name}: BUY SIGNAL - Prob: {prob_buy:.2f}, Price: {self.d_close[stock_name][0]:.2f}, Size: {size_to_buy}')
                        self.orders[stock_name] = self.buy(data=d, size=size_to_buy)
                        # Optional: Place associated stop-loss order immediately (alternative to managing in next)
                        # stop_price = self.d_close[stock_name][0] * (1 - self.p.stop_loss_pct) # Example fixed percentage stop
                        # self.stop_orders[stock_name] = self.sell(data=d, size=size_to_buy, exectype=bt.Order.Stop, price=stop_price, parent=self.orders[stock_name])

            else: # We have a position
                if prob_buy < self.p.sell_threshold: # Condition to exit the long position
                    self.log(f'{stock_name}: SELL SIGNAL - Prob: {prob_buy:.2f}, Price: {self.d_close[stock_name][0]:.2f}, Size: {position}')
                    self.orders[stock_name] = self.close(data=d) # Close the existing position
                    # Cancel any existing stop loss order for this position
                    if self.stop_orders.get(stock_name):
                         self.cancel(self.stop_orders[stock_name])
                         self.stop_orders[stock_name] = None

                # --- Manual Stop-Loss Check (if not using broker stop orders) ---
                # if self.d_low[stock_name][0] < stop_price_level_for_this_position:
                #     self.log(f'{stock_name}: STOP LOSS HIT - Selling at Market')
                #     self.orders[stock_name] = self.close(data=d)
                #     if self.stop_orders.get(stock_name): # Should have already been triggered if using broker stops
                #         self.cancel(self.stop_orders[stock_name])
                #         self.stop_orders[stock_name] = None

    def stop(self):
        """Actions to perform at the end of the backtest."""
        final_value = self.broker.getvalue()
        self.log(f'Ending Value {final_value:.2f}', doprint=True)
        # Any other final calculations or reporting

# --- Example of how this strategy might be used in backtest.py ---
# (This part would typically be in a separate backtest runner script)
# if __name__ == '__main__':
#     # This block is for demonstration/testing and would be moved
#     # to the main backtesting script (e.g., backtest.py)
#
#     # 1. Load the trained model pipeline
#     try:
#         model_pipeline = joblib.load(MODEL_SAVE_PATH)
#         logging.info(f"Model loaded successfully from {MODEL_SAVE_PATH}")
#     except FileNotFoundError:
#         logging.error(f"Model file not found at {MODEL_SAVE_PATH}. Train the model first.")
#         exit()
#     except Exception as e:
#         logging.error(f"Error loading model: {e}", exc_info=True)
#         exit()
#
#     # 2. Load the full featureset (needed for backtrader data feeds)
#     #    This assumes final_features_df includes OHLCV and all engineered features
#     final_features_df = load_processed_data(PROCESSED_DATA_PATH)
#     if final_features_df is None:
#          exit()
#
#     # Extract feature columns used during training (excluding target, identifiers etc.)
#     # Ensure this list matches the one used in train_and_evaluate
#     # Need to get this list reliably, maybe save it with the model
#     # For now, re-calculate it based on columns present
#     potential_feature_cols = [col for col in final_features_df.columns if col not in
#                               ['Open', 'High', 'Low', 'Close', 'Volume', 'Ticker', 'Target',
#                                'SQLDATE', 'GLOBALEVENTID', 'Actor1Name', 'Actor2Name',
#                                'SOURCEURL', 'Title From URL', 'CompanyName', 'Sector', 'Industry'] and
#                               '_lag_' in col or col in ['SMA_20', 'SMA_50', 'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9',
#                                                          'MACDs_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
#                                                          'BBB_20_2.0', 'BBP_20_2.0', 'ATR_14', 'OBV',
#                                                          'gdelt_event_count', 'gdelt_goldstein_mean', 'gdelt_goldstein_sum',
#                                                          'gdelt_goldstein_min', 'gdelt_goldstein_max', 'gdelt_tone_mean',
#                                                          'gdelt_tone_sum', 'gdelt_num_articles_sum', 'gdelt_title_sentiment_mean']]
#
#     # Create a Cerebro engine instance
#     cerebro = bt.Cerebro(stdstats=False) # Disable standard observers initially
#
#     # Add the strategy
#     cerebro.addstrategy(MLTradingStrategy, model=model_pipeline, feature_cols=potential_feature_cols)
#
#     # Add Data Feeds (Example for a few tickers)
#     tickers_to_backtest = final_features_df.index.get_level_values('Ticker').unique()[:5] # Limit for example
#     final_features_df_reset = final_features_df.reset_index()
#
#     for ticker in tickers_to_backtest:
#         df_ticker = final_features_df_reset[final_features_df_reset['Ticker'] == ticker].copy()
#         if not df_ticker.empty:
#             df_ticker.set_index('Date', inplace=True)
#             df_ticker.rename(columns={
#                 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
#             }, inplace=True)
#             df_ticker['openinterest'] = 0 # Add dummy openinterest if needed by Backtrader
#
#             # Ensure all feature columns exist for this ticker's data feed
#             data_feed = bt.feeds.PandasData(dataname=df_ticker,
#                                             datetime=None, # Use index
#                                             open='open', high='high', low='low', close='close', volume='volume', openinterest=None,
#                                             # Add custom feature lines dynamically
#                                             **{col: -1 for col in potential_feature_cols}) # Map features to lines
#
#             cerebro.adddata(data_feed, name=ticker)
#             logging.info(f"Added data feed for {ticker}")
#
#     # Set initial cash and commission
#     cerebro.broker.setcash(100000.0)
#     cerebro.broker.setcommission(commission=0.001) # Example: 0.1% commission
#     cerebro.addsizer(bt.sizers.PercentSizer, percents=90 / len(tickers_to_backtest)) # Example sizing: distribute 90% capital
#
#     # Add Analyzers
#     cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='mysharpe', timeframe=bt.TimeFrame.Days, compression=252) # Annualized Sharpe
#     cerebro.addanalyzer(btanalyzers.DrawDown, _name='mydrawdown')
#     cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='mytradeanalyzer')
#     cerebro.addanalyzer(btanalyzers.SQN, _name='mysqn')
#
#     # Run the backtest
#     logging.info('Starting CEREBRO Run...')
#     results = cerebro.run()
#     logging.info('CEREBRO Run Complete.')
#
#     # Print Analysis
#     strat = results[0]
#     print('\n--- Backtest Results ---')
#     print(f'Final Portfolio Value: {cerebro.broker.getvalue():,.2f}')
#     print(f'Net Profit/Loss: {cerebro.broker.getvalue() - cerebro.broker.startingcash:,.2f}')
#
#     if hasattr(strat.analyzers, 'mysharpe'):
#       sharpe_analysis = strat.analyzers.mysharpe.get_analysis()
#       print(f"Annualized Sharpe Ratio: {sharpe_analysis.get('sharperatio', 'N/A')}")
#     if hasattr(strat.analyzers, 'mydrawdown'):
#       drawdown_analysis = strat.analyzers.mydrawdown.get_analysis()
#       print(f"Max Drawdown: {drawdown_analysis.max.drawdown:.2f}%")
#       print(f"Max Money Drawdown: {drawdown_analysis.max.moneydown:,.2f}")
#     if hasattr(strat.analyzers, 'mytradeanalyzer'):
#         trade_analysis = strat.analyzers.mytradeanalyzer.get_analysis()
#         if trade_analysis and trade_analysis.total and trade_analysis.total.closed > 0:
#             print(f"Total Trades: {trade_analysis.total.total}")
#             print(f"Winning Trades: {trade_analysis.won.total}")
#             print(f"Losing Trades: {trade_analysis.lost.total}")
#             print(f"Win Rate: {trade_analysis.won.total / trade_analysis.total.closed * 100:.2f}%")
#             print(f"Avg Win ($): {trade_analysis.won.pnl.average:.2f}")
#             print(f"Avg Loss ($): {trade_analysis.lost.pnl.average:.2f}")
#             print(f"Profit Factor: {abs(trade_analysis.won.pnl.total / trade_analysis.lost.pnl.total) if trade_analysis.lost.pnl.total != 0 else 'Inf'}")
#         else:
#             print("No trades were executed.")
#     if hasattr(strat.analyzers, 'mysqn'):
#        sqn_analysis = strat.analyzers.mysqn.get_analysis()
#        print(f"System Quality Number (SQN): {sqn_analysis.get('sqn', 'N/A'):.2f}")
#
#     # cerebro.plot() # Uncomment to plot results if matplotlib is installed and backend is suitable
# else:
#      # This part executes only when the script is imported, not run directly
#      log.info("trading_strategy.py imported as a module.")
#