import yfinance as yf
import pandas as pd
import asyncio
from telegram import Bot
from telegram.error import TelegramError
import logging
import time
import numpy as np # For numerical operations, especially in indicator calculations
from datetime import datetime, timezone # For time-based scheduling

# --- Configuration ---
# IMPORTANT: Replace with your actual Telegram Bot Token and Chat ID
# Get your bot token from BotFather on Telegram.
# To get your chat ID, send a message to your bot, then go to
# https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
# and look for the 'chat' object's 'id'.
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID" # Make sure this is a string

# Crypto asset to trade (e.g., BTC-USD, ETH-USD)
CRYPTO_SYMBOL = "BTC-USD"

# --- Data Fetching Parameters ---
# Lower Timeframe (LTF) for entry/exit signals
LTF_INTERVAL = "1h"
LTF_PERIOD = "10d" # Needs enough data for all indicators and HTF
# Higher Timeframe (HTF) for trend filtering
HTF_INTERVAL = "4h"
HTF_PERIOD = "30d" # Needs enough data for HTF EMA

# --- Trading Strategy Parameters (Illustrative - NOT for live trading) ---
# Simple Moving Average (SMA) lengths for LTF signals
SMA_FAST_PERIOD = 20
SMA_SLOW_PERIOD = 50

# Relative Strength Index (RSI) parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Bollinger Bands (BB) parameters
BB_PERIOD = 20
BB_STD_DEV = 2

# Average True Range (ATR) parameters for volatility-adjusted stops
ATR_PERIOD = 14
ATR_STOP_LOSS_MULTIPLIER = 1.5 # e.g., 1.5 * ATR for stop loss
ATR_TAKE_PROFIT_MULTIPLIER = 3.0 # e.g., 3.0 * ATR for take profit

# Higher Timeframe (HTF) Trend Filter (Exponential Moving Average)
HTF_TREND_EMA_PERIOD = 200

# --- Risk Management Parameters ---
INITIAL_CAPITAL = 10000.0
TRANSACTION_FEE_PERCENT = 0.001 # e.g., 0.1%
RISK_PER_TRADE_PERCENT = 0.01 # Risk 1% of capital per trade

# --- Custom Trading Schedule ---
# Set to True for 24/7 trading, False to enable specific trading hours
CONTINUOUS_TRADING_ENABLED = True
# If CONTINUOUS_TRADING_ENABLED is False, define trading hours in UTC
# Example: 9 AM to 5 PM UTC (17:00)
TRADING_START_HOUR_UTC = 9
TRADING_END_HOUR_UTC = 17 # Exclusive, so 17 means up to 16:59:59

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Variables for Bot State ---
bot = Bot(token=TELEGRAM_BOT_TOKEN)
current_position_units = 0.0 # Number of crypto units held
current_capital_usd = INITIAL_CAPITAL
last_ltf_price = None
last_htf_trend = "UNKNOWN" # "BULLISH", "BEARISH", "SIDEWAYS", "UNKNOWN"

# Store active trade details for stop-loss/take-profit management
active_trade = {
    "entry_price": 0.0,
    "stop_loss": 0.0,
    "take_profit": 0.0,
    "is_active": False
}

# --- Helper Functions ---

async def send_telegram_message(message: str):
    """Sends a message to the configured Telegram chat."""
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        logging.info(f"Telegram message sent: {message}")
    except TelegramError as e:
        logging.error(f"Error sending Telegram message: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while sending Telegram message: {e}")

async def fetch_data(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """
    Fetches historical data using yfinance.
    NOTE: yfinance is for historical data. For real-time, use exchange APIs.
    """
    try:
        logging.info(f"Fetching {interval} data for {symbol} with period {period}...")
        data = yf.download(symbol, interval=interval, period=period)
        if data.empty:
            logging.warning(f"No {interval} data fetched for {symbol}. Check symbol/interval/period.")
            return pd.DataFrame()
        # Ensure data is sorted by index (timestamp)
        data = data.sort_index()
        logging.info(f"{interval} data fetched successfully. Rows: {len(data)}")
        return data
    except Exception as e:
        logging.error(f"Error fetching {interval} data for {symbol}: {e}")
        return pd.DataFrame()

def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates SMAs, RSI, Bollinger Bands, and ATR for the given DataFrame.
    """
    if data.empty or 'Close' not in data.columns:
        return data

    # Calculate SMAs
    data[f'SMA_{SMA_FAST_PERIOD}'] = data['Close'].rolling(window=SMA_FAST_PERIOD).mean()
    data[f'SMA_{SMA_SLOW_PERIOD}'] = data['Close'].rolling(window=SMA_SLOW_PERIOD).mean()

    # Calculate RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    # Using EWM for RSI as per standard calculation
    avg_gain = gain.ewm(com=RSI_PERIOD - 1, min_periods=RSI_PERIOD).mean()
    avg_loss = loss.ewm(com=RSI_PERIOD - 1, min_periods=RSI_PERIOD).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Calculate Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=BB_PERIOD).mean()
    data['BB_StdDev'] = data['Close'].rolling(window=BB_PERIOD).std()
    data['BB_Upper'] = data['BB_Middle'] + (data['BB_StdDev'] * BB_STD_DEV)
    data['BB_Lower'] = data['BB_Middle'] - (data['BB_StdDev'] * BB_STD_DEV)

    # Calculate ATR
    # True Range (TR) = max[(High - Low), abs(High - Close_prev), abs(Low - Close_prev)]
    high_low = data['High'] - data['Low']
    high_close_prev = np.abs(data['High'] - data['Close'].shift(1))
    low_close_prev = np.abs(data['Low'] - data['Close'].shift(1))
    data['TR'] = pd.DataFrame({'HL': high_low, 'HCP': high_close_prev, 'LCP': low_close_prev}).max(axis=1)
    data['ATR'] = data['TR'].ewm(com=ATR_PERIOD - 1, min_periods=ATR_PERIOD).mean()

    return data

def calculate_htf_trend(htf_data: pd.DataFrame) -> str:
    """
    Determines the higher timeframe trend using an EMA.
    """
    if htf_data.empty or len(htf_data) < HTF_TREND_EMA_PERIOD:
        return "UNKNOWN"

    htf_data['EMA_HTF'] = htf_data['Close'].ewm(span=HTF_TREND_EMA_PERIOD, adjust=False).mean()

    # Check the latest EMA vs. price, or EMA slope
    latest_ema = htf_data['EMA_HTF'].iloc[-1]
    latest_price = htf_data['Close'].iloc[-1]

    if latest_price > latest_ema:
        return "BULLISH"
    elif latest_price < latest_ema:
        return "BEARISH"
    else:
        return "SIDEWAYS" # Price is exactly on EMA, or very flat

async def execute_trade(action: str, current_price: float, ltf_data_for_atr: pd.DataFrame):
    """
    Simulates a trade execution, updates bot's position and capital,
    and manages stop-loss/take-profit for the active trade.
    This is a SIMULATION. No real money is involved.
    """
    global current_position_units, current_capital_usd, active_trade

    message = ""
    fee = 0.0

    if action == "BUY":
        # Calculate units based on risk per trade and ATR-based stop-loss
        # Need to ensure ATR is available for the latest price
        if 'ATR' not in ltf_data_for_atr.columns or pd.isna(ltf_data_for_atr['ATR'].iloc[-1]):
            logging.warning("ATR not available for dynamic position sizing. Skipping BUY.")
            return

        current_atr = ltf_data_for_atr['ATR'].iloc[-1]
        risk_amount_usd = current_capital_usd * RISK_PER_TRADE_PERCENT
        stop_loss_distance = ATR_STOP_LOSS_MULTIPLIER * current_atr

        if stop_loss_distance <= 0: # Avoid division by zero or negative distance
            logging.warning("Calculated stop-loss distance is non-positive. Skipping BUY.")
            return

        # Calculate units to buy based on risk and stop-loss distance
        # Units = Risk_Amount / Stop_Loss_Per_Unit
        # Stop_Loss_Per_Unit = Entry_Price - Stop_Loss_Price
        # Assuming stop-loss is below entry for long
        calculated_units = risk_amount_usd / stop_loss_distance

        # Ensure we don't buy more than we can afford
        cost = calculated_units * current_price
        if cost > current_capital_usd:
            calculated_units = current_capital_usd / current_price
            cost = current_capital_usd # Adjust cost to remaining capital
            logging.warning(f"Adjusted buy units due to insufficient capital. New units: {calculated_units:.4f}")

        if calculated_units <= 0:
            logging.warning("Calculated units to buy is zero or negative. Skipping BUY.")
            return

        trade_units = calculated_units
        fee = trade_units * current_price * TRANSACTION_FEE_PERCENT
        current_capital_usd -= (trade_units * current_price + fee)
        current_position_units += trade_units

        # Set up active trade details
        active_trade["entry_price"] = current_price
        active_trade["stop_loss"] = current_price - stop_loss_distance
        active_trade["take_profit"] = current_price + (ATR_TAKE_PROFIT_MULTIPLIER * current_atr)
        active_trade["is_active"] = True

        message = (f"📈 BUY Signal! Bought {trade_units:.4f} {CRYPTO_SYMBOL.split('-')[0]} "
                   f"at ${current_price:.2f}. Fee: ${fee:.2f}.\n"
                   f"Simulated SL: ${active_trade['stop_loss']:.2f}, TP: ${active_trade['take_profit']:.2f}.\n"
                   f"Remaining Capital: ${current_capital_usd:.2f}.")
        logging.info(message)
        await send_telegram_message(message)

    elif action == "SELL":
        # Always sell all units in a simple strategy for now
        trade_units = current_position_units
        if trade_units <= 0:
            logging.warning("No position to sell. Skipping SELL.")
            return

        fee = trade_units * current_price * TRANSACTION_FEE_PERCENT
        current_capital_usd += (trade_units * current_price - fee)
        current_position_units = 0.0 # Close position

        # Reset active trade details
        active_trade = {
            "entry_price": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "is_active": False
        }

        message = (f"📉 SELL Signal! Sold {trade_units:.4f} {CRYPTO_SYMBOL.split('-')[0]} "
                   f"at ${current_price:.2f}. Fee: ${fee:.2f}.\n"
                   f"Current Capital: ${current_capital_usd:.2f}.")
        logging.info(message)
        await send_telegram_message(message)

    elif action == "HOLD":
        logging.info("HOLD: No trade action taken.")
    else:
        logging.warning(f"Invalid trade action: {action}")

async def check_active_trade_management(current_price: float):
    """
    Checks if an active trade needs to be closed due to stop-loss or take-profit.
    """
    global current_position_units, current_capital_usd, active_trade

    if active_trade["is_active"] and current_position_units > 0:
        if current_price <= active_trade["stop_loss"]:
            logging.info(f"STOP LOSS HIT! Current Price: ${current_price:.2f}, SL: ${active_trade['stop_loss']:.2f}")
            # Pass dummy ltf_data_for_atr as it's not used in SELL
            await execute_trade("SELL", current_price, pd.DataFrame())
            await send_telegram_message(f"🚨 STOP LOSS HIT for {CRYPTO_SYMBOL} at ${current_price:.2f}!")
        elif current_price >= active_trade["take_profit"]:
            logging.info(f"TAKE PROFIT HIT! Current Price: ${current_price:.2f}, TP: ${active_trade['take_profit']:.2f}")
            # Pass dummy ltf_data_for_atr as it's not used in SELL
            await execute_trade("SELL", current_price, pd.DataFrame())
            await send_telegram_message(f"✅ TAKE PROFIT HIT for {CRYPTO_SYMBOL} at ${current_price:.2f}!")
        else:
            logging.info(f"Active trade: Price ${current_price:.2f} (SL: ${active_trade['stop_loss']:.2f}, TP: ${active_trade['take_profit']:.2f})")

async def trading_logic(ltf_data: pd.DataFrame, htf_trend: str):
    """
    Implements a combined strategy with HTF trend filter, SMA, RSI, and BB.
    This is a simulation and not a robust, profitable trading strategy.
    """
    global current_position_units, last_ltf_price

    # Ensure enough data for all indicator calculations
    min_periods_required = max(SMA_FAST_PERIOD, SMA_SLOW_PERIOD, RSI_PERIOD, BB_PERIOD, ATR_PERIOD)
    if ltf_data.empty or len(ltf_data) < min_periods_required:
        logging.warning(f"LTF: Not enough data for indicator calculation or trading logic. Need at least {min_periods_required} data points.")
        return

    # Get the latest data point from LTF
    latest_row = ltf_data.iloc[-1]
    current_price = latest_row['Close']
    fast_sma = latest_row[f'SMA_{SMA_FAST_PERIOD}']
    slow_sma = latest_row[f'SMA_{SMA_SLOW_PERIOD}']
    rsi = latest_row['RSI']
    bb_upper = latest_row['BB_Upper']
    bb_lower = latest_row['BB_Lower']
    atr = latest_row['ATR']

    # Check if all indicators are calculated (will be NaN for initial periods)
    if pd.isna(fast_sma) or pd.isna(slow_sma) or pd.isna(rsi) or \
       pd.isna(bb_upper) or pd.isna(bb_lower) or pd.isna(atr):
        logging.info("LTF: Some indicators not yet calculated for latest data point. Waiting for more data.")
        return

    logging.info(f"LTF Data: Price ${current_price:.2f} | Fast SMA: ${fast_sma:.2f} | Slow SMA: ${slow_sma:.2f} | RSI: {rsi:.2f} | BB Upper: ${bb_upper:.2f} | BB Lower: ${bb_lower:.2f} | ATR: {atr:.2f}")
    logging.info(f"HTF Trend: {htf_trend}")

    # Determine previous SMA values for crossover detection
    # Need at least 2 rows for previous values
    if len(ltf_data) >= 2:
        previous_row = ltf_data.iloc[-2]
        prev_fast_sma = previous_row[f'SMA_{SMA_FAST_PERIOD}']
        prev_slow_sma = previous_row[f'SMA_{SMA_SLOW_PERIOD}']
    else:
        prev_fast_sma, prev_slow_sma = None, None

    # --- Buy Signal Logic ---
    buy_signal = False
    if htf_trend == "BULLISH" and current_position_units == 0: # Only buy if HTF is bullish and no open position
        # SMA Crossover Buy
        sma_buy_condition = (prev_fast_sma is not None and prev_slow_sma is not None and
                             fast_sma > slow_sma and prev_fast_sma <= prev_slow_sma)
        # RSI Oversold Confirmation
        rsi_buy_condition = rsi < RSI_OVERSOLD
        # Price near Lower Bollinger Band (potential bounce from oversold)
        bb_buy_condition = current_price <= bb_lower

        if sma_buy_condition and rsi_buy_condition and bb_buy_condition:
            buy_signal = True
            logging.info("COMBINED BUY signal detected (SMA Crossover + RSI Oversold + BB Lower)!")

    # --- Sell Signal Logic ---
    sell_signal = False
    if current_position_units > 0: # Only sell if there's an open position
        # SMA Crossover Sell
        sma_sell_condition = (prev_fast_sma is not None and prev_slow_sma is not None and
                              fast_sma < slow_sma and prev_fast_sma >= prev_slow_sma)
        # RSI Overbought Confirmation
        rsi_sell_condition = rsi > RSI_OVERBOUGHT
        # Price near Upper Bollinger Band (potential reversal from overbought)
        bb_sell_condition = current_price >= bb_upper

        if sma_sell_condition and rsi_sell_condition: # Removed BB for sell to allow exit on SMA/RSI alone
            sell_signal = True
            logging.info("COMBINED SELL signal detected (SMA Crossover + RSI Overbought)!")

    # --- Execute Trades Based on Signals ---
    if buy_signal:
        await execute_trade("BUY", current_price, ltf_data) # Pass ltf_data for ATR
    elif sell_signal:
        await execute_trade("SELL", current_price, ltf_data) # Pass ltf_data for ATR (not used in SELL but for consistency)
    else:
        await execute_trade("HOLD", current_price, ltf_data) # Pass ltf_data for ATR (not used in HOLD but for consistency)

    last_ltf_price = current_price # Update last known price

async def run_bot():
    """Main function to run the trading bot."""
    logging.info("Starting enhanced crypto trading bot...")
    await send_telegram_message(f"🚀 Enhanced Crypto Trading Bot for {CRYPTO_SYMBOL} has started!")
    trading_mode_message = "24/7 continuous trading enabled." if CONTINUOUS_TRADING_ENABLED else \
                           f"Trading active between {TRADING_START_HOUR_UTC}:00 and {TRADING_END_HOUR_UTC}:00 UTC daily."
    await send_telegram_message(f"Trading Mode: {trading_mode_message}")
    await send_telegram_message(f"Strategy: HTF Trend Filter ({HTF_TREND_EMA_PERIOD} EMA on {HTF_INTERVAL}) + LTF Signals (SMA {SMA_FAST_PERIOD}/{SMA_SLOW_PERIOD}, RSI {RSI_PERIOD} [{RSI_OVERSOLD}/{RSI_OVERBOUGHT}], BB {BB_PERIOD}/{BB_STD_DEV}).")
    await send_telegram_message(f"Risk Management: {RISK_PER_TRADE_PERCENT*100:.0f}% risk per trade, ATR-based SL ({ATR_STOP_LOSS_MULTIPLIER}x ATR), TP ({ATR_TAKE_PROFIT_MULTIPLIER}x ATR).")

    while True:
        current_utc_hour = datetime.now(timezone.utc).hour
        is_within_trading_hours = CONTINUOUS_TRADING_ENABLED or \
                                  (TRADING_START_HOUR_UTC <= current_utc_hour < TRADING_END_HOUR_UTC)

        if is_within_trading_hours:
            # 1. Fetch HTF data for trend filtering
            htf_data = await fetch_data(CRYPTO_SYMBOL, HTF_INTERVAL, HTF_PERIOD)
            if not htf_data.empty:
                global last_htf_trend
                last_htf_trend = calculate_htf_trend(htf_data)
                logging.info(f"Calculated HTF Trend: {last_htf_trend}")
            else:
                logging.warning("Skipping HTF trend calculation due to empty data.")
                last_htf_trend = "UNKNOWN" # Reset if data fetch fails

            # 2. Fetch LTF data for signals and execution
            ltf_data = await fetch_data(CRYPTO_SYMBOL, LTF_INTERVAL, LTF_PERIOD)
            if not ltf_data.empty:
                ltf_data_with_indicators = calculate_indicators(ltf_data.copy()) # Use a copy to avoid SettingWithCopyWarning
                await trading_logic(ltf_data_with_indicators, last_htf_trend)
            else:
                logging.warning("Skipping LTF trading logic due to empty data.")

            # 3. Check for active trade stop-loss/take-profit (even if no new signal)
            if last_ltf_price and active_trade["is_active"]:
                await check_active_trade_management(last_ltf_price)

        else:
            logging.info(f"Outside defined trading hours ({TRADING_START_HOUR_UTC}:00-{TRADING_END_HOUR_UTC}:00 UTC). Holding.")
            if current_position_units > 0:
                logging.info("Holding existing position as outside trading hours.")
            else:
                logging.info("No active trades. Waiting for trading hours to resume.")

        # 4. Report current status periodically
        current_value = current_capital_usd
        if current_position_units > 0 and last_ltf_price:
            current_value += (current_position_units * last_ltf_price)
        elif current_position_units > 0 and active_trade["is_active"]:
            # If no recent price, estimate value based on entry price if trade is active
            current_value += (current_position_units * active_trade["entry_price"])

        status_message = (
            f"📊 Current Status for {CRYPTO_SYMBOL} ({LTF_INTERVAL}):\n"
            f"Current UTC Hour: {current_utc_hour:02d}\n"
            f"HTF Trend: {last_htf_trend}\n"
            f"Price: ${last_ltf_price:.2f} (latest LTF)\n"
            f"Position: {'LONG' if current_position_units > 0 else 'FLAT'} ({current_position_units:.4f} units)\n"
            f"Simulated Capital: ${current_capital_usd:.2f}\n"
            f"Total Simulated Value: ${current_value:.2f}"
        )
        logging.info(status_message)
        await send_telegram_message(status_message)

        # 5. Wait for the next interval.
        # The sleep duration should ideally align with the LTF_INTERVAL for data freshness.
        sleep_seconds = 300 # Default to 5 minutes
        if LTF_INTERVAL == "1m":
            sleep_seconds = 60
        elif LTF_INTERVAL == "5m":
            sleep_seconds = 300
        elif LTF_INTERVAL == "1h":
            sleep_seconds = 3600
        elif LTF_INTERVAL == "1d":
            sleep_seconds = 86400
        logging.info(f"Waiting for {sleep_seconds} seconds until next data fetch cycle...")
        await asyncio.sleep(sleep_seconds)


if __name__ == "__main__":
    # Ensure you have installed the required libraries:
    # pip install -r requirements.txt
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        logging.info("Bot stopped by user.")
    except Exception as e:
        logging.critical(f"An unhandled error occurred: {e}", exc_info=True) # exc_info to print traceback
