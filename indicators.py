def author():
    return "mwang709"

def study_group():
    return "mwang709"

# Indicator 1: Simple Moving Average (SMA)
def sma(prices, window=20):
    return prices.rolling(window=window).mean()

# Indicator 2: Bollinger Bands
def bb(prices, window=20, num_std=2):
    sma_values = sma(prices, window)
    std = prices.rolling(window=window).std()
    upper_band = sma_values + (num_std * std)
    lower_band = sma_values - (num_std * std)
    bb_value = (prices - sma_values) / (2 * std)
    return upper_band, lower_band, bb_value

# Indicator 3: Relative Strength Index (RSI)
def rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Indicator 4: Momentum
def mmt(prices, window=20):
    return prices / prices.shift(window) - 1

# Indicator 5: Moving Average Convergence Divergence (MACD)
def macd(prices, short=12, long=26, signal=9):
    short_ema = prices.ewm(span=short, adjust=False).mean()
    long_ema = prices.ewm(span=long, adjust=False).mean()
    macd_line = short_ema - long_ema
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, macd_signal, macd_line - macd_signal
