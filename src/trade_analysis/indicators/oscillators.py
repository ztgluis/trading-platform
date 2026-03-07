"""Composite oscillators: Two-Pole Oscillator, Momentum Bias Index.

These are higher-level indicators built from primitives (SMA, EMA, HMA, z-score).
Based on TradingView indicators by BigBeluga and AlgoAlpha.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _two_pole_filter(values: np.ndarray, length: int) -> np.ndarray:
    """Apply a cascaded double-EMA (two-pole) smoothing filter.

    Two sequential first-order EMA passes create a second-order low-pass filter
    with steeper noise rolloff than a single EMA of the same period.

    Args:
        values: Input array (e.g., z-scored deviation).
        length: Filter length (alpha = 2 / (length + 1)).

    Returns:
        Smoothed array (same length as input).
    """
    alpha = 2.0 / (length + 1)
    n = len(values)
    smooth1 = np.full(n, np.nan)
    smooth2 = np.full(n, np.nan)

    for i in range(n):
        if np.isnan(values[i]):
            continue
        if np.isnan(smooth1[i - 1]) if i > 0 else True:
            smooth1[i] = values[i]
        else:
            smooth1[i] = (1 - alpha) * smooth1[i - 1] + alpha * values[i]

        if np.isnan(smooth2[i - 1]) if i > 0 else True:
            smooth2[i] = smooth1[i]
        else:
            smooth2[i] = (1 - alpha) * smooth2[i - 1] + alpha * smooth1[i]

    return smooth2


def detect_crossovers(series: pd.Series, signal: pd.Series) -> pd.Series:
    """Detect where series crosses above signal (crossover).

    Returns a boolean Series that is True on the bar where
    series goes from <= signal to > signal.
    """
    prev_below = series.shift(1) <= signal.shift(1)
    curr_above = series > signal
    return prev_below & curr_above


def detect_crossunders(series: pd.Series, signal: pd.Series) -> pd.Series:
    """Detect where series crosses below signal (crossunder).

    Returns a boolean Series that is True on the bar where
    series goes from >= signal to < signal.
    """
    prev_above = series.shift(1) >= signal.shift(1)
    curr_below = series < signal
    return prev_above & curr_below


# ---------------------------------------------------------------------------
# Two-Pole Oscillator  [BigBeluga]
# ---------------------------------------------------------------------------

def add_two_pole_oscillator(
    df: pd.DataFrame,
    filter_length: int = 15,
    zscore_period: int = 25,
    signal_lag: int = 4,
) -> pd.DataFrame:
    """Add Two-Pole Oscillator columns.

    Algorithm:
        1. Z-score of (close - SMA(close, zscore_period))
        2. Two-pole smooth filter (cascaded double-EMA)
        3. Signal line = oscillator lagged by signal_lag bars
        4. Buy = crossover(osc, signal) while osc < 0
        5. Sell = crossunder(osc, signal) while osc > 0

    Args:
        df: Canonical OHLCV DataFrame.
        filter_length: Two-pole filter length (default: 15).
        zscore_period: Period for SMA and z-score normalization (default: 25).
        signal_lag: Bars to lag the signal line (default: 4).

    Returns:
        DataFrame with columns:
            - two_pole: oscillator value
            - two_pole_signal: lagged signal line
            - two_pole_buy: bool buy signals
            - two_pole_sell: bool sell signals
    """
    result = df.copy()
    close = df["close"]

    # Step 1: Z-score normalization
    sma1 = ta.sma(close, length=zscore_period)
    deviation = close - sma1
    dev_mean = ta.sma(deviation, length=zscore_period)
    dev_std = ta.stdev(deviation, length=zscore_period)

    with np.errstate(divide="ignore", invalid="ignore"):
        z_score = (deviation - dev_mean) / dev_std
    z_score = z_score.fillna(0.0)

    # Step 2: Two-pole filter
    filtered = _two_pole_filter(z_score.values, filter_length)
    osc = pd.Series(filtered, index=df.index)

    # Step 3: Signal line (lagged)
    signal = osc.shift(signal_lag)

    # Step 4: Buy/sell signals
    buy = detect_crossovers(osc, signal) & (osc < 0)
    sell = detect_crossunders(osc, signal) & (osc > 0)

    result["two_pole"] = osc
    result["two_pole_signal"] = signal
    result["two_pole_buy"] = buy.fillna(False)
    result["two_pole_sell"] = sell.fillna(False)

    return result


# ---------------------------------------------------------------------------
# Momentum Bias Index  [AlgoAlpha]
# ---------------------------------------------------------------------------

def add_momentum_bias_index(
    df: pd.DataFrame,
    momentum_length: int = 10,
    bias_length: int = 5,
    smooth_length: int = 10,
    impulse_boundary_length: int = 30,
    stdev_multiplier: float = 3.0,
    smooth: bool = True,
) -> pd.DataFrame:
    """Add Momentum Bias Index columns.

    Algorithm:
        1. Volatility-normalized momentum: (close - close[n]) / EMA(range, n) * 100
        2. Split into positive/negative components
        3. Accumulate each over bias_length, optionally smooth with HMA
        4. Adaptive boundary = EMA(avg) + stdev(avg) * multiplier
        5. TP signals when dominant bias peaks while above boundary

    Args:
        df: Canonical OHLCV DataFrame.
        momentum_length: Price momentum lookback (default: 10).
        bias_length: Accumulation window (default: 5).
        smooth_length: HMA smoothing period (default: 10).
        impulse_boundary_length: Boundary EMA/stdev lookback (default: 30).
        stdev_multiplier: Boundary stdev multiplier (default: 3.0).
        smooth: Whether to apply HMA smoothing (default: True).

    Returns:
        DataFrame with columns:
            - mbi_up_bias: accumulated positive momentum bias
            - mbi_down_bias: accumulated negative momentum bias
            - mbi_boundary: adaptive impulse boundary
            - mbi_bullish_tp: bool bullish take-profit signal
            - mbi_bearish_tp: bool bearish take-profit signal
            - mbi_trend: 1 (bullish) / -1 (bearish) / 0 (neutral)
    """
    result = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Step 1: Volatility-normalized momentum
    momentum = close - close.shift(momentum_length)
    avg_range = ta.ema(high - low, length=momentum_length)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = momentum / avg_range * 100
    normalized = normalized.fillna(0.0)

    # Step 2: Separate positive/negative
    mom_up = normalized.clip(lower=0)
    mom_down = normalized.clip(upper=0)

    # Step 3: Accumulate bias
    sum_up = mom_up.rolling(bias_length).sum()
    sum_down = -mom_down.rolling(bias_length).sum()  # make positive

    if smooth:
        up_bias = ta.hma(sum_up, length=smooth_length).clip(lower=0)
        down_bias = ta.hma(sum_down, length=smooth_length).clip(lower=0)
    else:
        up_bias = sum_up.clip(lower=0)
        down_bias = sum_down.clip(lower=0)

    # Fill NaNs after HMA warmup
    up_bias = up_bias.fillna(0.0)
    down_bias = down_bias.fillna(0.0)

    # Step 4: Adaptive impulse boundary
    avg_bias = (up_bias + down_bias) / 2
    boundary = (
        ta.ema(avg_bias, length=impulse_boundary_length)
        + ta.stdev(avg_bias, length=impulse_boundary_length) * stdev_multiplier
    )
    boundary = boundary.fillna(0.0)

    # Step 5: Take-profit signals (momentum exhaustion)
    down_declining = down_bias < down_bias.shift(1)
    bullish_tp = (
        down_declining
        & ~(down_bias.shift(1) < down_bias.shift(2))  # just started declining
        & (down_bias > boundary)
        & (down_bias > up_bias)
    )

    up_declining = up_bias < up_bias.shift(1)
    bearish_tp = (
        up_declining
        & ~(up_bias.shift(1) < up_bias.shift(2))  # just started declining
        & (up_bias > boundary)
        & (up_bias > down_bias)
    )

    # Trend direction
    trend = pd.Series(0, index=df.index)
    trend = trend.where(~(up_bias > down_bias), 1)
    trend = trend.where(~(down_bias > up_bias), -1)

    result["mbi_up_bias"] = up_bias
    result["mbi_down_bias"] = down_bias
    result["mbi_boundary"] = boundary
    result["mbi_bullish_tp"] = bullish_tp.fillna(False)
    result["mbi_bearish_tp"] = bearish_tp.fillna(False)
    result["mbi_trend"] = trend

    return result
