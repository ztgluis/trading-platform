"""Composite trend signal systems: Zero Lag Trend Signals, Volumatic VIDYA.

These are complete signal-generating systems built from lower-level indicators.
Each adds trend state, buy/sell signals, and confirmation columns.
Based on TradingView indicators by AlgoAlpha and BigBeluga.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta


# ---------------------------------------------------------------------------
# Shared: Trend State Machine (Supertrend-like hysteresis latch)
# ---------------------------------------------------------------------------

def trend_state_machine(
    close: pd.Series,
    upper_band: pd.Series,
    lower_band: pd.Series,
) -> pd.Series:
    """Compute trend direction using band-crossover hysteresis.

    Trend flips to +1 when close crosses above upper_band,
    flips to -1 when close crosses below lower_band.
    Otherwise retains its previous state (latch behavior).

    Args:
        close: Close price series.
        upper_band: Upper threshold band.
        lower_band: Lower threshold band.

    Returns:
        Series of trend values: +1 (bullish) or -1 (bearish).
    """
    n = len(close)
    trend = np.zeros(n, dtype=int)

    for i in range(1, n):
        if np.isnan(upper_band.iloc[i]) or np.isnan(lower_band.iloc[i]):
            trend[i] = trend[i - 1]
            continue

        if close.iloc[i] > upper_band.iloc[i] and close.iloc[i - 1] <= upper_band.iloc[i - 1]:
            trend[i] = 1
        elif close.iloc[i] < lower_band.iloc[i] and close.iloc[i - 1] >= lower_band.iloc[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]

    return pd.Series(trend, index=close.index, name="trend")


# ---------------------------------------------------------------------------
# Zero Lag Trend Signals  [AlgoAlpha]
# ---------------------------------------------------------------------------

def add_zero_lag_trend_signals(
    df: pd.DataFrame,
    length: int = 70,
    multiplier: float = 1.2,
) -> pd.DataFrame:
    """Add Zero Lag Trend Signals columns.

    Algorithm:
        1. ZLEMA = EMA(close + (close - close[lag]), length)
        2. Volatility = highest(ATR(length), length * 3) * multiplier
        3. Trend state machine: +1 when close > zlema + vol, -1 when close < zlema - vol
        4. Reversal signals on trend flips
        5. Pullback entry signals on ZLEMA crossovers within established trend

    Args:
        df: Canonical OHLCV DataFrame.
        length: ZLEMA and ATR lookback period (default: 70).
        multiplier: ATR band width multiplier (default: 1.2).

    Returns:
        DataFrame with columns:
            - zlts_zlema: Zero-Lag EMA value
            - zlts_upper: upper band (zlema + volatility)
            - zlts_lower: lower band (zlema - volatility)
            - zlts_trend: +1 bullish / -1 bearish
            - zlts_trend_buy: bool reversal buy signal
            - zlts_trend_sell: bool reversal sell signal
            - zlts_entry_buy: bool pullback buy entry
            - zlts_entry_sell: bool pullback sell entry
    """
    result = df.copy()
    close = df["close"]

    # Step 1: ZLEMA
    lag = int(np.floor((length - 1) / 2))
    adjusted = close + (close - close.shift(lag))
    zlema = ta.ema(adjusted, length=length)

    # Step 2: ATR-based volatility bands
    atr = ta.atr(high=df["high"], low=df["low"], close=close, length=length)
    # Highest ATR over 3x the period for stable bands
    volatility = atr.rolling(length * 3).max() * multiplier

    upper = zlema + volatility
    lower = zlema - volatility

    # Step 3: Trend state machine
    trend = trend_state_machine(close, upper, lower)

    # Step 4: Reversal signals (trend flips)
    trend_buy = (trend == 1) & (trend.shift(1) == -1)
    trend_sell = (trend == -1) & (trend.shift(1) == 1)

    # Step 5: Pullback entry signals
    # Buy: close crosses above zlema while already in uptrend
    cross_above_zlema = (close > zlema) & (close.shift(1) <= zlema.shift(1))
    entry_buy = cross_above_zlema & (trend == 1) & (trend.shift(1) == 1)

    # Sell: close crosses below zlema while already in downtrend
    cross_below_zlema = (close < zlema) & (close.shift(1) >= zlema.shift(1))
    entry_sell = cross_below_zlema & (trend == -1) & (trend.shift(1) == -1)

    result["zlts_zlema"] = zlema
    result["zlts_upper"] = upper
    result["zlts_lower"] = lower
    result["zlts_trend"] = trend
    result["zlts_trend_buy"] = trend_buy.fillna(False)
    result["zlts_trend_sell"] = trend_sell.fillna(False)
    result["zlts_entry_buy"] = entry_buy.fillna(False)
    result["zlts_entry_sell"] = entry_sell.fillna(False)

    return result


# ---------------------------------------------------------------------------
# Volumatic VIDYA  [BigBeluga]
# ---------------------------------------------------------------------------

def _compute_vidya(
    src: np.ndarray,
    length: int,
    momentum_period: int,
) -> np.ndarray:
    """Compute raw VIDYA (before final SMA smoothing).

    Internal helper matching BigBeluga's Pine Script implementation.
    """
    n = len(src)

    # Bar-by-bar momentum
    mom = np.diff(src, prepend=np.nan)

    # Rolling sums of positive and negative momentum
    pos_mom = np.where(np.isnan(mom), 0.0, np.where(mom >= 0, mom, 0.0))
    neg_mom = np.where(np.isnan(mom), 0.0, np.where(mom < 0, -mom, 0.0))

    sum_pos = pd.Series(pos_mom).rolling(momentum_period).sum().values
    sum_neg = pd.Series(neg_mom).rolling(momentum_period).sum().values

    # Absolute CMO
    total = sum_pos + sum_neg
    with np.errstate(divide="ignore", invalid="ignore"):
        abs_cmo = np.abs(100.0 * (sum_pos - sum_neg) / total)
    abs_cmo = np.where(np.isfinite(abs_cmo), abs_cmo, 0.0)

    # Adaptive EMA
    alpha = 2.0 / (length + 1)
    vidya = np.full(n, np.nan)
    vidya[0] = src[0]
    for i in range(1, n):
        if np.isnan(vidya[i - 1]):
            vidya[i] = src[i]
        else:
            factor = alpha * abs_cmo[i] / 100.0
            vidya[i] = factor * src[i] + (1.0 - factor) * vidya[i - 1]

    return vidya


def add_volumatic_vidya(
    df: pd.DataFrame,
    vidya_length: int = 10,
    vidya_momentum: int = 20,
    band_distance: float = 2.0,
    atr_period: int = 200,
    smoothing: int = 15,
) -> pd.DataFrame:
    """Add Volumatic VIDYA trend system columns.

    Algorithm:
        1. VIDYA = adaptive MA using CMO as speed factor, SMA-smoothed
        2. Bands = VIDYA +/- ATR(200) * band_distance
        3. Trend state machine on band crossovers
        4. Volume delta: cumulative buy vs sell volume within each trend phase

    Args:
        df: Canonical OHLCV DataFrame.
        vidya_length: Base VIDYA smoothing length (default: 10).
        vidya_momentum: CMO lookback for VIDYA (default: 20).
        band_distance: ATR multiplier for bands (default: 2.0).
        atr_period: ATR calculation period (default: 200).
        smoothing: Final SMA applied to raw VIDYA (default: 15).

    Returns:
        DataFrame with columns:
            - vvidya: smoothed VIDYA value
            - vvidya_upper: upper band
            - vvidya_lower: lower band
            - vvidya_trend: +1 bullish / -1 bearish
            - vvidya_trend_buy: bool trend reversal buy
            - vvidya_trend_sell: bool trend reversal sell
            - vvidya_vol_delta_pct: volume delta as percentage
    """
    result = df.copy()
    close = df["close"]

    # Step 1: Compute VIDYA + SMA smoothing
    raw_vidya = _compute_vidya(close.values, vidya_length, vidya_momentum)
    vidya_smoothed = pd.Series(raw_vidya, index=df.index).rolling(smoothing).mean()

    # Step 2: ATR bands
    atr = ta.atr(high=df["high"], low=df["low"], close=close, length=atr_period)
    upper = vidya_smoothed + atr * band_distance
    lower = vidya_smoothed - atr * band_distance

    # Step 3: Trend state machine
    trend = trend_state_machine(close, upper, lower)

    # Trend change signals
    trend_buy = (trend == 1) & (trend.shift(1) == -1)
    trend_sell = (trend == -1) & (trend.shift(1) == 1)

    # Step 4: Volume delta within each trend phase
    trend_changed = trend != trend.shift(1)
    # Group bars by trend phase (each flip starts a new group)
    phase_id = trend_changed.cumsum()

    # Classify volume as buy (close > open) or sell (close < open)
    buy_vol = df["volume"].where(close > df["open"], 0.0)
    sell_vol = df["volume"].where(close < df["open"], 0.0)

    # Cumulative within each phase
    cum_buy = buy_vol.groupby(phase_id).cumsum()
    cum_sell = sell_vol.groupby(phase_id).cumsum()

    avg_vol = (cum_buy + cum_sell) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        delta_pct = ((cum_buy - cum_sell) / avg_vol * 100).replace(
            [np.inf, -np.inf], 0.0
        )
    delta_pct = delta_pct.fillna(0.0)

    result["vvidya"] = vidya_smoothed
    result["vvidya_upper"] = upper
    result["vvidya_lower"] = lower
    result["vvidya_trend"] = trend
    result["vvidya_trend_buy"] = trend_buy.fillna(False)
    result["vvidya_trend_sell"] = trend_sell.fillna(False)
    result["vvidya_vol_delta_pct"] = delta_pct

    return result
