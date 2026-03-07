"""Indicator library — pure parameterized functions for technical analysis."""

from trade_analysis.indicators.trend import (
    add_atr,
    add_ema,
    add_hma,
    add_ma,
    add_sma,
    add_vidya,
    add_zlema,
)
from trade_analysis.indicators.momentum import add_macd, add_rsi, add_rsi_direction
from trade_analysis.indicators.structure import (
    detect_higher_lows,
    detect_lower_highs,
    detect_swing_highs,
    detect_swing_lows,
)
from trade_analysis.indicators.volume import add_volume_sma, detect_volume_spike
from trade_analysis.indicators.levels import (
    detect_pivot_levels,
    detect_round_numbers,
    find_nearest_level,
)
from trade_analysis.indicators.oscillators import (
    add_momentum_bias_index,
    add_two_pole_oscillator,
    detect_crossovers,
    detect_crossunders,
)
from trade_analysis.indicators.signals import (
    add_volumatic_vidya,
    add_zero_lag_trend_signals,
    trend_state_machine,
)

__all__ = [
    # Trend
    "add_sma",
    "add_ema",
    "add_hma",
    "add_zlema",
    "add_vidya",
    "add_atr",
    "add_ma",
    # Momentum
    "add_rsi",
    "add_rsi_direction",
    "add_macd",
    # Structure
    "detect_swing_highs",
    "detect_swing_lows",
    "detect_higher_lows",
    "detect_lower_highs",
    # Volume
    "add_volume_sma",
    "detect_volume_spike",
    # Levels
    "detect_pivot_levels",
    "detect_round_numbers",
    "find_nearest_level",
    # Oscillators
    "add_two_pole_oscillator",
    "add_momentum_bias_index",
    "detect_crossovers",
    "detect_crossunders",
    # Composite Signals
    "add_zero_lag_trend_signals",
    "add_volumatic_vidya",
    "trend_state_machine",
]
