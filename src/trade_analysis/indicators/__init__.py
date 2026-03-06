"""Indicator library — pure parameterized functions for technical analysis."""

from trade_analysis.indicators.trend import add_ema, add_ma, add_sma
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

__all__ = [
    # Trend
    "add_sma",
    "add_ema",
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
]
