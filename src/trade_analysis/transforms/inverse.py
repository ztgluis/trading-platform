"""Compute synthetic inverse price series for short-side signal detection."""

import pandas as pd

from trade_analysis.models.ohlcv import validate_ohlcv


def compute_inverse(
    df: pd.DataFrame,
    reference_price: float | None = None,
) -> pd.DataFrame:
    """Create a synthetic inverse OHLCV series.

    Uses R^2/price scaling so the inverse series stays in a human-readable
    price range. A 5% drop in the original appears as a ~5.26% rise in
    the inverse.

    OHLCV mapping:
    - open:   R^2 / original_open
    - high:   R^2 / original_low   (swapped: reciprocal is decreasing)
    - low:    R^2 / original_high  (swapped)
    - close:  R^2 / original_close
    - volume: copied as-is
    - timestamp: copied as-is

    Args:
        df: Canonical OHLCV DataFrame.
        reference_price: Scaling factor. Defaults to first close value.

    Returns:
        New DataFrame with inverted prices and updated metadata.
    """
    if len(df) == 0:
        return df.copy()

    if reference_price is None:
        reference_price = float(df["close"].iloc[0])

    r_squared = reference_price ** 2

    inv = df.copy()
    inv["open"] = r_squared / df["open"]
    inv["high"] = r_squared / df["low"]    # Swap: original low → inverse high
    inv["low"] = r_squared / df["high"]    # Swap: original high → inverse low
    inv["close"] = r_squared / df["close"]
    # Volume stays unchanged

    # Update metadata
    original_symbol = inv.attrs.get("symbol", "unknown")
    inv.attrs["is_inverse"] = True
    inv.attrs["inverse_of"] = original_symbol
    inv.attrs["symbol"] = f"{original_symbol}_INV"

    validate_ohlcv(inv)
    return inv
