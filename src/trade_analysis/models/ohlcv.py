"""Canonical OHLCV schema, enums, and validation."""

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from trade_analysis.exceptions import OHLCVValidationError


class Timeframe(Enum):
    H1 = "1H"
    H4 = "4H"
    DAILY = "Daily"
    WEEKLY = "Weekly"
    MONTHLY = "Monthly"


class AssetClass(Enum):
    STOCK = "stock"
    ETF = "etf"
    INDEX = "index"
    CRYPTO = "crypto"
    METAL = "metal"


# Canonical column names — all DataFrames must conform to this
OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

OHLCV_DTYPES = {
    "timestamp": "datetime64[ns, UTC]",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
}


@dataclass(frozen=True)
class OHLCVMeta:
    """Metadata attached to every OHLCV DataFrame (stored as df.attrs)."""

    symbol: str
    asset_class: AssetClass
    timeframe: Timeframe
    provider: str
    is_inverse: bool = False
    inverse_of: str | None = None

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "asset_class": self.asset_class.value,
            "timeframe": self.timeframe.value,
            "provider": self.provider,
            "is_inverse": self.is_inverse,
            "inverse_of": self.inverse_of,
        }


def attach_metadata(df: pd.DataFrame, meta: OHLCVMeta) -> pd.DataFrame:
    """Attach OHLCVMeta to a DataFrame's attrs."""
    df.attrs.update(meta.to_dict())
    return df


def validate_ohlcv(df: pd.DataFrame) -> bool:
    """Validate that a DataFrame conforms to canonical OHLCV schema.

    Raises OHLCVValidationError with details on failure.
    Returns True if valid.
    """
    errors = []

    # Check required columns
    missing = set(OHLCV_COLUMNS) - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {missing}")

    if errors:
        raise OHLCVValidationError("; ".join(errors))

    # Check dtypes
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        errors.append("'timestamp' is not datetime64")
    elif df["timestamp"].dt.tz is None:
        errors.append("'timestamp' is not timezone-aware (must be UTC)")

    for col in ["open", "high", "low", "close", "volume"]:
        if not pd.api.types.is_float_dtype(df[col]):
            errors.append(f"'{col}' is not float64, got {df[col].dtype}")

    if errors:
        raise OHLCVValidationError("; ".join(errors))

    # Check no duplicate timestamps
    if df["timestamp"].duplicated().any():
        dup_count = df["timestamp"].duplicated().sum()
        errors.append(f"{dup_count} duplicate timestamp(s)")

    # Check sorted ascending
    if not df["timestamp"].is_monotonic_increasing:
        errors.append("Timestamps are not sorted ascending")

    # Check no negative prices
    for col in ["open", "high", "low", "close"]:
        if (df[col] <= 0).any():
            errors.append(f"'{col}' contains non-positive values")

    # Check high >= low
    invalid_bars = df["high"] < df["low"]
    if invalid_bars.any():
        errors.append(f"{invalid_bars.sum()} bar(s) where high < low")

    # Check volume >= 0
    if (df["volume"] < 0).any():
        errors.append("'volume' contains negative values")

    if errors:
        raise OHLCVValidationError("; ".join(errors))

    return True


def create_empty_ohlcv() -> pd.DataFrame:
    """Return an empty DataFrame with the correct schema."""
    df = pd.DataFrame(columns=OHLCV_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype("float64")
    return df
