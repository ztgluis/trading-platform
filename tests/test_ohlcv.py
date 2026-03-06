"""Tests for OHLCV model and validation."""

import pandas as pd
import numpy as np
import pytest
from datetime import timezone

from trade_analysis.models.ohlcv import (
    AssetClass,
    OHLCVMeta,
    Timeframe,
    attach_metadata,
    create_empty_ohlcv,
    validate_ohlcv,
)
from trade_analysis.exceptions import OHLCVValidationError


def make_valid_ohlcv(rows: int = 5) -> pd.DataFrame:
    """Create a valid canonical OHLCV DataFrame."""
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC")
    rng = np.random.default_rng(42)
    closes = 150.0 + rng.standard_normal(rows).cumsum()
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": closes - rng.uniform(0, 2, rows),
        "high": closes + rng.uniform(1, 5, rows),
        "low": closes - rng.uniform(1, 5, rows),
        "close": closes,
        "volume": rng.uniform(1e6, 1e7, rows),
    })


class TestValidateOHLCV:
    def test_valid_dataframe_passes(self):
        df = make_valid_ohlcv()
        assert validate_ohlcv(df) is True

    def test_missing_column_raises(self):
        df = make_valid_ohlcv()
        df = df.drop(columns=["volume"])
        with pytest.raises(OHLCVValidationError, match="Missing columns"):
            validate_ohlcv(df)

    def test_non_utc_timestamp_raises(self):
        df = make_valid_ohlcv()
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        with pytest.raises(OHLCVValidationError, match="not timezone-aware"):
            validate_ohlcv(df)

    def test_non_datetime_timestamp_raises(self):
        df = make_valid_ohlcv()
        df["timestamp"] = df["timestamp"].astype(str)
        with pytest.raises(OHLCVValidationError, match="not datetime64"):
            validate_ohlcv(df)

    def test_duplicate_timestamps_raises(self):
        df = make_valid_ohlcv()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        with pytest.raises(OHLCVValidationError, match="duplicate timestamp"):
            validate_ohlcv(df)

    def test_unsorted_timestamps_raises(self):
        df = make_valid_ohlcv(10)
        df = df.iloc[::-1].reset_index(drop=True)
        with pytest.raises(OHLCVValidationError, match="not sorted ascending"):
            validate_ohlcv(df)

    def test_negative_price_raises(self):
        df = make_valid_ohlcv()
        df.loc[0, "close"] = -1.0
        with pytest.raises(OHLCVValidationError, match="non-positive"):
            validate_ohlcv(df)

    def test_high_less_than_low_raises(self):
        df = make_valid_ohlcv()
        df.loc[0, "high"] = 100.0
        df.loc[0, "low"] = 200.0
        with pytest.raises(OHLCVValidationError, match="high < low"):
            validate_ohlcv(df)

    def test_negative_volume_raises(self):
        df = make_valid_ohlcv()
        df.loc[0, "volume"] = -100.0
        with pytest.raises(OHLCVValidationError, match="negative"):
            validate_ohlcv(df)

    def test_zero_volume_passes(self):
        """Indices may have 0 volume."""
        df = make_valid_ohlcv()
        df["volume"] = 0.0
        assert validate_ohlcv(df) is True


class TestOHLCVMeta:
    def test_to_dict(self):
        meta = OHLCVMeta(
            symbol="AAPL",
            asset_class=AssetClass.STOCK,
            timeframe=Timeframe.DAILY,
            provider="yfinance",
        )
        d = meta.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["asset_class"] == "stock"
        assert d["timeframe"] == "Daily"
        assert d["is_inverse"] is False

    def test_inverse_meta(self):
        meta = OHLCVMeta(
            symbol="AAPL_INV",
            asset_class=AssetClass.STOCK,
            timeframe=Timeframe.DAILY,
            provider="yfinance",
            is_inverse=True,
            inverse_of="AAPL",
        )
        assert meta.is_inverse is True
        assert meta.inverse_of == "AAPL"

    def test_attach_metadata(self):
        df = make_valid_ohlcv()
        meta = OHLCVMeta(
            symbol="AAPL",
            asset_class=AssetClass.STOCK,
            timeframe=Timeframe.DAILY,
            provider="yfinance",
        )
        attach_metadata(df, meta)
        assert df.attrs["symbol"] == "AAPL"
        assert df.attrs["timeframe"] == "Daily"


class TestCreateEmptyOHLCV:
    def test_has_correct_columns(self):
        df = create_empty_ohlcv()
        assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]

    def test_is_empty(self):
        df = create_empty_ohlcv()
        assert len(df) == 0


class TestTimeframeEnum:
    def test_values(self):
        assert Timeframe.H4.value == "4H"
        assert Timeframe.DAILY.value == "Daily"
        assert Timeframe.WEEKLY.value == "Weekly"

    def test_from_string(self):
        assert Timeframe("Daily") == Timeframe.DAILY
        assert Timeframe("4H") == Timeframe.H4


class TestAssetClassEnum:
    def test_values(self):
        assert AssetClass.STOCK.value == "stock"
        assert AssetClass.CRYPTO.value == "crypto"
        assert AssetClass.METAL.value == "metal"
