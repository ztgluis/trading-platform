"""Tests for inverse price series computation."""

import numpy as np
import pandas as pd
import pytest

from trade_analysis.models.ohlcv import validate_ohlcv
from trade_analysis.transforms.inverse import compute_inverse


class TestComputeInverse:
    def test_basic_inverse(self, sample_daily_ohlcv):
        inv = compute_inverse(sample_daily_ohlcv)
        assert len(inv) == len(sample_daily_ohlcv)
        assert validate_ohlcv(inv) is True

    def test_high_low_swap(self, sample_daily_ohlcv):
        """Original low becomes inverse high and vice versa."""
        ref = float(sample_daily_ohlcv["close"].iloc[0])
        r_sq = ref ** 2
        inv = compute_inverse(sample_daily_ohlcv)

        # For each row: inv_high should be R^2 / original_low
        expected_high = r_sq / sample_daily_ohlcv["low"]
        expected_low = r_sq / sample_daily_ohlcv["high"]

        pd.testing.assert_series_equal(
            inv["high"], expected_high, check_names=False
        )
        pd.testing.assert_series_equal(
            inv["low"], expected_low, check_names=False
        )

    def test_close_inverse(self, sample_daily_ohlcv):
        ref = float(sample_daily_ohlcv["close"].iloc[0])
        r_sq = ref ** 2
        inv = compute_inverse(sample_daily_ohlcv)

        expected_close = r_sq / sample_daily_ohlcv["close"]
        pd.testing.assert_series_equal(
            inv["close"], expected_close, check_names=False
        )

    def test_volume_unchanged(self, sample_daily_ohlcv):
        inv = compute_inverse(sample_daily_ohlcv)
        pd.testing.assert_series_equal(
            inv["volume"], sample_daily_ohlcv["volume"], check_names=False
        )

    def test_custom_reference_price(self, sample_daily_ohlcv):
        inv = compute_inverse(sample_daily_ohlcv, reference_price=200.0)
        r_sq = 200.0 ** 2
        expected_close = r_sq / sample_daily_ohlcv["close"]
        pd.testing.assert_series_equal(
            inv["close"], expected_close, check_names=False
        )

    def test_metadata_updated(self, sample_daily_ohlcv):
        sample_daily_ohlcv.attrs["symbol"] = "AAPL"
        inv = compute_inverse(sample_daily_ohlcv)
        assert inv.attrs["is_inverse"] is True
        assert inv.attrs["inverse_of"] == "AAPL"
        assert inv.attrs["symbol"] == "AAPL_INV"

    def test_first_close_as_reference(self, sample_daily_ohlcv):
        """Inverse of first close should equal first close (R^2/R = R)."""
        inv = compute_inverse(sample_daily_ohlcv)
        first_close = float(sample_daily_ohlcv["close"].iloc[0])
        inv_first_close = float(inv["close"].iloc[0])
        assert inv_first_close == pytest.approx(first_close)

    def test_round_trip_approximate(self, sample_daily_ohlcv):
        """Inverse of inverse should approximately equal original."""
        inv = compute_inverse(sample_daily_ohlcv)
        # Use original's first close as reference for double inverse
        ref = float(sample_daily_ohlcv["close"].iloc[0])
        double_inv = compute_inverse(inv, reference_price=ref)

        pd.testing.assert_series_equal(
            double_inv["close"],
            sample_daily_ohlcv["close"],
            check_names=False,
            atol=1e-10,
        )

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        inv = compute_inverse(df)
        assert len(inv) == 0
