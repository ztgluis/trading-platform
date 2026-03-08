"""Tests for live signal handler — extraction and mapping logic."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from trade_analysis.live.signal_handler import (
    _safe_float,
    compute_config_hash,
    extract_latest_signal,
    lookup_symbol,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def signal_df() -> pd.DataFrame:
    """DataFrame simulating generate_signals() output with a tradeable signal."""
    n = 5
    timestamps = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": [100.0] * n,
        "high": [105.0] * n,
        "low": [95.0] * n,
        "close": [102.0] * n,
        "volume": [1_000_000.0] * n,
        # Signal columns (latest bar has a long signal)
        "regime": ["bull"] * n,
        "regime_distance_pct": [5.0] * n,
        "regime_strongly_aligned": [True] * n,
        "signal_direction": [None, None, None, None, "long"],
        "signal_conditions_met": [0, 0, 0, 0, 2],
        "trend_bull": [False, False, False, False, True],
        "trend_bear": [False] * n,
        "structure_bull": [False, False, False, False, True],
        "structure_bear": [False] * n,
        "structure_multi_method": [False, False, False, False, True],
        "momentum_bull": [False] * n,
        "momentum_bear": [False] * n,
        "volume_spike": [False, False, False, False, True],
        "signal_score": [0, 0, 0, 0, 4],
        "signal_tradeable": [False, False, False, False, True],
        "entry_price": [None, None, None, None, 102.0],
        "exit_stop": [None, None, None, None, 95.0],
        "exit_target": [None, None, None, None, 116.0],
        "exit_trail_be": [None, None, None, None, 109.0],
        "exit_risk": [None, None, None, None, 7.0],
        "exit_reward": [None, None, None, None, 14.0],
        "exit_rr_ratio": [None, None, None, None, 2.0],
    })
    return df


@pytest.fixture()
def no_signal_df() -> pd.DataFrame:
    """DataFrame where the latest bar has no signal."""
    n = 3
    timestamps = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": [100.0] * n,
        "high": [105.0] * n,
        "low": [95.0] * n,
        "close": [102.0] * n,
        "volume": [1_000_000.0] * n,
        "signal_direction": [None, None, None],
        "signal_conditions_met": [0, 0, 0],
        "signal_score": [0, 0, 0],
        "signal_tradeable": [False, False, False],
        "regime": ["bull"] * n,
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExtractLatestSignal:
    def test_extracts_tradeable_long_signal(self, signal_df: pd.DataFrame) -> None:
        result = extract_latest_signal(signal_df, "AAPL", "stock", "Daily")
        assert result is not None
        assert result["symbol"] == "AAPL"
        assert result["asset_class"] == "stock"
        assert result["timeframe"] == "Daily"
        assert result["signal_direction"] == "long"
        assert result["signal_score"] == 4
        assert result["signal_tradeable"] is True
        assert result["bucket"] == "A"

    def test_returns_none_for_no_signal(self, no_signal_df: pd.DataFrame) -> None:
        result = extract_latest_signal(no_signal_df, "AAPL", "stock", "Daily")
        assert result is None

    def test_returns_none_for_empty_df(self) -> None:
        result = extract_latest_signal(pd.DataFrame(), "AAPL", "stock", "Daily")
        assert result is None

    def test_maps_exit_levels(self, signal_df: pd.DataFrame) -> None:
        result = extract_latest_signal(signal_df, "AAPL", "stock", "Daily")
        assert result["entry_price"] == 102.0
        assert result["exit_stop"] == 95.0
        assert result["exit_target"] == 116.0
        assert result["exit_rr_ratio"] == 2.0

    def test_maps_conditions(self, signal_df: pd.DataFrame) -> None:
        result = extract_latest_signal(signal_df, "AAPL", "stock", "Daily")
        assert result["trend_confirmed"] is True
        assert result["structure_confirmed"] is True
        assert result["structure_multi_method"] is True
        assert result["momentum_confirmed"] is False
        assert result["volume_spike"] is True

    def test_maps_regime(self, signal_df: pd.DataFrame) -> None:
        result = extract_latest_signal(signal_df, "AAPL", "stock", "Daily")
        assert result["regime"] == "bull"
        assert result["regime_distance_pct"] == 5.0
        assert result["regime_strongly_aligned"] is True

    def test_bucket_b_for_index(self, signal_df: pd.DataFrame) -> None:
        result = extract_latest_signal(signal_df, "^GSPC", "index", "Weekly")
        assert result is not None
        assert result["bucket"] == "B"

    def test_config_hash_nonempty(self, signal_df: pd.DataFrame) -> None:
        result = extract_latest_signal(signal_df, "AAPL", "stock", "Daily")
        assert result["config_hash"]
        assert len(result["config_hash"]) == 64  # SHA-256 hex digest


class TestSafeFloat:
    def test_normal_float(self) -> None:
        assert _safe_float(3.14) == 3.14

    def test_int_to_float(self) -> None:
        assert _safe_float(5) == 5.0

    def test_none_returns_none(self) -> None:
        assert _safe_float(None) is None

    def test_nan_returns_none(self) -> None:
        assert _safe_float(float("nan")) is None

    def test_numpy_nan_returns_none(self) -> None:
        assert _safe_float(np.nan) is None

    def test_string_number(self) -> None:
        assert _safe_float("3.14") == 3.14


class TestComputeConfigHash:
    def test_returns_hex_string(self) -> None:
        h = compute_config_hash()
        assert isinstance(h, str)
        assert len(h) == 64

    def test_consistent(self) -> None:
        h1 = compute_config_hash()
        h2 = compute_config_hash()
        assert h1 == h2


class TestLookupSymbol:
    def test_finds_known_symbol(self) -> None:
        sym = lookup_symbol("AAPL")
        assert sym is not None
        assert sym.ticker == "AAPL"
        assert sym.asset_class == "stock"

    def test_case_insensitive(self) -> None:
        sym = lookup_symbol("aapl")
        assert sym is not None
        assert sym.ticker == "AAPL"

    def test_returns_none_for_unknown(self) -> None:
        sym = lookup_symbol("ZZZZZZ")
        assert sym is None
