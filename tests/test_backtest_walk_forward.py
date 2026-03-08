"""Tests for M4 walk-forward split generation and runner."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from trade_analysis.backtester.config import BacktestConfig, WalkForwardConfig
from trade_analysis.backtester.models import WalkForwardResult
from trade_analysis.backtester.walk_forward import (
    generate_walk_forward_splits,
    run_walk_forward,
    _slice_df_by_dates,
)
from trade_analysis.exceptions import ConfigError
from trade_analysis.signals.engine import load_signal_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signal_df(
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    freq: str = "B",  # business days
    seed: int = 42,
) -> pd.DataFrame:
    """Create a minimal signal-enriched DataFrame for walk-forward testing.

    Generates daily OHLCV data with signal columns that trigger trades
    periodically (every ~20 bars).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end, freq=freq)
    n = len(dates)

    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    close = np.maximum(close, 10.0)  # keep positive

    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    open_ = close + rng.normal(0, 0.3, n)

    df = pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex(dates, tz="UTC"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(100000, 1000000, n).astype(float),
            # Signal columns
            "regime": "bull",
            "signal_direction": "long",
            "signal_score": 4,
            "signal_tradeable": False,  # default off
            # Exit levels — set relative to close
            "exit_stop": close - 3.0,
            "exit_target": close + 6.0,
            "exit_trail_be": close + 3.0,
        }
    )

    # Make every 20th bar tradeable (periodic entries)
    tradeable_mask = np.zeros(n, dtype=bool)
    tradeable_mask[::20] = True
    df["signal_tradeable"] = tradeable_mask

    return df


# ===========================================================================
# Split generation — anchored
# ===========================================================================


class TestSplitGenerationAnchored:
    """Test anchored walk-forward split generation."""

    def test_single_fold_3is_1oos(self):
        """2020-2024 with IS=3, OOS=1 → 1 fold (anchored)."""
        splits = generate_walk_forward_splits(
            start_date=date(2020, 1, 1),
            end_date=date(2024, 12, 31),
            in_sample_years=3,
            out_of_sample_years=1,
            anchored=True,
        )
        assert len(splits) == 1
        s = splits[0]
        assert s.fold == 0
        assert s.is_start == date(2020, 1, 1)
        assert s.is_end == date(2023, 1, 1)
        assert s.oos_start == date(2023, 1, 1)
        assert s.oos_end == date(2024, 1, 1)

    def test_two_folds_2is_1oos(self):
        """2020-2024 with IS=2, OOS=1, anchored → 2 folds."""
        splits = generate_walk_forward_splits(
            start_date=date(2020, 1, 1),
            end_date=date(2024, 12, 31),
            in_sample_years=2,
            out_of_sample_years=1,
            anchored=True,
        )
        assert len(splits) == 2

        # Fold 0: IS [2020, 2022), OOS [2022, 2023)
        assert splits[0].is_start == date(2020, 1, 1)
        assert splits[0].is_end == date(2022, 1, 1)
        assert splits[0].oos_start == date(2022, 1, 1)
        assert splits[0].oos_end == date(2023, 1, 1)

        # Fold 1: IS [2020, 2023), OOS [2023, 2024) (IS expanded)
        assert splits[1].is_start == date(2020, 1, 1)
        assert splits[1].is_end == date(2023, 1, 1)
        assert splits[1].oos_start == date(2023, 1, 1)
        assert splits[1].oos_end == date(2024, 1, 1)

    def test_anchored_is_always_starts_at_start(self):
        """All anchored folds start IS at the same date."""
        splits = generate_walk_forward_splits(
            start_date=date(2015, 1, 1),
            end_date=date(2024, 12, 31),
            in_sample_years=2,
            out_of_sample_years=1,
            anchored=True,
        )
        for s in splits:
            assert s.is_start == date(2015, 1, 1)

    def test_anchored_oos_windows_are_contiguous(self):
        """OOS windows should be contiguous (no gaps, no overlaps)."""
        splits = generate_walk_forward_splits(
            start_date=date(2015, 1, 1),
            end_date=date(2024, 12, 31),
            in_sample_years=2,
            out_of_sample_years=1,
            anchored=True,
        )
        assert len(splits) >= 2
        for i in range(1, len(splits)):
            assert splits[i].oos_start == splits[i - 1].oos_end


# ===========================================================================
# Split generation — rolling
# ===========================================================================


class TestSplitGenerationRolling:
    """Test rolling walk-forward split generation."""

    def test_rolling_single_fold(self):
        """2020-2024 with IS=3, OOS=1, rolling → 1 fold."""
        splits = generate_walk_forward_splits(
            start_date=date(2020, 1, 1),
            end_date=date(2024, 12, 31),
            in_sample_years=3,
            out_of_sample_years=1,
            anchored=False,
        )
        assert len(splits) == 1
        s = splits[0]
        assert s.is_start == date(2020, 1, 1)
        assert s.is_end == date(2023, 1, 1)
        assert s.oos_start == date(2023, 1, 1)
        assert s.oos_end == date(2024, 1, 1)

    def test_rolling_two_folds(self):
        """2020-2025 with IS=3, OOS=1, rolling → 2 folds."""
        splits = generate_walk_forward_splits(
            start_date=date(2020, 1, 1),
            end_date=date(2025, 12, 31),
            in_sample_years=3,
            out_of_sample_years=1,
            anchored=False,
        )
        assert len(splits) == 2

        # Fold 0: IS [2020, 2023), OOS [2023, 2024)
        assert splits[0].is_start == date(2020, 1, 1)
        assert splits[0].is_end == date(2023, 1, 1)

        # Fold 1: IS [2021, 2024), OOS [2024, 2025) — IS slides forward
        assert splits[1].is_start == date(2021, 1, 1)
        assert splits[1].is_end == date(2024, 1, 1)

    def test_rolling_is_fixed_size(self):
        """All rolling folds have the same IS length."""
        splits = generate_walk_forward_splits(
            start_date=date(2015, 1, 1),
            end_date=date(2024, 12, 31),
            in_sample_years=3,
            out_of_sample_years=1,
            anchored=False,
        )
        for s in splits:
            # IS should be approximately 3 years
            is_days = (s.is_end - s.is_start).days
            assert 1090 <= is_days <= 1100  # ~3 years ± leap year

    def test_rolling_no_oos_overlap(self):
        """OOS windows should not overlap for rolling splits."""
        splits = generate_walk_forward_splits(
            start_date=date(2015, 1, 1),
            end_date=date(2024, 12, 31),
            in_sample_years=3,
            out_of_sample_years=1,
            anchored=False,
        )
        for i in range(1, len(splits)):
            assert splits[i].oos_start >= splits[i - 1].oos_end


# ===========================================================================
# Split generation — edge cases
# ===========================================================================


class TestSplitEdgeCases:
    """Test edge cases and errors for split generation."""

    def test_insufficient_data_raises(self):
        """Date range shorter than IS+OOS raises ConfigError."""
        with pytest.raises(ConfigError, match="too short"):
            generate_walk_forward_splits(
                start_date=date(2020, 1, 1),
                end_date=date(2022, 6, 1),
                in_sample_years=3,
                out_of_sample_years=1,
            )

    def test_exactly_one_fold_boundary(self):
        """Date range exactly equals IS+OOS → 1 fold."""
        # IS=3, OOS=1 → need 4 years exactly
        splits = generate_walk_forward_splits(
            start_date=date(2020, 1, 1),
            end_date=date(2024, 1, 1),
            in_sample_years=3,
            out_of_sample_years=1,
        )
        assert len(splits) == 1

    def test_fold_numbers_sequential(self):
        """Fold numbers should be 0, 1, 2, ..."""
        splits = generate_walk_forward_splits(
            start_date=date(2015, 1, 1),
            end_date=date(2024, 12, 31),
            in_sample_years=2,
            out_of_sample_years=1,
            anchored=True,
        )
        for i, s in enumerate(splits):
            assert s.fold == i

    def test_oos_never_exceeds_end_date(self):
        """No OOS window should extend past end_date."""
        end = date(2024, 12, 31)
        splits = generate_walk_forward_splits(
            start_date=date(2015, 1, 1),
            end_date=end,
            in_sample_years=2,
            out_of_sample_years=1,
            anchored=True,
        )
        for s in splits:
            assert s.oos_end <= end


# ===========================================================================
# DataFrame slicing
# ===========================================================================


class TestSliceByDates:
    """Test _slice_df_by_dates helper."""

    def test_slices_correctly(self):
        """Slice returns only rows in [start, end)."""
        df = _make_signal_df(start="2020-01-01", end="2024-12-31")
        sliced = _slice_df_by_dates(df, date(2022, 1, 1), date(2023, 1, 1))

        assert len(sliced) > 0
        assert sliced["timestamp"].iloc[0] >= pd.Timestamp("2022-01-01", tz="UTC")
        assert sliced["timestamp"].iloc[-1] < pd.Timestamp("2023-01-01", tz="UTC")

    def test_slice_returns_empty_for_no_data(self):
        """Slice with no matching data returns empty DataFrame."""
        df = _make_signal_df(start="2020-01-01", end="2020-12-31")
        sliced = _slice_df_by_dates(df, date(2025, 1, 1), date(2026, 1, 1))
        assert len(sliced) == 0

    def test_slice_reset_index(self):
        """Sliced DataFrame should have a reset index starting at 0."""
        df = _make_signal_df(start="2020-01-01", end="2024-12-31")
        sliced = _slice_df_by_dates(df, date(2022, 1, 1), date(2023, 1, 1))
        assert sliced.index[0] == 0
        assert sliced.index[-1] == len(sliced) - 1


# ===========================================================================
# Walk-forward runner
# ===========================================================================


class TestRunWalkForward:
    """Test end-to-end walk-forward runner."""

    def test_run_walk_forward_returns_result(self):
        """run_walk_forward returns a WalkForwardResult."""
        df = _make_signal_df(start="2020-01-01", end="2024-12-31")
        config = BacktestConfig(
            start_date=date(2020, 1, 1),
            end_date=date(2024, 12, 31),
            initial_capital=100000.0,
            max_open_positions=1,
            walk_forward=WalkForwardConfig(
                in_sample_years=3,
                out_of_sample_years=1,
                anchored=True,
            ),
        )
        signal_config = load_signal_config()

        result = run_walk_forward(
            df, "AAPL", "stock", "Daily", config, signal_config
        )

        assert isinstance(result, WalkForwardResult)
        assert len(result.splits) == 1
        assert len(result.in_sample_results) == 1
        assert len(result.out_of_sample_results) == 1

    def test_walk_forward_no_config_raises(self):
        """run_walk_forward without walk_forward config raises ConfigError."""
        df = _make_signal_df()
        config = BacktestConfig(
            start_date=date(2020, 1, 1),
            end_date=date(2024, 12, 31),
            initial_capital=100000.0,
            max_open_positions=1,
            walk_forward=None,
        )
        signal_config = load_signal_config()

        with pytest.raises(ConfigError, match="walk_forward"):
            run_walk_forward(df, "AAPL", "stock", "Daily", config, signal_config)

    def test_is_and_oos_results_match_splits(self):
        """Number of IS/OOS results should match number of splits."""
        df = _make_signal_df(start="2015-01-01", end="2024-12-31")
        config = BacktestConfig(
            start_date=date(2015, 1, 1),
            end_date=date(2024, 12, 31),
            initial_capital=100000.0,
            max_open_positions=1,
            walk_forward=WalkForwardConfig(
                in_sample_years=2,
                out_of_sample_years=1,
                anchored=True,
            ),
        )
        signal_config = load_signal_config()

        result = run_walk_forward(
            df, "AAPL", "stock", "Daily", config, signal_config
        )

        assert len(result.in_sample_results) == len(result.splits)
        assert len(result.out_of_sample_results) == len(result.splits)

    def test_is_has_more_trades_than_oos(self):
        """IS period is longer, so it should generally have more trades."""
        df = _make_signal_df(start="2020-01-01", end="2024-12-31")
        config = BacktestConfig(
            start_date=date(2020, 1, 1),
            end_date=date(2024, 12, 31),
            initial_capital=100000.0,
            max_open_positions=1,
            walk_forward=WalkForwardConfig(
                in_sample_years=3,
                out_of_sample_years=1,
                anchored=True,
            ),
        )
        signal_config = load_signal_config()

        result = run_walk_forward(
            df, "AAPL", "stock", "Daily", config, signal_config
        )

        # IS has 3 years of data, OOS has 1 year — IS should have more trades
        is_trades = len(result.in_sample_results[0].trades)
        oos_trades = len(result.out_of_sample_results[0].trades)
        assert is_trades >= oos_trades

    def test_rolling_walk_forward(self):
        """Rolling walk-forward runs correctly."""
        df = _make_signal_df(start="2018-01-01", end="2025-12-31")
        config = BacktestConfig(
            start_date=date(2018, 1, 1),
            end_date=date(2025, 12, 31),
            initial_capital=100000.0,
            max_open_positions=1,
            walk_forward=WalkForwardConfig(
                in_sample_years=3,
                out_of_sample_years=1,
                anchored=False,
            ),
        )
        signal_config = load_signal_config()

        result = run_walk_forward(
            df, "AAPL", "stock", "Daily", config, signal_config
        )

        assert isinstance(result, WalkForwardResult)
        assert len(result.splits) >= 2
        # Each split should have a result
        assert len(result.in_sample_results) == len(result.splits)
