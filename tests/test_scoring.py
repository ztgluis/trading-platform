"""Tests for M3 signal scoring (scoring.py)."""

import numpy as np
import pandas as pd
import pytest

from trade_analysis.signals.scoring import (
    DEFAULT_WEIGHTS,
    compute_signal_score,
    determine_signal_direction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signal_df(
    n: int = 10,
    trend_bull: bool = False,
    trend_bear: bool = False,
    structure_bull: bool = False,
    structure_bear: bool = False,
    momentum_bull: bool = False,
    momentum_bear: bool = False,
    regime_allow_long: bool = True,
    regime_allow_short: bool = True,
    regime_strongly_aligned: bool = False,
    structure_multi_method: bool = False,
    volume_spike: bool = False,
) -> pd.DataFrame:
    """Create a DataFrame with condition and regime columns for testing."""
    return pd.DataFrame(
        {
            "close": np.full(n, 100.0),
            "trend_bull": np.full(n, trend_bull),
            "trend_bear": np.full(n, trend_bear),
            "structure_bull": np.full(n, structure_bull),
            "structure_bear": np.full(n, structure_bear),
            "momentum_bull": np.full(n, momentum_bull),
            "momentum_bear": np.full(n, momentum_bear),
            "regime_allow_long": np.full(n, regime_allow_long),
            "regime_allow_short": np.full(n, regime_allow_short),
            "regime_strongly_aligned": np.full(n, regime_strongly_aligned),
            "structure_multi_method": np.full(n, structure_multi_method),
            "volume_spike": np.full(n, volume_spike),
        }
    )


# ===========================================================================
# determine_signal_direction
# ===========================================================================


class TestDetermineSignalDirection:
    """Test 2-of-3 condition gate with regime filtering."""

    def test_output_columns(self):
        """Adds signal_direction and signal_conditions_met."""
        df = _make_signal_df()
        result = determine_signal_direction(df)
        assert "signal_direction" in result.columns
        assert "signal_conditions_met" in result.columns

    def test_no_signal_zero_conditions(self):
        """No conditions met → no signal direction."""
        df = _make_signal_df()
        result = determine_signal_direction(df)
        assert (result["signal_direction"].isna()).all()
        assert (result["signal_conditions_met"] == 0).all()

    def test_one_condition_not_enough(self):
        """Only 1 of 3 conditions → no signal."""
        df = _make_signal_df(trend_bull=True)
        result = determine_signal_direction(df)
        assert (result["signal_direction"].isna()).all()

    def test_two_conditions_long(self):
        """2 of 3 bullish conditions → long signal."""
        df = _make_signal_df(trend_bull=True, momentum_bull=True)
        result = determine_signal_direction(df)
        assert (result["signal_direction"] == "long").all()
        assert (result["signal_conditions_met"] == 2).all()

    def test_three_conditions_long(self):
        """3 of 3 bullish conditions → long signal with 3 conditions met."""
        df = _make_signal_df(
            trend_bull=True, structure_bull=True, momentum_bull=True
        )
        result = determine_signal_direction(df)
        assert (result["signal_direction"] == "long").all()
        assert (result["signal_conditions_met"] == 3).all()

    def test_two_conditions_short(self):
        """2 of 3 bearish conditions → short signal."""
        df = _make_signal_df(trend_bear=True, momentum_bear=True)
        result = determine_signal_direction(df)
        assert (result["signal_direction"] == "short").all()
        assert (result["signal_conditions_met"] == 2).all()

    def test_regime_blocks_long(self):
        """Regime doesn't allow long → no long signal despite conditions."""
        df = _make_signal_df(
            trend_bull=True, momentum_bull=True, regime_allow_long=False
        )
        result = determine_signal_direction(df)
        assert (result["signal_direction"].isna()).all()

    def test_regime_blocks_short(self):
        """Regime doesn't allow short → no short signal despite conditions."""
        df = _make_signal_df(
            trend_bear=True, momentum_bear=True, regime_allow_short=False
        )
        result = determine_signal_direction(df)
        assert (result["signal_direction"].isna()).all()

    def test_long_priority_over_short(self):
        """If both directions qualify, long takes priority."""
        df = _make_signal_df(
            trend_bull=True,
            momentum_bull=True,
            trend_bear=True,
            momentum_bear=True,
        )
        result = determine_signal_direction(df)
        assert (result["signal_direction"] == "long").all()

    def test_does_not_modify_original(self):
        """Original DataFrame is not modified."""
        df = _make_signal_df(trend_bull=True, momentum_bull=True)
        original_cols = list(df.columns)
        determine_signal_direction(df)
        assert list(df.columns) == original_cols


# ===========================================================================
# compute_signal_score
# ===========================================================================


class TestComputeSignalScore:
    """Test composite scoring logic."""

    def test_output_columns(self):
        """Adds signal_score and signal_tradeable."""
        df = _make_signal_df()
        df = determine_signal_direction(df)
        result = compute_signal_score(df)
        assert "signal_score" in result.columns
        assert "signal_tradeable" in result.columns

    def test_score_zero_no_signal(self):
        """No signal direction → score 0."""
        df = _make_signal_df()
        df = determine_signal_direction(df)
        result = compute_signal_score(df)
        assert (result["signal_score"] == 0).all()
        assert (result["signal_tradeable"] == False).all()  # noqa: E712

    def test_score_with_trend_only(self):
        """Long signal with only trend confirmed → score 1."""
        df = _make_signal_df(
            trend_bull=True, structure_bull=True  # 2 conditions to get signal
        )
        df = determine_signal_direction(df)
        result = compute_signal_score(df)

        # trend=1, structure_single=1 (assuming not multi-method)
        assert (result["signal_score"] == 2).all()

    def test_score_multi_method_structure(self):
        """Multi-method structure gets +2 instead of +1."""
        df = _make_signal_df(
            trend_bull=True,
            structure_bull=True,
            structure_multi_method=True,
        )
        df = determine_signal_direction(df)
        result = compute_signal_score(df)

        # trend=1, structure_multi=2 → score 3
        assert (result["signal_score"] == 3).all()

    def test_max_score_six(self):
        """All conditions + regime + volume → score 6."""
        df = _make_signal_df(
            trend_bull=True,
            structure_bull=True,
            momentum_bull=True,
            regime_strongly_aligned=True,
            structure_multi_method=True,
            volume_spike=True,
        )
        df = determine_signal_direction(df)
        result = compute_signal_score(df)

        # trend=1, structure_multi=2, momentum=1, regime=1, volume=1 = 6
        assert (result["signal_score"] == 6).all()

    def test_tradeable_threshold_default(self):
        """Score >= 3 is tradeable by default."""
        df = _make_signal_df(
            trend_bull=True,
            structure_bull=True,
            structure_multi_method=True,
        )
        df = determine_signal_direction(df)
        result = compute_signal_score(df)

        # Score = 3 → tradeable
        assert (result["signal_tradeable"] == True).all()  # noqa: E712

    def test_tradeable_threshold_custom(self):
        """Custom tradeable threshold works."""
        df = _make_signal_df(
            trend_bull=True, structure_bull=True  # score = 2
        )
        df = determine_signal_direction(df)

        result_low = compute_signal_score(df, tradeable_threshold=2)
        result_high = compute_signal_score(df, tradeable_threshold=4)

        assert (result_low["signal_tradeable"] == True).all()  # noqa: E712
        assert (result_high["signal_tradeable"] == False).all()  # noqa: E712

    def test_short_signal_scoring(self):
        """Short signals are scored correctly."""
        df = _make_signal_df(
            trend_bear=True,
            momentum_bear=True,
            regime_allow_long=False,
        )
        df = determine_signal_direction(df)
        result = compute_signal_score(df)

        # trend=1, momentum=1 → score 2
        assert (result["signal_score"] == 2).all()

    def test_volume_spike_absent(self):
        """If volume_spike column is missing, volume score is 0."""
        df = _make_signal_df(
            trend_bull=True, momentum_bull=True
        )
        df = df.drop(columns=["volume_spike"])
        df = determine_signal_direction(df)
        result = compute_signal_score(df)

        # trend=1, momentum=1 → score 2 (no volume bonus)
        assert (result["signal_score"] == 2).all()

    def test_custom_weights(self):
        """Custom scoring weights override defaults."""
        df = _make_signal_df(
            trend_bull=True, momentum_bull=True
        )
        df = determine_signal_direction(df)

        custom_weights = {
            "trend_confirmed": 3,
            "momentum_confirmed": 2,
            "structure_single_method": 1,
            "structure_multi_method": 2,
            "regime_strongly_aligned": 1,
            "volume_spike_on_entry": 1,
        }
        result = compute_signal_score(df, weights=custom_weights)

        # trend=3, momentum=2 → score 5
        assert (result["signal_score"] == 5).all()

    def test_does_not_modify_original(self):
        """Original DataFrame is not modified."""
        df = _make_signal_df(trend_bull=True, momentum_bull=True)
        df = determine_signal_direction(df)
        original_cols = list(df.columns)
        compute_signal_score(df)
        assert list(df.columns) == original_cols
