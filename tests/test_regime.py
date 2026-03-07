"""Tests for M3 regime detection (regime.py)."""

import numpy as np
import pandas as pd
import pytest

from trade_analysis.signals.regime import detect_regime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(closes: list[float], base_volume: float = 1e6) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a list of close prices."""
    n = len(closes)
    timestamps = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    closes_arr = np.array(closes, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": closes_arr * 0.999,
            "high": closes_arr * 1.005,
            "low": closes_arr * 0.995,
            "close": closes_arr,
            "volume": np.full(n, base_volume),
        }
    )


# ---------------------------------------------------------------------------
# Output columns
# ---------------------------------------------------------------------------


class TestRegimeOutputColumns:
    """Verify detect_regime adds the expected columns."""

    def test_columns_present(self, sample_200bar_ohlcv):
        result = detect_regime(sample_200bar_ohlcv)
        expected = {
            "regime_ma",
            "regime",
            "regime_allow_long",
            "regime_allow_short",
            "regime_strongly_aligned",
            "regime_distance_pct",
        }
        assert expected.issubset(set(result.columns))

    def test_does_not_modify_original(self, sample_200bar_ohlcv):
        original_cols = list(sample_200bar_ohlcv.columns)
        detect_regime(sample_200bar_ohlcv)
        assert list(sample_200bar_ohlcv.columns) == original_cols

    def test_preserves_original_columns(self, sample_200bar_ohlcv):
        result = detect_regime(sample_200bar_ohlcv)
        for col in sample_200bar_ohlcv.columns:
            assert col in result.columns


# ---------------------------------------------------------------------------
# Bull regime
# ---------------------------------------------------------------------------


class TestBullRegime:
    """Test bull regime detection when price is above MA."""

    def test_clear_bull_regime(self):
        """Price well above SMA(5) for many bars → bull regime."""
        # Start low (to build SMA), then jump up and stay
        closes = [100.0] * 5 + [120.0] * 10
        df = _make_ohlcv(closes)
        result = detect_regime(df, ma_period=5, transition_closes=3)

        # Last bars should be bull (price far above SMA)
        last_regime = result["regime"].iloc[-1]
        assert last_regime == "bull"

    def test_bull_allows_long(self):
        """Bull regime allows long signals."""
        closes = [100.0] * 5 + [120.0] * 10
        df = _make_ohlcv(closes)
        result = detect_regime(df, ma_period=5, transition_closes=3)

        bull_rows = result[result["regime"] == "bull"]
        assert (bull_rows["regime_allow_long"] == True).all()  # noqa: E712

    def test_bull_blocks_short(self):
        """Bull regime blocks short signals."""
        closes = [100.0] * 5 + [120.0] * 10
        df = _make_ohlcv(closes)
        result = detect_regime(df, ma_period=5, transition_closes=3)

        bull_rows = result[result["regime"] == "bull"]
        assert (bull_rows["regime_allow_short"] == False).all()  # noqa: E712


# ---------------------------------------------------------------------------
# Bear regime
# ---------------------------------------------------------------------------


class TestBearRegime:
    """Test bear regime detection when price is below MA."""

    def test_clear_bear_regime(self):
        """Price well below SMA(5) for many bars → bear regime."""
        closes = [100.0] * 5 + [80.0] * 10
        df = _make_ohlcv(closes)
        result = detect_regime(df, ma_period=5, transition_closes=3)

        last_regime = result["regime"].iloc[-1]
        assert last_regime == "bear"

    def test_bear_allows_short(self):
        """Bear regime allows short signals."""
        closes = [100.0] * 5 + [80.0] * 10
        df = _make_ohlcv(closes)
        result = detect_regime(df, ma_period=5, transition_closes=3)

        bear_rows = result[result["regime"] == "bear"]
        assert (bear_rows["regime_allow_short"] == True).all()  # noqa: E712

    def test_bear_blocks_long(self):
        """Bear regime blocks long signals."""
        closes = [100.0] * 5 + [80.0] * 10
        df = _make_ohlcv(closes)
        result = detect_regime(df, ma_period=5, transition_closes=3)

        bear_rows = result[result["regime"] == "bear"]
        assert (bear_rows["regime_allow_long"] == False).all()  # noqa: E712


# ---------------------------------------------------------------------------
# Transition regime
# ---------------------------------------------------------------------------


class TestTransitionRegime:
    """Test transition regime for ambiguous price action."""

    def test_transition_during_warmup(self):
        """Bars during MA warmup should be transition."""
        closes = [100.0] * 10
        df = _make_ohlcv(closes)
        result = detect_regime(df, ma_period=10, transition_closes=3)

        # First 9 bars will have NaN MA → transition
        for i in range(9):
            assert result["regime"].iloc[i] == "transition"

    def test_transition_allows_both_directions(self):
        """Transition regime allows both long and short."""
        closes = [100.0] * 10
        df = _make_ohlcv(closes)
        result = detect_regime(df, ma_period=10, transition_closes=3)

        transition_rows = result[result["regime"] == "transition"]
        if len(transition_rows) > 0:
            assert (transition_rows["regime_allow_long"] == True).all()  # noqa: E712
            assert (transition_rows["regime_allow_short"] == True).all()  # noqa: E712

    def test_oscillating_price_stays_transition(self):
        """Price oscillating around MA doesn't confirm either regime."""
        # Alternate above and below (never 3 consecutive on one side)
        closes = [100.0] * 5  # warmup
        for i in range(20):
            closes.append(102.0 if i % 2 == 0 else 98.0)
        df = _make_ohlcv(closes)
        result = detect_regime(df, ma_period=5, transition_closes=3)

        # Check that not all late bars are bull or bear
        late = result.iloc[-10:]
        # Should contain transition (since price flips every bar, can't get 3 consecutive)
        assert "transition" in late["regime"].values


# ---------------------------------------------------------------------------
# Strongly aligned
# ---------------------------------------------------------------------------


class TestStronglyAligned:
    """Test the strongly aligned flag (price far from regime MA)."""

    def test_strongly_aligned_bull(self):
        """Price >5% above MA in bull regime → strongly aligned."""
        # Use a large jump so SMA(5) lags significantly behind price
        closes = [100.0] * 5 + [130.0] * 10
        df = _make_ohlcv(closes)
        result = detect_regime(
            df, ma_period=5, transition_closes=3, strong_alignment_pct=5.0
        )

        # First bars after jump: SMA still includes 100s, price = 130
        # e.g. bar 7: SMA = (100,100,130,130,130)/5 = 118, dist = (130-118)/118*100 ≈ 10%
        bull_aligned = result[
            (result["regime"] == "bull") & (result["regime_strongly_aligned"])
        ]
        assert len(bull_aligned) > 0

    def test_strongly_aligned_bear(self):
        """Price >5% below MA in bear regime → strongly aligned."""
        closes = [100.0] * 5 + [70.0] * 10
        df = _make_ohlcv(closes)
        result = detect_regime(
            df, ma_period=5, transition_closes=3, strong_alignment_pct=5.0
        )

        bear_aligned = result[
            (result["regime"] == "bear") & (result["regime_strongly_aligned"])
        ]
        assert len(bear_aligned) > 0

    def test_not_strongly_aligned_when_close_to_ma(self):
        """Price near MA → not strongly aligned."""
        closes = [100.0] * 20
        df = _make_ohlcv(closes)
        result = detect_regime(
            df, ma_period=5, transition_closes=3, strong_alignment_pct=5.0
        )

        # All prices = SMA → distance ~0%
        assert (result["regime_strongly_aligned"] == False).all()  # noqa: E712


# ---------------------------------------------------------------------------
# Distance percentage
# ---------------------------------------------------------------------------


class TestDistancePct:
    """Test regime_distance_pct values."""

    def test_distance_pct_positive_above_ma(self):
        """Price above MA → positive distance %."""
        closes = [100.0] * 5 + [110.0] * 5
        df = _make_ohlcv(closes)
        result = detect_regime(df, ma_period=5, transition_closes=3)

        # Last bar: close=110, SMA(5) should be close to 110 (all 110s)
        # But the first bar after jump: SMA = (100+100+100+100+110)/5 = 102
        # Distance = (110 - 102) / 102 * 100 ≈ 7.8%
        valid = result["regime_distance_pct"].dropna()
        assert (valid.iloc[-3:] >= 0).all()

    def test_distance_pct_negative_below_ma(self):
        """Price below MA → negative distance %."""
        closes = [100.0] * 5 + [90.0] * 5
        df = _make_ohlcv(closes)
        result = detect_regime(df, ma_period=5, transition_closes=3)

        valid = result["regime_distance_pct"].dropna()
        assert (valid.iloc[-3:] <= 0).all()

    def test_distance_pct_near_zero_when_flat(self):
        """Flat price → distance ≈ 0."""
        closes = [100.0] * 20
        df = _make_ohlcv(closes)
        result = detect_regime(df, ma_period=5, transition_closes=3)

        valid = result["regime_distance_pct"].dropna()
        assert (valid.abs() < 1.0).all()


# ---------------------------------------------------------------------------
# MA type
# ---------------------------------------------------------------------------


class TestMAType:
    """Test different MA types for regime detection."""

    def test_ema_regime(self):
        """EMA can be used for regime detection."""
        closes = [100.0] * 5 + [120.0] * 10
        df = _make_ohlcv(closes)
        result = detect_regime(df, ma_type="ema", ma_period=5, transition_closes=3)

        assert result["regime"].iloc[-1] == "bull"
        assert result["regime_ma"].notna().any()

    def test_sma_is_default(self, sample_200bar_ohlcv):
        """SMA is the default MA type."""
        result = detect_regime(sample_200bar_ohlcv)
        # Just verify it runs without error
        assert "regime_ma" in result.columns


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestRegimeEdgeCases:
    """Edge cases and NaN handling."""

    def test_nan_during_ma_warmup(self):
        """regime_ma should be NaN during warmup period."""
        closes = [100.0] * 20
        df = _make_ohlcv(closes)
        result = detect_regime(df, ma_period=10)

        nan_count = result["regime_ma"].isna().sum()
        assert nan_count >= 1  # At least some NaN during warmup

    def test_regime_values_are_valid(self, sample_200bar_ohlcv):
        """All regime values should be one of bull/bear/transition."""
        result = detect_regime(sample_200bar_ohlcv, ma_period=50)
        valid_regimes = {"bull", "bear", "transition"}
        assert set(result["regime"].unique()).issubset(valid_regimes)

    def test_transition_closes_parameter(self):
        """Higher transition_closes requires more bars to confirm regime."""
        closes = [100.0] * 5 + [110.0] * 3
        df = _make_ohlcv(closes)

        # With transition_closes=3, should just barely confirm bull
        r3 = detect_regime(df, ma_period=5, transition_closes=3)

        # With transition_closes=10, should stay transition
        r10 = detect_regime(df, ma_period=5, transition_closes=10)

        # r10 should have fewer (or equal) bull bars than r3
        bull_count_3 = (r3["regime"] == "bull").sum()
        bull_count_10 = (r10["regime"] == "bull").sum()
        assert bull_count_10 <= bull_count_3
