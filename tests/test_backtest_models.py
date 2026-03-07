"""Tests for M4 backtest data models (models.py)."""

from datetime import date

import pandas as pd
import pytest

from trade_analysis.backtester.config import BacktestConfig
from trade_analysis.backtester.models import (
    BacktestResult,
    Position,
    Trade,
    WalkForwardSplit,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trade(
    pnl_r: float = 1.5,
    direction: str = "long",
    exit_reason: str = "target",
    entry_score: int = 4,
    entry_regime: str = "bull",
) -> Trade:
    return Trade(
        symbol="AAPL",
        asset_class="stock",
        timeframe="Daily",
        bucket="A",
        direction=direction,
        entry_timestamp=pd.Timestamp("2021-06-01", tz="UTC"),
        entry_price=150.0,
        entry_signal_score=entry_score,
        entry_regime=entry_regime,
        exit_timestamp=pd.Timestamp("2021-06-15", tz="UTC"),
        exit_price=157.5,
        exit_reason=exit_reason,
        pnl_r=pnl_r,
        pnl_dollar=7.5 if direction == "long" else -7.5,
        duration_bars=10,
        duration_calendar_days=14,
    )


def _make_config() -> BacktestConfig:
    return BacktestConfig(
        start_date=date(2020, 1, 1),
        end_date=date(2024, 12, 31),
        initial_capital=100000.0,
        max_open_positions=1,
        walk_forward=None,
    )


# ===========================================================================
# Position
# ===========================================================================


class TestPosition:
    """Test Position (mutable open trade)."""

    def test_creation(self):
        pos = Position(
            symbol="AAPL",
            asset_class="stock",
            timeframe="Daily",
            bucket="A",
            direction="long",
            entry_bar_index=100,
            entry_timestamp=pd.Timestamp("2021-06-01", tz="UTC"),
            entry_price=150.0,
            entry_signal_score=4,
            entry_regime="bull",
            stop_loss=145.0,
            target=160.0,
            trail_breakeven_price=155.0,
            max_hold_bars=20,
        )
        assert pos.symbol == "AAPL"
        assert pos.direction == "long"
        assert pos.entry_price == 150.0

    def test_current_stop_initializes_to_stop_loss(self):
        pos = Position(
            symbol="AAPL",
            asset_class="stock",
            timeframe="Daily",
            bucket="A",
            direction="long",
            entry_bar_index=100,
            entry_timestamp=pd.Timestamp("2021-06-01", tz="UTC"),
            entry_price=150.0,
            entry_signal_score=4,
            entry_regime="bull",
            stop_loss=145.0,
            target=160.0,
            trail_breakeven_price=155.0,
            max_hold_bars=20,
        )
        assert pos.current_stop == 145.0

    def test_mutable_stop(self):
        pos = Position(
            symbol="AAPL",
            asset_class="stock",
            timeframe="Daily",
            bucket="A",
            direction="long",
            entry_bar_index=100,
            entry_timestamp=pd.Timestamp("2021-06-01", tz="UTC"),
            entry_price=150.0,
            entry_signal_score=4,
            entry_regime="bull",
            stop_loss=145.0,
            target=160.0,
            trail_breakeven_price=155.0,
            max_hold_bars=20,
        )
        pos.current_stop = 150.0
        pos.stop_trailed_to_breakeven = True
        assert pos.current_stop == 150.0
        assert pos.stop_trailed_to_breakeven is True

    def test_max_hold_bars_none(self):
        pos = Position(
            symbol="^GSPC",
            asset_class="index",
            timeframe="Weekly",
            bucket="B",
            direction="long",
            entry_bar_index=50,
            entry_timestamp=pd.Timestamp("2021-01-04", tz="UTC"),
            entry_price=3700.0,
            entry_signal_score=5,
            entry_regime="bull",
            stop_loss=3600.0,
            target=4000.0,
            trail_breakeven_price=3850.0,
            max_hold_bars=None,
        )
        assert pos.max_hold_bars is None


# ===========================================================================
# Trade
# ===========================================================================


class TestTrade:
    """Test Trade (frozen closed trade)."""

    def test_is_winner_positive_r(self):
        t = _make_trade(pnl_r=1.5)
        assert t.is_winner is True

    def test_is_winner_negative_r(self):
        t = _make_trade(pnl_r=-1.0)
        assert t.is_winner is False

    def test_is_winner_zero_r(self):
        t = _make_trade(pnl_r=0.0)
        assert t.is_winner is False

    def test_frozen(self):
        t = _make_trade()
        with pytest.raises(AttributeError):
            t.pnl_r = 99.0  # type: ignore[misc]

    def test_exit_reasons(self):
        for reason in ("stop", "trail_stop", "target", "max_hold", "end_of_data"):
            t = _make_trade(exit_reason=reason)
            assert t.exit_reason == reason


# ===========================================================================
# BacktestResult
# ===========================================================================


class TestBacktestResult:
    """Test BacktestResult and to_dataframe()."""

    def test_to_dataframe_with_trades(self):
        trades = [
            _make_trade(pnl_r=2.0, exit_reason="target"),
            _make_trade(pnl_r=-1.0, exit_reason="stop"),
        ]
        result = BacktestResult(
            trades=trades,
            config=_make_config(),
            signal_config_hash="abc123",
            symbol="AAPL",
            asset_class="stock",
            timeframe="Daily",
            start_date=date(2020, 1, 1),
            end_date=date(2024, 12, 31),
        )
        df = result.to_dataframe()
        assert len(df) == 2
        assert list(df["pnl_r"]) == [2.0, -1.0]
        assert list(df["exit_reason"]) == ["target", "stop"]

    def test_to_dataframe_columns(self):
        result = BacktestResult(
            trades=[_make_trade()],
            config=_make_config(),
            signal_config_hash="abc123",
            symbol="AAPL",
            asset_class="stock",
            timeframe="Daily",
            start_date=date(2020, 1, 1),
            end_date=date(2024, 12, 31),
        )
        df = result.to_dataframe()
        expected = {
            "symbol", "asset_class", "timeframe", "bucket", "direction",
            "entry_timestamp", "entry_price", "entry_signal_score", "entry_regime",
            "exit_timestamp", "exit_price", "exit_reason",
            "pnl_r", "pnl_dollar", "duration_bars", "duration_calendar_days",
            "is_winner",
        }
        assert set(df.columns) == expected

    def test_to_dataframe_empty(self):
        result = BacktestResult(
            trades=[],
            config=_make_config(),
            signal_config_hash="abc123",
            symbol="AAPL",
            asset_class="stock",
            timeframe="Daily",
            start_date=date(2020, 1, 1),
            end_date=date(2024, 12, 31),
        )
        df = result.to_dataframe()
        assert len(df) == 0
        assert "pnl_r" in df.columns

    def test_is_winner_in_dataframe(self):
        trades = [
            _make_trade(pnl_r=2.0),
            _make_trade(pnl_r=-1.0),
        ]
        result = BacktestResult(
            trades=trades,
            config=_make_config(),
            signal_config_hash="abc123",
            symbol="AAPL",
            asset_class="stock",
            timeframe="Daily",
            start_date=date(2020, 1, 1),
            end_date=date(2024, 12, 31),
        )
        df = result.to_dataframe()
        assert list(df["is_winner"]) == [True, False]

    def test_frozen(self):
        result = BacktestResult(
            trades=[],
            config=_make_config(),
            signal_config_hash="abc123",
            symbol="AAPL",
            asset_class="stock",
            timeframe="Daily",
            start_date=date(2020, 1, 1),
            end_date=date(2024, 12, 31),
        )
        with pytest.raises(AttributeError):
            result.symbol = "MSFT"  # type: ignore[misc]


# ===========================================================================
# WalkForwardSplit
# ===========================================================================


class TestWalkForwardSplit:
    """Test WalkForwardSplit dataclass."""

    def test_creation(self):
        split = WalkForwardSplit(
            fold=1,
            is_start=date(2020, 1, 1),
            is_end=date(2022, 12, 31),
            oos_start=date(2023, 1, 1),
            oos_end=date(2023, 12, 31),
        )
        assert split.fold == 1
        assert split.is_start == date(2020, 1, 1)
        assert split.oos_end == date(2023, 12, 31)

    def test_frozen(self):
        split = WalkForwardSplit(
            fold=1,
            is_start=date(2020, 1, 1),
            is_end=date(2022, 12, 31),
            oos_start=date(2023, 1, 1),
            oos_end=date(2023, 12, 31),
        )
        with pytest.raises(AttributeError):
            split.fold = 2  # type: ignore[misc]
