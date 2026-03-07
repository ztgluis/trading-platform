"""Backtest configuration: YAML loader and config dataclasses."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from trade_analysis.exceptions import ConfigError


# ---------------------------------------------------------------------------
# Config Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WalkForwardConfig:
    """Walk-forward validation parameters."""

    in_sample_years: int
    out_of_sample_years: int
    anchored: bool  # True = expanding IS window, False = rolling


@dataclass(frozen=True)
class BacktestConfig:
    """Complete backtest configuration loaded from YAML."""

    start_date: date
    end_date: date
    initial_capital: float
    max_open_positions: int
    walk_forward: WalkForwardConfig | None


# ---------------------------------------------------------------------------
# Config Loader
# ---------------------------------------------------------------------------


def load_backtest_config(
    config_path: Path | None = None,
) -> BacktestConfig:
    """Load backtest configuration from YAML.

    Args:
        config_path: Path to backtest.yaml. If None, uses default location.

    Returns:
        BacktestConfig dataclass.

    Raises:
        ConfigError: If config is missing, malformed, or missing required fields.
    """
    if config_path is None:
        config_path = Path("config/backtest.yaml")

    if not config_path.exists():
        raise ConfigError(f"Backtest config not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if not raw or "backtest" not in raw:
        raise ConfigError("Backtest config must have a top-level 'backtest' key")

    bt = raw["backtest"]

    # Parse date range
    date_range = bt.get("date_range")
    if not date_range:
        raise ConfigError("Backtest config must have a 'date_range' section")

    start_str = date_range.get("start")
    end_str = date_range.get("end")
    if not start_str or not end_str:
        raise ConfigError("date_range must have 'start' and 'end' fields")

    try:
        start_date = date.fromisoformat(start_str)
        end_date = date.fromisoformat(end_str)
    except (ValueError, TypeError) as e:
        raise ConfigError(f"Invalid date format in date_range: {e}") from e

    if start_date >= end_date:
        raise ConfigError(
            f"start_date ({start_date}) must be before end_date ({end_date})"
        )

    # Parse walk-forward (optional)
    wf_raw = bt.get("walk_forward")
    walk_forward: WalkForwardConfig | None = None
    if wf_raw:
        is_years = wf_raw.get("in_sample_years")
        oos_years = wf_raw.get("out_of_sample_years")
        if is_years is None or oos_years is None:
            raise ConfigError(
                "walk_forward must have 'in_sample_years' and 'out_of_sample_years'"
            )
        walk_forward = WalkForwardConfig(
            in_sample_years=int(is_years),
            out_of_sample_years=int(oos_years),
            anchored=bool(wf_raw.get("anchored", True)),
        )

    return BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=float(bt.get("initial_capital", 100000.0)),
        max_open_positions=int(bt.get("max_open_positions", 1)),
        walk_forward=walk_forward,
    )
