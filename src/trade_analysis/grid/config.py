"""Grid configuration: parameter ranges, target asset, and ranking settings.

Loads grid sweep parameters from config/grid.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from trade_analysis.exceptions import ConfigError


_VALID_RANK_BY = {"total_r", "avg_r", "profit_factor", "win_rate"}


@dataclass(frozen=True)
class GridConfig:
    """Configuration for a grid parameter sweep."""

    # Target
    symbol: str
    asset_class: str
    timeframe: str

    # Parameters to sweep: param_name → list of values
    parameters: dict[str, list] = field(default_factory=dict)

    # Filtering and ranking
    min_trades: int = 30
    rank_by: str = "total_r"


def load_grid_config(config_path: Path | None = None) -> GridConfig:
    """Load grid configuration from YAML.

    Args:
        config_path: Path to grid.yaml. If None, uses default location.

    Returns:
        GridConfig dataclass.

    Raises:
        ConfigError: If file is missing, malformed, or has invalid values.
    """
    path = config_path or Path("config/grid.yaml")
    if not path.exists():
        raise ConfigError(f"Grid config not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not raw or "grid" not in raw:
        raise ConfigError("Grid config must have a top-level 'grid' key")

    grid = raw["grid"]

    # Target
    target = grid.get("target", {})
    if not target:
        raise ConfigError("Grid config must have a 'target' section")

    symbol = target.get("symbol")
    if not symbol:
        raise ConfigError("Grid target must specify 'symbol'")

    asset_class = target.get("asset_class")
    if not asset_class:
        raise ConfigError("Grid target must specify 'asset_class'")

    timeframe = target.get("timeframe", "Daily")

    # Parameters
    parameters = grid.get("parameters", {})
    if not parameters:
        raise ConfigError("Grid config must have at least one parameter to sweep")

    # Validate parameter values are lists
    for param_name, values in parameters.items():
        if not isinstance(values, list):
            raise ConfigError(
                f"Parameter '{param_name}' must be a list of values, "
                f"got {type(values).__name__}"
            )
        if len(values) == 0:
            raise ConfigError(f"Parameter '{param_name}' must have at least one value")

    # Ranking
    rank_by = grid.get("rank_by", "total_r")
    if rank_by not in _VALID_RANK_BY:
        raise ConfigError(
            f"Invalid rank_by '{rank_by}', must be one of: {sorted(_VALID_RANK_BY)}"
        )

    # Min trades
    min_trades = int(grid.get("min_trades", 30))
    if min_trades < 0:
        raise ConfigError(f"min_trades must be non-negative, got {min_trades}")

    return GridConfig(
        symbol=symbol,
        asset_class=asset_class,
        timeframe=timeframe,
        parameters=parameters,
        min_trades=min_trades,
        rank_by=rank_by,
    )
