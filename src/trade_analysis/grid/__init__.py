"""Grid Runner — parameter optimization across signal engine configurations."""

from trade_analysis.grid.config import GridConfig, load_grid_config
from trade_analysis.grid.parameters import (
    ALL_KNOWN_PARAMS,
    apply_params_to_config,
    generate_parameter_grid,
)
from trade_analysis.grid.robustness import analyze_robustness, find_robust_zones
from trade_analysis.grid.runner import GridResult, GridRunner

__all__ = [
    # Config
    "GridConfig",
    "load_grid_config",
    # Parameters
    "ALL_KNOWN_PARAMS",
    "apply_params_to_config",
    "generate_parameter_grid",
    # Runner
    "GridResult",
    "GridRunner",
    # Robustness
    "analyze_robustness",
    "find_robust_zones",
]
