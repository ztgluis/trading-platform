"""Grid runner: sweep parameter combinations through the full pipeline.

Runs generate_signals() → Backtester.run() → compute_backtest_stats()
for each parameter combination and collects results into a GridResult.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from trade_analysis.backtester.config import BacktestConfig
from trade_analysis.backtester.engine import Backtester
from trade_analysis.backtester.stats import compute_backtest_stats
from trade_analysis.grid.config import GridConfig
from trade_analysis.grid.parameters import (
    apply_params_to_config,
    generate_parameter_grid,
)
from trade_analysis.signals.engine import SignalEngineConfig, generate_signals


# ---------------------------------------------------------------------------
# GridResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GridResult:
    """Results of a grid parameter sweep.

    Contains one row per parameter combination with stats and param values.
    """

    rows: list[dict] = field(default_factory=list)
    grid_config: GridConfig | None = None

    @property
    def total_combos(self) -> int:
        """Total number of parameter combinations tested."""
        return len(self.rows)

    def to_dataframe(self) -> pd.DataFrame:
        """Full results matrix as a DataFrame."""
        if not self.rows:
            return pd.DataFrame()
        return pd.DataFrame(self.rows)

    def sufficient_only(self) -> pd.DataFrame:
        """Filter to results with >= min_trades."""
        df = self.to_dataframe()
        if df.empty:
            return df
        min_trades = self.grid_config.min_trades if self.grid_config else 30
        return df[df["total_trades"] >= min_trades].reset_index(drop=True)

    def rank(self, by: str | None = None) -> pd.DataFrame:
        """Rank results by a metric (descending), filtered to sufficient trades.

        Args:
            by: Metric to sort by. If None, uses grid_config.rank_by.

        Returns:
            Sorted DataFrame with rank column.
        """
        metric = by or (self.grid_config.rank_by if self.grid_config else "total_r")
        df = self.sufficient_only()
        if df.empty:
            return df
        ranked = df.sort_values(metric, ascending=False).reset_index(drop=True)
        ranked.insert(0, "rank", range(1, len(ranked) + 1))
        return ranked

    def top_n(self, n: int = 10, by: str | None = None) -> pd.DataFrame:
        """Top N results ranked by a metric."""
        return self.rank(by=by).head(n)

    def format_report(self, top_n: int = 10) -> str:
        """Human-readable summary of grid results."""
        lines = [
            "=" * 60,
            "GRID SWEEP RESULTS",
            "=" * 60,
            f"Total combinations:  {self.total_combos}",
        ]

        df_suff = self.sufficient_only()
        lines.append(f"Sufficient trades:   {len(df_suff)} / {self.total_combos}")

        if df_suff.empty:
            lines.append("No combinations met the minimum trade count.")
            lines.append("=" * 60)
            return "\n".join(lines)

        metric = self.grid_config.rank_by if self.grid_config else "total_r"
        lines.append(f"Ranked by:           {metric}")
        lines.append("")

        # Get parameter column names
        param_names = list(self.grid_config.parameters.keys()) if self.grid_config else []

        # Top results
        ranked = self.rank()
        top = ranked.head(top_n)
        lines.append(f"--- TOP {min(top_n, len(top))} ---")

        for _, row in top.iterrows():
            params_str = ", ".join(
                f"{p}={row[p]}" for p in param_names if p in row
            )
            lines.append(
                f"  #{int(row['rank']):2d}  {params_str}  "
                f"n={int(row['total_trades'])}  "
                f"WR={row['win_rate']:.0%}  "
                f"avgR={row['avg_r']:+.2f}  "
                f"totalR={row['total_r']:+.1f}  "
                f"PF={row['profit_factor']:.2f}"
            )

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# GridRunner
# ---------------------------------------------------------------------------


class GridRunner:
    """Runs the full pipeline across all parameter combinations.

    Usage:
        runner = GridRunner(grid_config, bt_config, signal_config)
        result = runner.run(ohlcv_df)
    """

    def __init__(
        self,
        grid_config: GridConfig,
        backtest_config: BacktestConfig,
        signal_config: SignalEngineConfig,
    ) -> None:
        self._grid_config = grid_config
        self._bt_config = backtest_config
        self._signal_config = signal_config

    def run(self, ohlcv: pd.DataFrame) -> GridResult:
        """Run pipeline for each parameter combination.

        Args:
            ohlcv: Raw OHLCV DataFrame (before signal enrichment).

        Returns:
            GridResult with stats for each combination.
        """
        combos = generate_parameter_grid(self._grid_config.parameters)
        rows: list[dict] = []

        for combo in combos:
            row = self._run_single(ohlcv, combo)
            rows.append(row)

        return GridResult(rows=rows, grid_config=self._grid_config)

    def _run_single(self, ohlcv: pd.DataFrame, params: dict) -> dict:
        """Run pipeline for a single parameter combination.

        Returns a dict with param values + stats.
        """
        # 1. Modify signal config
        modified_config = apply_params_to_config(
            self._signal_config,
            params,
            self._grid_config.asset_class,
        )

        # 2. Generate signals with modified config
        enriched = generate_signals(
            ohlcv,
            asset_class=self._grid_config.asset_class,
            config=modified_config,
        )

        # 3. Run backtest
        backtester = Backtester(self._bt_config, modified_config)
        result = backtester.run(
            enriched,
            self._grid_config.symbol,
            self._grid_config.asset_class,
            self._grid_config.timeframe,
        )

        # 4. Compute stats
        stats = compute_backtest_stats(result)

        # 5. Combine params + stats
        row = dict(params)  # copy param values
        row.update(stats)

        return row
