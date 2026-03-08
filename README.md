# Trade Analysis Platform

Personal swing trading platform for systematic strategy validation and signal generation across stocks, ETFs, indices, crypto, and metals.

## Architecture

The platform is organized into milestones:

| Milestone | Status | Description |
|-----------|--------|-------------|
| M1: Data Layer | Complete | Fetch and normalize OHLCV data across asset classes |
| M2: Indicator Library | Complete | MA/ATR/VIDYA, RSI/MACD, oscillators, trend signals, structure, levels |
| M3: Signal Engine | Complete | 2-of-3 condition entry, regime filter, scoring (0-6), exit levels |
| M4: Backtester | Complete | Historical replay with walk-forward splits |
| M5: Grid Runner | Complete | Parameter optimization across combinations |
| M6: Results Analyzer | Complete | Hypothesis testing (H1-H5) + Supabase persistence |
| M7: Dashboard | Planned | Streamlit results explorer |
| M8: Live Runner (Paper) | Planned | TradingView webhook integration |
| M9: Live Runner (Real) | Planned | Schwab API execution |

## Setup

```bash
# Clone the repo
git clone git@github.com:ztgluis/trading-platform.git
cd trading-platform

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (including dev tools)
pip install -e ".[dev]"

# Set up environment variables
cp .env.example .env
# Edit .env with your actual API keys
```

## Project Structure

```
trade-analysis/
├── config/          # YAML configuration files
├── db/migrations/   # Supabase SQL migration scripts
├── docs/            # PRD and documentation
├── src/trade_analysis/
│   ├── config/      # Config loading + env var resolution
│   ├── models/      # OHLCV schema and validation
│   ├── providers/   # Data providers (yfinance, ccxt, schwab)
│   ├── cache/       # Parquet-based local cache
│   ├── transforms/  # Normalize, timeframe aggregation, inverse
│   ├── indicators/  # Technical indicators (trend, momentum, structure, volume, levels)
│   ├── signals/     # Signal engine (regime, conditions, scoring, exits)
│   ├── backtester/  # Historical replay engine, stats, walk-forward
│   ├── grid/        # Parameter grid sweep, robustness analysis
│   ├── analyzer/    # Hypothesis evaluators (H1-H5), Supabase persistence
│   └── data_manager.py  # Main orchestrator
├── tests/           # pytest test suite
└── scripts/         # CLI utilities
```

## Usage

### Fetch data programmatically

```python
from trade_analysis.data_manager import DataManager
from trade_analysis.models.ohlcv import Timeframe

dm = DataManager()

# Fetch daily AAPL data
df = dm.get_ohlcv("AAPL", Timeframe.DAILY)

# Fetch with inverse (for short-side signal detection)
df_inv = dm.get_ohlcv("AAPL", Timeframe.DAILY, inverse=True)

# Fetch crypto
df_btc = dm.get_ohlcv("BTC/USDT", Timeframe.H4)

# Fetch multiple symbols
results = dm.get_multiple(["AAPL", "MSFT", "NVDA"], Timeframe.DAILY)
```

### Apply indicators

```python
from trade_analysis.indicators import (
    add_sma, add_ema, add_hma, add_zlema, add_vidya, add_atr,
    add_rsi, add_macd,
    detect_swing_highs, detect_swing_lows,
    detect_volume_spike, detect_pivot_levels,
    detect_round_numbers, find_nearest_level,
    add_two_pole_oscillator, add_momentum_bias_index,
    add_zero_lag_trend_signals, add_volumatic_vidya,
)

# Trend indicators (SMA, EMA, HMA, ZLEMA, VIDYA, ATR)
df = add_sma(df, period=50)
df = add_ema(df, period=21)
df = add_zlema(df, period=70)
df = add_vidya(df, length=10, momentum_period=20)
df = add_atr(df, period=14)

# Momentum indicators
df = add_rsi(df, period=14)
df = add_macd(df)

# Oscillators (from TradingView community indicators)
df = add_two_pole_oscillator(df, filter_length=15)
df = add_momentum_bias_index(df, momentum_length=10)

# Composite trend signal systems
df = add_zero_lag_trend_signals(df, length=70, multiplier=1.2)
df = add_volumatic_vidya(df, vidya_length=10, band_distance=2.0)

# Structure + volume + levels
df = detect_swing_highs(df, lookback=5)
df = detect_volume_spike(df, period=20, threshold=1.5)
levels = detect_pivot_levels(df, lookback=5)
```

### Generate trading signals

```python
from trade_analysis.signals import generate_signals, load_signal_config

# Load config (auto-loads from config/signals.yaml)
config = load_signal_config()

# Run full signal pipeline on OHLCV data
result = generate_signals(df, asset_class="stock", config=config)

# Filter for tradeable signals (score >= 3)
tradeable = result[result["signal_tradeable"]]
print(tradeable[["close", "signal_direction", "signal_score",
                  "exit_stop", "exit_target", "exit_rr_ratio"]])

# Use individual components
from trade_analysis.signals import detect_regime, evaluate_trend_condition

regime_df = detect_regime(df, ma_type="sma", ma_period=200)
print(regime_df[["close", "regime", "regime_distance_pct"]])
```

### Run backtests

```python
from trade_analysis.backtester import (
    Backtester, BacktestConfig, load_backtest_config,
    compute_backtest_stats, format_stats_report,
    run_walk_forward,
)
from trade_analysis.signals import generate_signals, load_signal_config

# Load configs
bt_config = load_backtest_config()  # from config/backtest.yaml
signal_config = load_signal_config()

# Generate signals on OHLCV data
enriched = generate_signals(df, asset_class="stock", config=signal_config)

# Run backtest
backtester = Backtester(bt_config, signal_config)
result = backtester.run(enriched, "AAPL", "stock", "Daily")

# Summary statistics
stats = compute_backtest_stats(result)
print(format_stats_report(stats))

# Trade log as DataFrame
trade_log = result.to_dataframe()
print(trade_log[["entry_timestamp", "exit_reason", "pnl_r", "duration_bars"]])

# Walk-forward validation
wf_result = run_walk_forward(
    enriched, "AAPL", "stock", "Daily", bt_config, signal_config
)
for i, split in enumerate(wf_result.splits):
    oos_stats = compute_backtest_stats(wf_result.out_of_sample_results[i])
    print(f"Fold {i} OOS: {oos_stats['total_trades']} trades, "
          f"WR={oos_stats['win_rate']:.0%}, avgR={oos_stats['avg_r']:+.2f}")
```

### Run parameter grid sweeps

```python
from trade_analysis.grid import (
    GridRunner, load_grid_config,
    analyze_robustness, find_robust_zones,
)
from trade_analysis.backtester import load_backtest_config
from trade_analysis.signals import load_signal_config

# Load configs
grid_config = load_grid_config()    # from config/grid.yaml
bt_config = load_backtest_config()
signal_config = load_signal_config()

# Run grid sweep (RSI period x MA period)
runner = GridRunner(grid_config, bt_config, signal_config)
result = runner.run(df)  # df = raw OHLCV DataFrame

# View ranked results
print(result.format_report())
top_5 = result.top_n(5)

# Robustness analysis — find stable parameter zones
zones = find_robust_zones(result.sufficient_only(), metric="total_r")
for param, param_zones in zones.items():
    for zone in param_zones:
        print(f"{param}: values {zone['values']} → avg {zone['avg_metric']:.2f}")
```

### Analyze grid results (hypothesis testing)

```python
from trade_analysis.analyzer import (
    evaluate_all, format_hypothesis_report,
    SupabaseClient, persist_grid_run, persist_hypothesis_results,
)

# Evaluate hypotheses H1-H5 on grid results
grid_df = result.to_dataframe()  # from GridResult
hypotheses = evaluate_all(grid_df)

# Print formatted report
print(format_hypothesis_report(hypotheses))
# [+] H1 SUPPORTED: Trend filter improves avg R by +0.15. Best period: 30.
# [-] H2 REFUTED: MA type has negligible impact (spread=0.012).
# [+] H3 SUPPORTED: Best period: 30 (avg R=+0.25, robust).
# [~] H4 NOT_TESTABLE: Cannot test — crossover not implemented.
# [+] H5 SUPPORTED: Higher rsi_bull_threshold improves win rate by +8.5%.

# Optional: persist to Supabase (skips gracefully if not configured)
sb = SupabaseClient()
run_id = persist_grid_run(sb, grid_config, result)
persist_hypothesis_results(sb, hypotheses, grid_run_id=run_id)
```

### CLI smoke test

```bash
python -m scripts.fetch_sample AAPL Daily
python -m scripts.fetch_sample BTC/USDT 4H
python -m scripts.fetch_sample AAPL Daily --inverse
```

## Running Tests

```bash
pytest tests/ -v         # 524 tests
pytest tests/ -v --cov   # With coverage
```

## Tech Stack

- **Language**: Python 3.12+
- **Data**: pandas, pyarrow (Parquet cache)
- **Stocks/ETFs/Metals**: yfinance
- **Crypto**: CCXT
- **Database**: Supabase (PostgreSQL)
- **Indicators**: pandas-ta + custom (SMA, EMA, HMA, ZLEMA, VIDYA, ATR, RSI, MACD, oscillators, trend signals)
- **Signals**: Regime detection, 2-of-3 condition gate, composite scoring, exit levels
- **Backtester**: Bar-by-bar replay, stop/target/trail exits, walk-forward validation
- **Grid Runner**: Parameter sweeps, ranked results, robustness zone detection
- **Analyzer**: H1-H5 hypothesis evaluators, Supabase persistence (optional)
- **Config**: YAML + python-dotenv for secrets
