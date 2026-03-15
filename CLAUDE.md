# CLAUDE.md — Project Context for Claude Code

## Project Overview

Personal swing trading platform for systematic strategy validation and live signal generation. Covers stocks, ETFs, indices, crypto, and metals. Organized into 9 milestones (M1–M9).

**Milestone Status:**
- M1 Data Layer — Complete
- M2 Indicator Library — Complete
- M3 Signal Engine — Complete
- M4 Backtester — Complete
- M5 Grid Runner — Complete
- M6 Results Analyzer — Complete
- M7 Streamlit Dashboard — Complete
- M8 Live Runner (Paper) — Complete
- M9 Live Runner (Real) — Complete (Schwab API execution, dry-run default, manual approval)

**PRD:** Full product requirements in `docs/swing_trading_prd.docx` (binary .docx — use `textutil -convert txt -stdout docs/swing_trading_prd.docx` on macOS to read). Contains 21 hypotheses (H1–H21), strategy rules, overfitting safeguards.

## Environment Setup

```bash
# Python 3.13 venv — IMPORTANT: pip shebang is broken, always use:
.venv/bin/python3 -m pip install -e ".[dev,dashboard,live]"

# Run tests
pytest tests/

# Run dashboard
streamlit run src/trade_analysis/dashboard/app.py

# Run live webhook server
uvicorn src.trade_analysis.live.app:app --reload
```

**Required env vars** (in `.env`):
- `SUPABASE_URL` / `SUPABASE_KEY` — optional, platform degrades to in-memory mode without them
- `WEBHOOK_SECRET` — required for live runner auth

## Architecture

```
src/trade_analysis/
├── config/       # YAML loader with ${ENV_VAR} resolution
├── models/       # OHLCVMeta (frozen dataclass), Timeframe/AssetClass enums
├── providers/    # yfinance, ccxt, schwab data providers
├── cache/        # Parquet-based OHLCV cache (TTL per timeframe)
├── transforms/   # Normalize, timeframe aggregation, inverse
├── indicators/   # MA/ATR/VIDYA, RSI/MACD, oscillators, trend signals, structure, levels
├── signals/      # Regime detection, 2-of-3 condition gate, scoring (0-6), exit levels
├── backtester/   # Bar-by-bar replay, stop/target/trail exits, walk-forward validation
├── grid/         # Parameter sweeps, GridResult (rank/top_n/sufficient_only), robustness zones
├── analyzer/     # H1-H5 hypothesis evaluators, SupabaseClient, persistence
├── dashboard/    # Streamlit + Plotly (8 pages incl. execution management)
├── live/         # FastAPI webhook: TradingView → signal engine → execution pipeline
├── execution/    # Schwab API execution (dry-run/live), risk manager, order lifecycle
├── data_manager.py   # Main orchestrator for OHLCV fetching
└── exceptions.py     # TradeAnalysisError hierarchy
```

**Config files** in `config/`:
- `symbols.yaml` — asset universe (12 symbols, 5 asset classes, buckets A/B)
- `data_sources.yaml` — provider configs (yfinance, ccxt, schwab)
- `signals.yaml` — engine params (buckets A/B, regime, conditions, scoring weights)
- `cache.yaml` — TTL, storage path, max age
- `backtest.yaml` — date ranges, capital, walk-forward params
- `grid.yaml` — parameter grid definitions
- `live.yaml` — webhook auth, server, persistence toggle, execution config (dry_run, limits)

**DB migrations** in `db/migrations/` (001–006): run in Supabase SQL editor in order.

## Key Conventions

### Immutable configs
All config/model classes use `@dataclass(frozen=True)`. Modify via `dataclasses.replace()`.

### SupabaseClient graceful degradation
`SupabaseClient()` in `analyzer/persistence.py` checks for env vars on init. If missing, `self.enabled = False` and all write operations become no-ops. Always check `sb.enabled` before assuming DB connectivity.

### Signal engine pipeline
`generate_signals(df, asset_class, config)` → DataFrame with 30+ columns (regime, conditions, direction, score 0-6, exit levels). Bucket A = stocks/ETFs/crypto (4H/Daily), Bucket B = indices/metals (Weekly/Monthly).

### Exception hierarchy
```
TradeAnalysisError
├── ConfigError
├── ProviderError
│   ├── ProviderConnectionError
│   ├── ProviderRateLimitError (retry_after_seconds)
│   └── SymbolNotFoundError
├── OHLCVValidationError
├── CacheError
├── BacktestError
└── ExecutionError
    ├── OrderRejectedError
    ├── KillSwitchEngagedError
    ├── PositionLimitError
    └── SchwabConnectionError
        └── SchwabTokenExpiredError
```

### Execution pipeline
Signal → `OrderExecutor.create_proposal()` → pending_approval → user approves via dashboard/API → `OrderExecutor.approve_proposal()` → SchwabClient places order → fill + position persisted. Kill switch and position limits enforced by `RiskManager`. Dry-run mode (default) simulates all Schwab interactions.

### Test patterns
- Flat structure in `tests/`, one file per module
- Shared fixtures in `tests/conftest.py`: `sample_daily_ohlcv`, `sample_1h_ohlcv`, `sample_200bar_ohlcv`, `sample_yfinance_raw`, `sample_ccxt_raw`
- Fixed random seeds (`np.random.default_rng(42)`) for reproducibility
- Mock external services (providers, Supabase) with `unittest.mock`
- Dashboard/live tests mock `st.*` and use FastAPI `TestClient`

### Import style
Absolute imports: `from trade_analysis.signals.engine import generate_signals`. Public APIs re-exported via `__init__.py`.

### Linting
Ruff with line-length=100, target py312. Rules: E, F, I, N, W, UP.

## Git Workflow

- Post-commit hook at `.git/hooks/post-commit` auto-pushes via `git push 2>/dev/null &`
- All commits are automatically pushed to origin
- SSH key auth configured

## Common Gotchas

1. **pip path**: The venv shebang points to an old path. Always use `.venv/bin/python3 -m pip` instead of `pip` or `.venv/bin/pip`.
2. **Edit tool**: Must read a file before editing it — re-read if context was lost.
3. **Sorted dicts in tests**: `sorted()` on string keys gives alphabetical order (e.g., "bear" before "bull"). Use value-based assertions, not index-based.
4. **OHLCV metadata**: Stored in `df.attrs` dict (symbol, asset_class, timeframe, provider, is_inverse, inverse_of).
5. **Dashboard testability**: Pure logic (chart builders, filter application, data expansion) is separated from Streamlit rendering for unit testing.
