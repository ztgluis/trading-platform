# Trade Analysis Platform

Personal swing trading platform for systematic strategy validation and signal generation across stocks, ETFs, indices, crypto, and metals.

## Architecture

The platform is organized into milestones:

| Milestone | Status | Description |
|-----------|--------|-------------|
| M1: Data Layer | Complete | Fetch and normalize OHLCV data across asset classes |
| M2: Indicator Library | Planned | Parameterized MA, RSI, MACD, volume indicators |
| M3: Signal Engine | Planned | 3-condition entry check with scoring |
| M4: Backtester | Planned | Historical replay with walk-forward splits |
| M5: Grid Runner | Planned | Parameter optimization across combinations |
| M6: Results Analyzer | Planned | Hypothesis testing + Supabase persistence |
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

### CLI smoke test

```bash
python -m scripts.fetch_sample AAPL Daily
python -m scripts.fetch_sample BTC/USDT 4H
python -m scripts.fetch_sample AAPL Daily --inverse
```

## Running Tests

```bash
pytest tests/ -v         # 108 tests
pytest tests/ -v --cov   # With coverage
```

## Tech Stack

- **Language**: Python 3.12+
- **Data**: pandas, pyarrow (Parquet cache)
- **Stocks/ETFs/Metals**: yfinance
- **Crypto**: CCXT
- **Database**: Supabase (PostgreSQL)
- **Config**: YAML + python-dotenv for secrets
