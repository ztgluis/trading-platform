"""Fetch a symbol and print the resulting DataFrame.

Usage:
    python -m scripts.fetch_sample AAPL Daily
    python -m scripts.fetch_sample BTC/USDT 4H
    python -m scripts.fetch_sample AAPL Daily --inverse
"""

import argparse
import logging
import sys

from trade_analysis.data_manager import DataManager
from trade_analysis.models.ohlcv import Timeframe


def main():
    parser = argparse.ArgumentParser(description="Fetch OHLCV data for a symbol")
    parser.add_argument("symbol", help="Ticker symbol (e.g., AAPL, BTC/USDT)")
    parser.add_argument("timeframe", help="Timeframe (1H, 4H, Daily, Weekly, Monthly)")
    parser.add_argument("--inverse", action="store_true", help="Compute inverse series")
    parser.add_argument("--refresh", action="store_true", help="Force cache refresh")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        tf = Timeframe(args.timeframe)
    except ValueError:
        print(f"Invalid timeframe: {args.timeframe}")
        print(f"Valid options: {[t.value for t in Timeframe]}")
        sys.exit(1)

    dm = DataManager()
    df = dm.get_ohlcv(
        args.symbol, tf,
        inverse=args.inverse,
        force_refresh=args.refresh,
    )

    print(f"\n{'='*60}")
    print(f"Symbol:    {df.attrs.get('symbol', args.symbol)}")
    print(f"Timeframe: {df.attrs.get('timeframe', args.timeframe)}")
    print(f"Provider:  {df.attrs.get('provider', 'unknown')}")
    print(f"Inverse:   {df.attrs.get('is_inverse', False)}")
    print(f"Rows:      {len(df)}")
    print(f"Range:     {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
    print(f"{'='*60}\n")

    print(df.to_string(max_rows=20))

    print(f"\n--- Summary Stats ---")
    print(df[["open", "high", "low", "close", "volume"]].describe().to_string())


if __name__ == "__main__":
    main()
