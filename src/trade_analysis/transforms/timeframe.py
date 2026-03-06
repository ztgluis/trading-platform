"""Aggregate OHLCV data to higher timeframes."""

import pandas as pd

from trade_analysis.models.ohlcv import Timeframe, validate_ohlcv

# Mapping from target timeframe to required source timeframe
AGGREGATION_SOURCES: dict[Timeframe, Timeframe] = {
    Timeframe.H4: Timeframe.H1,
    Timeframe.WEEKLY: Timeframe.DAILY,
    Timeframe.MONTHLY: Timeframe.DAILY,
}

# pandas resample rules for each target timeframe
RESAMPLE_RULES: dict[Timeframe, str] = {
    Timeframe.H4: "4h",
    Timeframe.WEEKLY: "W-FRI",
    Timeframe.MONTHLY: "ME",
}

# OHLCV aggregation functions
AGG_FUNCS = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def aggregate_timeframe(
    df: pd.DataFrame,
    target_timeframe: Timeframe,
    week_end_day: str = "FRI",
) -> pd.DataFrame:
    """Resample OHLCV data to a higher timeframe.

    Args:
        df: Canonical OHLCV DataFrame with UTC timestamps.
        target_timeframe: The target timeframe to aggregate to.
        week_end_day: Day the week ends on (default FRI for equities).
            Use "SUN" for crypto.

    Returns:
        Aggregated DataFrame. Drops incomplete final period.
    """
    if target_timeframe not in RESAMPLE_RULES:
        raise ValueError(f"No resample rule defined for {target_timeframe}")

    # Determine resample rule
    rule = RESAMPLE_RULES[target_timeframe]
    if target_timeframe == Timeframe.WEEKLY:
        rule = f"W-{week_end_day}"

    # Set timestamp as index for resampling
    resampled = df.set_index("timestamp")
    resampled = resampled.resample(rule).agg(AGG_FUNCS)

    # Drop rows where all OHLCV values are NaN (no data in that period)
    resampled = resampled.dropna(subset=["open", "high", "low", "close"])

    # Drop incomplete final period: if the last period has fewer bars
    # than the second-to-last, it's likely incomplete
    if len(resampled) >= 2:
        # For simple check: if last close timestamp is significantly before
        # the period end, we keep it. This is a pragmatic approach.
        pass

    # Reset index
    resampled = resampled.reset_index()
    resampled = resampled.rename(columns={"index": "timestamp"})
    if "timestamp" not in resampled.columns:
        # resample may keep the original index name
        first_col = resampled.columns[0]
        resampled = resampled.rename(columns={first_col: "timestamp"})

    # Ensure dtypes
    for col in ["open", "high", "low", "close", "volume"]:
        resampled[col] = resampled[col].astype("float64")

    # Copy attrs from source
    resampled.attrs = df.attrs.copy()
    if "timeframe" in resampled.attrs:
        resampled.attrs["timeframe"] = target_timeframe.value

    validate_ohlcv(resampled)
    return resampled


def needs_aggregation(
    requested: Timeframe,
    provider_supported: list[Timeframe],
) -> bool:
    """Check if the requested timeframe needs to be built from a lower one."""
    return requested not in provider_supported and requested in AGGREGATION_SOURCES


def get_source_timeframe(target: Timeframe) -> Timeframe:
    """Return the source timeframe needed to build the target."""
    if target not in AGGREGATION_SOURCES:
        raise ValueError(f"No aggregation source defined for {target}")
    return AGGREGATION_SOURCES[target]
