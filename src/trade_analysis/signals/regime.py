"""Regime detection: bull / bear / transition classification.

Determines the market regime using a long-period moving average (default
SMA 200). Regime dictates which signal directions are allowed and provides
a "strongly aligned" bonus for scoring.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta


def detect_regime(
    df: pd.DataFrame,
    ma_type: str = "sma",
    ma_period: int = 200,
    transition_closes: int = 3,
    strong_alignment_pct: float = 5.0,
) -> pd.DataFrame:
    """Classify each bar into a market regime.

    Algorithm:
        1. Compute the regime MA (SMA or EMA) of the close price.
        2. Count consecutive closes above / below the MA.
        3. Regime = "bull" when >= *transition_closes* consecutive closes above MA,
           "bear" when >= *transition_closes* consecutive closes below MA,
           "transition" otherwise.
        4. Allow long signals in bull regime, short signals in bear regime.
           Both directions are allowed during transition.
        5. "Strongly aligned" when price is > *strong_alignment_pct* above (bull)
           or below (bear) the regime MA.

    Args:
        df: Canonical OHLCV DataFrame.
        ma_type: Moving average type ("sma" or "ema").
        ma_period: MA lookback period (default: 200).
        transition_closes: Consecutive closes above/below MA required to
            confirm regime (default: 3).
        strong_alignment_pct: Distance % threshold for "strongly aligned"
            (default: 5.0).

    Returns:
        DataFrame with columns:
            - regime_ma: the regime moving average value
            - regime: "bull" / "bear" / "transition"
            - regime_allow_long: bool — long signals permitted
            - regime_allow_short: bool — short signals permitted
            - regime_strongly_aligned: bool — price is far from MA
            - regime_distance_pct: signed distance from MA as percentage
    """
    result = df.copy()
    close = df["close"]

    # Step 1: Compute regime MA
    if ma_type == "ema":
        regime_ma = ta.ema(close, length=ma_period)
    else:
        regime_ma = ta.sma(close, length=ma_period)

    result["regime_ma"] = regime_ma

    # Step 2: Distance from MA (signed percentage)
    with np.errstate(divide="ignore", invalid="ignore"):
        distance_pct = ((close - regime_ma) / regime_ma * 100).replace(
            [np.inf, -np.inf], np.nan
        )
    result["regime_distance_pct"] = distance_pct

    # Step 3: Count consecutive closes above/below MA
    above = (close > regime_ma).astype(int)
    below = (close < regime_ma).astype(int)

    # Running count: resets to 0 when condition breaks
    n = len(df)
    consec_above = np.zeros(n, dtype=int)
    consec_below = np.zeros(n, dtype=int)

    for i in range(n):
        if np.isnan(regime_ma.iloc[i]):
            continue
        if above.iloc[i]:
            consec_above[i] = (consec_above[i - 1] + 1) if i > 0 else 1
        if below.iloc[i]:
            consec_below[i] = (consec_below[i - 1] + 1) if i > 0 else 1

    # Step 4: Classify regime with hysteresis
    # Once in bull/bear, stay there until the other side gets enough closes
    regime = pd.Series("transition", index=df.index)
    allow_long = pd.Series(True, index=df.index)
    allow_short = pd.Series(True, index=df.index)
    strongly_aligned = pd.Series(False, index=df.index)

    current_regime = "transition"
    for i in range(n):
        if np.isnan(regime_ma.iloc[i]):
            regime.iloc[i] = "transition"
            continue

        if consec_above[i] >= transition_closes:
            current_regime = "bull"
        elif consec_below[i] >= transition_closes:
            current_regime = "bear"

        regime.iloc[i] = current_regime

        if current_regime == "bull":
            allow_long.iloc[i] = True
            allow_short.iloc[i] = False
        elif current_regime == "bear":
            allow_long.iloc[i] = False
            allow_short.iloc[i] = True
        else:
            # Transition: allow both
            allow_long.iloc[i] = True
            allow_short.iloc[i] = True

    # Step 5: Strongly aligned
    dist = distance_pct.fillna(0.0)
    strongly_aligned = (
        ((regime == "bull") & (dist > strong_alignment_pct))
        | ((regime == "bear") & (dist < -strong_alignment_pct))
    )

    result["regime"] = regime
    result["regime_allow_long"] = allow_long
    result["regime_allow_short"] = allow_short
    result["regime_strongly_aligned"] = strongly_aligned
    result["regime_distance_pct"] = distance_pct

    return result
