"""Tests for dashboard data_loader — pure transformation logic.

Supabase-dependent functions are tested via mocking; the key testable
function is expand_params() which flattens JSONB params into columns.
"""

from __future__ import annotations

import pandas as pd
import pytest

from trade_analysis.dashboard.data_loader import expand_params


# ---------------------------------------------------------------------------
# expand_params tests
# ---------------------------------------------------------------------------


class TestExpandParams:
    """Tests for JSONB params expansion."""

    def test_expands_dict_params_into_columns(self) -> None:
        df = pd.DataFrame({
            "id": [1, 2],
            "params": [
                {"rsi_period": 14, "trend_ma_period": 30},
                {"rsi_period": 16, "trend_ma_period": 40},
            ],
            "total_r": [5.0, 3.0],
        })
        result = expand_params(df)

        assert "params" not in result.columns
        assert "rsi_period" in result.columns
        assert "trend_ma_period" in result.columns
        assert list(result["rsi_period"]) == [14, 16]
        assert list(result["trend_ma_period"]) == [30, 40]
        assert list(result["total_r"]) == [5.0, 3.0]

    def test_preserves_existing_columns(self) -> None:
        df = pd.DataFrame({
            "id": [1],
            "params": [{"rsi_period": 14}],
            "win_rate": [0.65],
            "avg_r": [0.25],
        })
        result = expand_params(df)

        assert "id" in result.columns
        assert "win_rate" in result.columns
        assert "avg_r" in result.columns
        assert result.iloc[0]["win_rate"] == 0.65

    def test_handles_empty_dataframe(self) -> None:
        df = pd.DataFrame()
        result = expand_params(df)
        assert result.empty

    def test_handles_no_params_column(self) -> None:
        df = pd.DataFrame({"id": [1], "total_r": [5.0]})
        result = expand_params(df)
        assert list(result.columns) == ["id", "total_r"]

    def test_handles_none_params_values(self) -> None:
        df = pd.DataFrame({
            "id": [1, 2],
            "params": [{"rsi_period": 14}, None],
            "total_r": [5.0, 3.0],
        })
        result = expand_params(df)

        assert "rsi_period" in result.columns
        assert result.iloc[0]["rsi_period"] == 14
        # None params produce NaN for the expanded columns
        assert pd.isna(result.iloc[1]["rsi_period"])

    def test_handles_empty_dict_params(self) -> None:
        df = pd.DataFrame({
            "id": [1],
            "params": [{}],
            "total_r": [5.0],
        })
        result = expand_params(df)

        assert "params" not in result.columns
        assert "id" in result.columns
        assert "total_r" in result.columns

    def test_single_param(self) -> None:
        df = pd.DataFrame({
            "params": [{"rsi_period": 10}, {"rsi_period": 14}, {"rsi_period": 18}],
            "total_r": [2.0, 5.0, 3.0],
        })
        result = expand_params(df)

        assert list(result["rsi_period"]) == [10, 14, 18]
        assert len(result.columns) == 2  # rsi_period + total_r

    def test_many_params(self) -> None:
        df = pd.DataFrame({
            "params": [{
                "rsi_period": 14,
                "trend_ma_period": 30,
                "trend_ma_type": "ema",
                "atr_period": 14,
            }],
            "total_r": [5.0],
        })
        result = expand_params(df)

        assert len(result.columns) == 5  # 4 params + total_r
        assert result.iloc[0]["trend_ma_type"] == "ema"

    def test_preserves_index(self) -> None:
        df = pd.DataFrame(
            {"params": [{"rsi_period": 14}], "total_r": [5.0]},
            index=[42],
        )
        result = expand_params(df)
        assert result.index.tolist() == [42]

    def test_numeric_types_preserved(self) -> None:
        df = pd.DataFrame({
            "params": [{"rsi_period": 14, "target_r": 2.5}],
            "total_r": [5.0],
        })
        result = expand_params(df)

        assert result.iloc[0]["rsi_period"] == 14
        assert result.iloc[0]["target_r"] == 2.5

    def test_mixed_param_keys_across_rows(self) -> None:
        """Rows with different param keys produce NaN for missing keys."""
        df = pd.DataFrame({
            "params": [
                {"rsi_period": 14},
                {"trend_ma_period": 30},
            ],
            "total_r": [5.0, 3.0],
        })
        result = expand_params(df)

        assert "rsi_period" in result.columns
        assert "trend_ma_period" in result.columns
        assert result.iloc[0]["rsi_period"] == 14
        assert pd.isna(result.iloc[0]["trend_ma_period"])
        assert pd.isna(result.iloc[1]["rsi_period"])
        assert result.iloc[1]["trend_ma_period"] == 30
