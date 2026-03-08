"""Fresh Sweep page — run a grid sweep from the dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render(grid_df: pd.DataFrame, run_meta: pd.Series | None) -> None:
    """Render the fresh sweep configuration and execution page."""
    st.header("Run Fresh Sweep")
    st.markdown(
        "Configure and run a parameter grid sweep. Results will be stored "
        "in session and optionally persisted to Supabase."
    )

    # ---- Configuration ----
    col1, col2, col3 = st.columns(3)
    with col1:
        symbol = st.text_input("Symbol", value="AAPL", key="sweep_symbol")
    with col2:
        asset_class = st.selectbox(
            "Asset Class",
            options=["stock", "etf", "index", "crypto", "metal"],
            key="sweep_asset_class",
        )
    with col3:
        timeframe = st.selectbox(
            "Timeframe",
            options=["Daily", "4H", "Weekly"],
            key="sweep_timeframe",
        )

    st.subheader("Parameters to Sweep")
    st.markdown("Define parameter ranges as comma-separated values.")

    param_col1, param_col2 = st.columns(2)
    with param_col1:
        rsi_values = st.text_input(
            "RSI Period",
            value="10, 12, 14, 16, 18, 20",
            key="sweep_rsi",
        )
    with param_col2:
        ma_values = st.text_input(
            "Trend MA Period",
            value="10, 20, 30, 40, 50",
            key="sweep_ma",
        )

    min_trades = st.number_input("Min Trades", value=30, min_value=1, key="sweep_min_trades")
    rank_by = st.selectbox(
        "Rank By",
        options=["total_r", "avg_r", "profit_factor", "win_rate"],
        key="sweep_rank_by",
    )

    # ---- Parse parameters ----
    def _parse_int_list(s: str) -> list[int]:
        return [int(x.strip()) for x in s.split(",") if x.strip()]

    # ---- Run button ----
    if st.button("Run Sweep", type="primary"):
        try:
            params: dict[str, list] = {}
            if rsi_values.strip():
                params["rsi_period"] = _parse_int_list(rsi_values)
            if ma_values.strip():
                params["trend_ma_period"] = _parse_int_list(ma_values)

            if not params:
                st.error("Define at least one parameter to sweep.")
                return

            total_combos = 1
            for v in params.values():
                total_combos *= len(v)

            st.info(f"Running {total_combos} combinations...")

            with st.spinner("Fetching data and running grid sweep..."):
                from trade_analysis.data_manager import DataManager
                from trade_analysis.grid import GridConfig
                from trade_analysis.backtester import load_backtest_config
                from trade_analysis.signals import load_signal_config
                from trade_analysis.dashboard.data_loader import run_fresh_sweep
                from trade_analysis.models.ohlcv import Timeframe

                grid_config = GridConfig(
                    parameters=params,
                    symbol=symbol,
                    asset_class=asset_class,
                    timeframe=timeframe,
                    min_trades=min_trades,
                    rank_by=rank_by,
                )
                bt_config = load_backtest_config()
                signal_config = load_signal_config()

                dm = DataManager()
                ohlcv_df = dm.get_ohlcv(symbol, Timeframe(timeframe))

                result = run_fresh_sweep(grid_config, bt_config, signal_config, ohlcv_df)

            # Store in session state
            result_df = result.to_dataframe()
            st.session_state["fresh_grid_df"] = result_df
            st.session_state["fresh_run_meta"] = pd.Series({
                "symbol": symbol,
                "asset_class": asset_class,
                "timeframe": timeframe,
                "parameters": params,
                "min_trades": min_trades,
                "rank_by": rank_by,
                "total_combos": len(result_df),
                "sufficient_combos": len(
                    result_df[result_df["sufficient_trades"] == True]  # noqa: E712
                ) if "sufficient_trades" in result_df.columns else len(result_df),
            })

            st.success(
                f"Sweep complete! {len(result_df)} combinations tested. "
                f"Switch to Overview to explore results."
            )

        except Exception as exc:
            st.error(f"Sweep failed: {exc}")
            raise

    # ---- Show existing fresh results if any ----
    if "fresh_grid_df" in st.session_state and not st.session_state["fresh_grid_df"].empty:
        st.markdown("---")
        st.subheader("Current Session Results")
        fresh_df = st.session_state["fresh_grid_df"]
        st.markdown(f"**{len(fresh_df)}** combinations in memory.")

        if st.button("Clear Session Results"):
            del st.session_state["fresh_grid_df"]
            if "fresh_run_meta" in st.session_state:
                del st.session_state["fresh_run_meta"]
            st.rerun()
