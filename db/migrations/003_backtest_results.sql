-- M4: Backtest results tables for persisting trade logs and run metadata
-- Apply to Supabase via SQL Editor or CLI

-- Backtest run metadata
CREATE TABLE IF NOT EXISTS backtest_runs (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    asset_class VARCHAR(10) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,

    -- Config
    initial_capital DOUBLE PRECISION NOT NULL DEFAULT 100000.0,
    max_open_positions SMALLINT NOT NULL DEFAULT 1,
    signal_config_hash VARCHAR(64),

    -- Summary stats
    total_trades INTEGER NOT NULL DEFAULT 0,
    win_rate DOUBLE PRECISION,
    avg_r DOUBLE PRECISION,
    total_r DOUBLE PRECISION,
    profit_factor DOUBLE PRECISION,
    max_drawdown_r DOUBLE PRECISION,
    max_consecutive_wins INTEGER,
    max_consecutive_losses INTEGER,
    sufficient_trades BOOLEAN NOT NULL DEFAULT FALSE,

    -- Walk-forward
    walk_forward_fold INTEGER,  -- NULL for full backtest, fold number for WF
    walk_forward_type VARCHAR(3) CHECK (walk_forward_type IN ('IS', 'OOS')),

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Individual trade records
CREATE TABLE IF NOT EXISTS backtest_trades (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES backtest_runs(id) ON DELETE CASCADE,

    symbol VARCHAR(20) NOT NULL,
    asset_class VARCHAR(10) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    bucket CHAR(1) NOT NULL CHECK (bucket IN ('A', 'B')),
    direction VARCHAR(5) NOT NULL CHECK (direction IN ('long', 'short')),

    -- Entry
    entry_timestamp TIMESTAMPTZ NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL,
    entry_signal_score SMALLINT NOT NULL,
    entry_regime VARCHAR(12) NOT NULL,

    -- Exit
    exit_timestamp TIMESTAMPTZ NOT NULL,
    exit_price DOUBLE PRECISION NOT NULL,
    exit_reason VARCHAR(15) NOT NULL CHECK (
        exit_reason IN ('stop', 'trail_stop', 'target', 'max_hold', 'end_of_data')
    ),

    -- P&L
    pnl_r DOUBLE PRECISION NOT NULL,
    pnl_dollar DOUBLE PRECISION NOT NULL,
    is_winner BOOLEAN NOT NULL,

    -- Duration
    duration_bars INTEGER NOT NULL,
    duration_calendar_days INTEGER NOT NULL,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_backtest_runs_symbol
    ON backtest_runs(symbol, timeframe, start_date, end_date);

CREATE INDEX IF NOT EXISTS idx_backtest_trades_run
    ON backtest_trades(run_id);

CREATE INDEX IF NOT EXISTS idx_backtest_trades_symbol
    ON backtest_trades(symbol, timeframe, entry_timestamp);

CREATE INDEX IF NOT EXISTS idx_backtest_trades_exit_reason
    ON backtest_trades(exit_reason);

CREATE INDEX IF NOT EXISTS idx_backtest_trades_winner
    ON backtest_trades(is_winner)
    WHERE is_winner = TRUE;
