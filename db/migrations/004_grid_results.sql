-- M5: Grid runner tables for persisting parameter sweep results
-- Apply to Supabase via SQL Editor or CLI

-- Grid sweep run metadata
CREATE TABLE IF NOT EXISTS grid_runs (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    asset_class VARCHAR(10) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,

    -- Grid configuration
    parameters JSONB NOT NULL,      -- swept parameter ranges
    min_trades INTEGER NOT NULL DEFAULT 30,
    rank_by VARCHAR(20) NOT NULL DEFAULT 'total_r',
    total_combos INTEGER NOT NULL DEFAULT 0,
    sufficient_combos INTEGER NOT NULL DEFAULT 0,

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Individual parameter combination results
CREATE TABLE IF NOT EXISTS grid_results (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES grid_runs(id) ON DELETE CASCADE,

    -- Parameters used (flattened JSON)
    params JSONB NOT NULL,

    -- Summary stats
    total_trades INTEGER NOT NULL DEFAULT 0,
    win_rate DOUBLE PRECISION,
    avg_r DOUBLE PRECISION,
    total_r DOUBLE PRECISION,
    profit_factor DOUBLE PRECISION,
    max_drawdown_r DOUBLE PRECISION,
    sufficient_trades BOOLEAN NOT NULL DEFAULT FALSE,

    -- Robustness (populated by M6 or post-analysis)
    is_robust BOOLEAN,
    is_isolated_peak BOOLEAN,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_grid_runs_symbol
    ON grid_runs(symbol, asset_class, timeframe);

CREATE INDEX IF NOT EXISTS idx_grid_results_run
    ON grid_results(run_id);

CREATE INDEX IF NOT EXISTS idx_grid_results_total_r
    ON grid_results(total_r DESC);

CREATE INDEX IF NOT EXISTS idx_grid_results_sufficient
    ON grid_results(sufficient_trades)
    WHERE sufficient_trades = TRUE;
