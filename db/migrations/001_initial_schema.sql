-- M1: Initial schema for the swing trading platform
-- Apply to Supabase via SQL Editor or CLI

-- OHLCV price data
CREATE TABLE IF NOT EXISTS ohlcv_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    asset_class VARCHAR(10) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    provider VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, timeframe, provider, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_tf
    ON ohlcv_data(symbol, timeframe, timestamp);

CREATE INDEX IF NOT EXISTS idx_ohlcv_provider
    ON ohlcv_data(provider);

-- Symbol configuration (mirrors symbols.yaml for DB-driven workflows)
CREATE TABLE IF NOT EXISTS symbols (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL UNIQUE,
    asset_class VARCHAR(10) NOT NULL,
    provider VARCHAR(20) NOT NULL,
    exchange VARCHAR(20),
    inverse_ticker VARCHAR(20),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
