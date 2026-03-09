-- M9: Order execution tables for proposals, orders, fills, positions
-- Apply to Supabase via SQL Editor or CLI

-- Order proposals generated from tradeable signals
CREATE TABLE IF NOT EXISTS order_proposals (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    asset_class VARCHAR(10) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    direction VARCHAR(5) NOT NULL CHECK (direction IN ('long', 'short')),
    entry_price DOUBLE PRECISION NOT NULL,
    stop_loss DOUBLE PRECISION NOT NULL,
    target_price DOUBLE PRECISION NOT NULL,
    rr_ratio DOUBLE PRECISION,
    signal_score SMALLINT,
    regime VARCHAR(12),
    config_hash VARCHAR(64),
    signal_id BIGINT REFERENCES signals(id),
    suggested_qty INT NOT NULL DEFAULT 0,
    status VARCHAR(20) NOT NULL DEFAULT 'pending_approval'
        CHECK (status IN ('pending_approval', 'approved', 'rejected', 'expired')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    decided_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_proposals_status
    ON order_proposals(status)
    WHERE status = 'pending_approval';

CREATE INDEX IF NOT EXISTS idx_proposals_symbol
    ON order_proposals(symbol, created_at DESC);

-- Orders placed with the broker (or simulated in dry-run)
CREATE TABLE IF NOT EXISTS orders (
    id BIGSERIAL PRIMARY KEY,
    proposal_id BIGINT NOT NULL REFERENCES order_proposals(id),
    schwab_order_id VARCHAR(64),
    symbol VARCHAR(20) NOT NULL,
    direction VARCHAR(5) NOT NULL CHECK (direction IN ('long', 'short')),
    qty INT NOT NULL CHECK (qty > 0),
    order_type VARCHAR(10) NOT NULL CHECK (order_type IN ('market', 'limit')),
    limit_price DOUBLE PRECISION,
    status VARCHAR(20) NOT NULL DEFAULT 'placed'
        CHECK (status IN ('placed', 'filled', 'partially_filled', 'cancelled', 'rejected')),
    dry_run BOOLEAN NOT NULL DEFAULT TRUE,
    placed_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_orders_status
    ON orders(status);

CREATE INDEX IF NOT EXISTS idx_orders_proposal
    ON orders(proposal_id);

-- Order fills (partial or complete)
CREATE TABLE IF NOT EXISTS fills (
    id BIGSERIAL PRIMARY KEY,
    order_id BIGINT NOT NULL REFERENCES orders(id),
    fill_price DOUBLE PRECISION NOT NULL,
    fill_qty INT NOT NULL CHECK (fill_qty > 0),
    commission DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    filled_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_fills_order
    ON fills(order_id);

-- Live positions (open and closed)
CREATE TABLE IF NOT EXISTS live_positions (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    direction VARCHAR(5) NOT NULL CHECK (direction IN ('long', 'short')),
    qty INT NOT NULL CHECK (qty > 0),
    avg_entry_price DOUBLE PRECISION NOT NULL,
    current_stop DOUBLE PRECISION,
    current_target DOUBLE PRECISION,
    order_id BIGINT REFERENCES orders(id),
    opened_at TIMESTAMPTZ DEFAULT NOW(),
    closed_at TIMESTAMPTZ,
    close_reason VARCHAR(30)
);

CREATE INDEX IF NOT EXISTS idx_positions_open
    ON live_positions(symbol)
    WHERE closed_at IS NULL;

-- Kill switch state
CREATE TABLE IF NOT EXISTS kill_switch (
    id BIGSERIAL PRIMARY KEY,
    enabled BOOLEAN NOT NULL DEFAULT FALSE,
    reason TEXT,
    toggled_at TIMESTAMPTZ DEFAULT NOW(),
    toggled_by VARCHAR(50) DEFAULT 'system'
);

-- Insert initial kill switch state (disabled)
INSERT INTO kill_switch (enabled, reason, toggled_by)
VALUES (FALSE, 'Initial state', 'migration')
ON CONFLICT DO NOTHING;
