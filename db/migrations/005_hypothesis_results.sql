-- M6: Hypothesis evaluation results
-- Apply to Supabase via SQL Editor or CLI

CREATE TABLE IF NOT EXISTS hypothesis_results (
    id BIGSERIAL PRIMARY KEY,
    grid_run_id BIGINT REFERENCES grid_runs(id) ON DELETE SET NULL,

    hypothesis_id VARCHAR(10) NOT NULL,   -- "H1", "H2", etc.
    question TEXT NOT NULL,
    verdict VARCHAR(20) NOT NULL,         -- supported, refuted, inconclusive, not_testable
    evidence JSONB NOT NULL DEFAULT '{}',
    summary TEXT NOT NULL DEFAULT '',

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_hypothesis_results_run
    ON hypothesis_results(grid_run_id);

CREATE INDEX IF NOT EXISTS idx_hypothesis_results_verdict
    ON hypothesis_results(hypothesis_id, verdict);
