"""Tests for M3 signal engine config loader (engine.py)."""

from pathlib import Path

import pytest
import yaml

from trade_analysis.exceptions import ConfigError
from trade_analysis.signals.engine import (
    BucketConfig,
    SignalEngineConfig,
    get_bucket_for_asset,
    load_signal_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    """Write a dict to a temporary YAML file and return its path."""
    p = tmp_path / "signals.yaml"
    p.write_text(yaml.dump(data, default_flow_style=False))
    return p


def _minimal_config() -> dict:
    """Return the smallest valid config dict."""
    return {
        "signals": {
            "buckets": {
                "A": {
                    "name": "Short Swing",
                    "asset_classes": ["stock", "etf", "crypto"],
                    "primary_timeframe": "4H",
                    "confirmation_timeframe": "Daily",
                    "trend_ma_type": "ema",
                    "trend_ma_period": 21,
                    "target_r_multiple": 2.0,
                    "trail_breakeven_r": 1.0,
                },
                "B": {
                    "name": "Long Swing",
                    "asset_classes": ["index", "metal"],
                    "primary_timeframe": "Weekly",
                    "confirmation_timeframe": "Monthly",
                    "trend_ma_type": "sma",
                    "trend_ma_period": 50,
                    "target_r_multiple": 3.0,
                    "trail_breakeven_r": 1.5,
                },
            },
        },
    }


# ---------------------------------------------------------------------------
# load_signal_config — happy path
# ---------------------------------------------------------------------------


class TestLoadSignalConfig:
    """Test loading signal engine configuration from YAML."""

    def test_load_real_config(self):
        """Load the actual config/signals.yaml shipped with the project."""
        config_path = Path("config/signals.yaml")
        if not config_path.exists():
            pytest.skip("config/signals.yaml not found (CI or relocated)")

        cfg = load_signal_config(config_path)

        assert isinstance(cfg, SignalEngineConfig)
        assert isinstance(cfg.bucket_a, BucketConfig)
        assert isinstance(cfg.bucket_b, BucketConfig)

    def test_bucket_a_values(self):
        """Bucket A has the expected Short Swing values from the real config."""
        cfg = load_signal_config(Path("config/signals.yaml"))
        a = cfg.bucket_a

        assert a.name == "Short Swing"
        assert a.asset_classes == ["stock", "etf", "crypto"]
        assert a.primary_timeframe == "4H"
        assert a.confirmation_timeframe == "Daily"
        assert a.trend_ma_type == "ema"
        assert a.trend_ma_period == 21
        assert a.max_hold_weeks == 4
        assert a.target_r_multiple == 2.0
        assert a.trail_breakeven_r == 1.0

    def test_bucket_b_values(self):
        """Bucket B has the expected Long Swing values from the real config."""
        cfg = load_signal_config(Path("config/signals.yaml"))
        b = cfg.bucket_b

        assert b.name == "Long Swing"
        assert b.asset_classes == ["index", "metal"]
        assert b.primary_timeframe == "Weekly"
        assert b.confirmation_timeframe == "Monthly"
        assert b.trend_ma_type == "sma"
        assert b.trend_ma_period == 50
        assert b.max_hold_weeks is None
        assert b.target_r_multiple == 3.0
        assert b.trail_breakeven_r == 1.5

    def test_regime_values(self):
        """Regime parameters loaded correctly."""
        cfg = load_signal_config(Path("config/signals.yaml"))

        assert cfg.regime_ma_type == "sma"
        assert cfg.regime_ma_period == 200
        assert cfg.regime_transition_closes == 3
        assert cfg.regime_strong_alignment_pct == 5.0

    def test_condition_values(self):
        """Structure and momentum condition parameters loaded correctly."""
        cfg = load_signal_config(Path("config/signals.yaml"))

        # Structure
        assert cfg.swing_lookback == 3
        assert cfg.level_proximity_pct == 3.0
        assert cfg.pivot_lookback == 5
        assert cfg.pivot_merge_distance_pct == 0.5

        # Momentum
        assert cfg.rsi_period == 14
        assert cfg.rsi_bull_threshold == 50
        assert cfg.rsi_bear_threshold == 50
        assert cfg.macd_fast == 12
        assert cfg.macd_slow == 26
        assert cfg.macd_signal == 9

    def test_scoring_values(self):
        """Scoring weights and threshold loaded correctly."""
        cfg = load_signal_config(Path("config/signals.yaml"))

        assert cfg.scoring_weights["trend_confirmed"] == 1
        assert cfg.scoring_weights["structure_single_method"] == 1
        assert cfg.scoring_weights["structure_multi_method"] == 2
        assert cfg.scoring_weights["momentum_confirmed"] == 1
        assert cfg.scoring_weights["regime_strongly_aligned"] == 1
        assert cfg.scoring_weights["volume_spike_on_entry"] == 1
        assert cfg.tradeable_threshold == 3

    def test_volume_values(self):
        """Volume parameters loaded correctly."""
        cfg = load_signal_config(Path("config/signals.yaml"))

        assert cfg.volume_sma_period == 20
        assert cfg.volume_spike_threshold == 1.5

    def test_exit_values(self):
        """Exit parameters loaded correctly."""
        cfg = load_signal_config(Path("config/signals.yaml"))

        assert cfg.atr_period == 14
        assert cfg.stop_method == "swing"
        assert cfg.atr_stop_multiplier == 1.5


# ---------------------------------------------------------------------------
# load_signal_config — defaults fallback
# ---------------------------------------------------------------------------


class TestConfigDefaults:
    """Test that missing optional sections fall back to sensible defaults."""

    def test_minimal_config_loads(self, tmp_path):
        """A config with only buckets should load using defaults for the rest."""
        p = _write_yaml(tmp_path, _minimal_config())
        cfg = load_signal_config(p)

        # Regime defaults
        assert cfg.regime_ma_type == "sma"
        assert cfg.regime_ma_period == 200
        assert cfg.regime_transition_closes == 3
        assert cfg.regime_strong_alignment_pct == 5.0

        # Momentum defaults
        assert cfg.rsi_period == 14
        assert cfg.macd_fast == 12
        assert cfg.macd_slow == 26
        assert cfg.macd_signal == 9

        # Volume defaults
        assert cfg.volume_sma_period == 20
        assert cfg.volume_spike_threshold == 1.5

        # Exit defaults
        assert cfg.atr_period == 14
        assert cfg.stop_method == "swing"
        assert cfg.atr_stop_multiplier == 1.5

        # Scoring defaults
        assert cfg.tradeable_threshold == 3

    def test_max_hold_weeks_none(self, tmp_path):
        """max_hold_weeks can be null (None) in YAML."""
        data = _minimal_config()
        data["signals"]["buckets"]["A"]["max_hold_weeks"] = None
        p = _write_yaml(tmp_path, data)
        cfg = load_signal_config(p)

        assert cfg.bucket_a.max_hold_weeks is None

    def test_max_hold_weeks_integer(self, tmp_path):
        """max_hold_weeks can be an integer."""
        data = _minimal_config()
        data["signals"]["buckets"]["A"]["max_hold_weeks"] = 6
        p = _write_yaml(tmp_path, data)
        cfg = load_signal_config(p)

        assert cfg.bucket_a.max_hold_weeks == 6


# ---------------------------------------------------------------------------
# load_signal_config — error cases
# ---------------------------------------------------------------------------


class TestConfigErrors:
    """Test config validation and error handling."""

    def test_file_not_found(self, tmp_path):
        """Non-existent config file raises ConfigError."""
        with pytest.raises(ConfigError, match="not found"):
            load_signal_config(tmp_path / "nonexistent.yaml")

    def test_missing_signals_key(self, tmp_path):
        """YAML without 'signals' top-level key raises ConfigError."""
        p = _write_yaml(tmp_path, {"something_else": {}})
        with pytest.raises(ConfigError, match="signals"):
            load_signal_config(p)

    def test_empty_file(self, tmp_path):
        """Empty YAML file raises ConfigError."""
        p = tmp_path / "signals.yaml"
        p.write_text("")
        with pytest.raises(ConfigError, match="signals"):
            load_signal_config(p)

    def test_missing_bucket_a(self, tmp_path):
        """Config without bucket A raises ConfigError."""
        data = _minimal_config()
        del data["signals"]["buckets"]["A"]
        p = _write_yaml(tmp_path, data)
        with pytest.raises(ConfigError, match="bucket"):
            load_signal_config(p)

    def test_missing_bucket_b(self, tmp_path):
        """Config without bucket B raises ConfigError."""
        data = _minimal_config()
        del data["signals"]["buckets"]["B"]
        p = _write_yaml(tmp_path, data)
        with pytest.raises(ConfigError, match="bucket"):
            load_signal_config(p)

    def test_missing_required_bucket_field(self, tmp_path):
        """Bucket missing a required field raises ConfigError."""
        data = _minimal_config()
        del data["signals"]["buckets"]["A"]["name"]
        p = _write_yaml(tmp_path, data)
        with pytest.raises(ConfigError, match="name"):
            load_signal_config(p)

    def test_missing_asset_classes(self, tmp_path):
        """Bucket missing asset_classes raises ConfigError."""
        data = _minimal_config()
        del data["signals"]["buckets"]["B"]["asset_classes"]
        p = _write_yaml(tmp_path, data)
        with pytest.raises(ConfigError, match="asset_classes"):
            load_signal_config(p)


# ---------------------------------------------------------------------------
# BucketConfig immutability
# ---------------------------------------------------------------------------


class TestBucketConfigFrozen:
    """BucketConfig is a frozen dataclass."""

    def test_bucket_is_frozen(self):
        """Cannot modify BucketConfig fields after creation."""
        b = BucketConfig(
            name="Test",
            asset_classes=["stock"],
            primary_timeframe="4H",
            confirmation_timeframe="Daily",
            trend_ma_type="ema",
            trend_ma_period=21,
            max_hold_weeks=4,
            target_r_multiple=2.0,
            trail_breakeven_r=1.0,
        )
        with pytest.raises(AttributeError):
            b.name = "Changed"  # type: ignore[misc]


class TestSignalEngineConfigFrozen:
    """SignalEngineConfig is a frozen dataclass."""

    def test_config_is_frozen(self):
        """Cannot modify SignalEngineConfig fields after creation."""
        cfg = load_signal_config(Path("config/signals.yaml"))
        with pytest.raises(AttributeError):
            cfg.regime_ma_period = 100  # type: ignore[misc]


# ---------------------------------------------------------------------------
# get_bucket_for_asset
# ---------------------------------------------------------------------------


class TestGetBucketForAsset:
    """Test bucket-for-asset mapping."""

    @pytest.fixture
    def config(self) -> SignalEngineConfig:
        return load_signal_config(Path("config/signals.yaml"))

    def test_stock_maps_to_bucket_a(self, config):
        bucket = get_bucket_for_asset("stock", config)
        assert bucket.name == "Short Swing"

    def test_etf_maps_to_bucket_a(self, config):
        bucket = get_bucket_for_asset("etf", config)
        assert bucket.name == "Short Swing"

    def test_crypto_maps_to_bucket_a(self, config):
        bucket = get_bucket_for_asset("crypto", config)
        assert bucket.name == "Short Swing"

    def test_index_maps_to_bucket_b(self, config):
        bucket = get_bucket_for_asset("index", config)
        assert bucket.name == "Long Swing"

    def test_metal_maps_to_bucket_b(self, config):
        bucket = get_bucket_for_asset("metal", config)
        assert bucket.name == "Long Swing"

    def test_case_insensitive(self, config):
        """Asset class lookup is case-insensitive."""
        bucket = get_bucket_for_asset("STOCK", config)
        assert bucket.name == "Short Swing"

        bucket = get_bucket_for_asset("ETF", config)
        assert bucket.name == "Short Swing"

        bucket = get_bucket_for_asset("Metal", config)
        assert bucket.name == "Long Swing"

    def test_unknown_asset_class_raises(self, config):
        """Unknown asset class raises ConfigError."""
        with pytest.raises(ConfigError, match="not mapped"):
            get_bucket_for_asset("forex", config)

    def test_empty_string_raises(self, config):
        """Empty string as asset class raises ConfigError."""
        with pytest.raises(ConfigError, match="not mapped"):
            get_bucket_for_asset("", config)
