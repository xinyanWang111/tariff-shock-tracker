"""
Unit tests for src/event_study.py
Run with:  pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from src.event_study import (
    estimate_market_model,
    run_event_study,
    compute_caar,
    results_to_dataframe,
)


@pytest.fixture
def synthetic_data():
    """
    Build 300 days of synthetic returns where the true model is
    stock = 0.0005 + 1.2 * market + noise.
    """
    rng = np.random.default_rng(seed=123)
    dates = pd.date_range("2024-01-01", periods=300, freq="B")
    market = pd.Series(rng.normal(0.0003, 0.012, 300), index=dates, name="^GSPC")
    stock = (0.0005 + 1.2 * market
             + pd.Series(rng.normal(0, 0.008, 300), index=dates))
    stock.name = "TEST"
    return stock, market


# =============================================================
# Market model estimation
# =============================================================
class TestMarketModel:
    def test_beta_recovers_true_value(self, synthetic_data):
        stock, market = synthetic_data
        alpha, beta, sigma = estimate_market_model(stock, market)
        assert 1.0 < beta < 1.4
        assert sigma > 0
        assert abs(alpha) < 0.005

    def test_raises_on_short_window(self):
        short = pd.Series(np.random.randn(20))
        with pytest.raises(ValueError, match="too short"):
            estimate_market_model(short, short)


# =============================================================
# Event study
# =============================================================
class TestRunEventStudy:
    def test_returns_valid_result(self, synthetic_data):
        stock, market = synthetic_data
        event_date = stock.index[200]
        result = run_event_study(
            stock, market, event_date,
            estimation_window=100, estimation_gap=10,
            event_window_pre=5, event_window_post=10,
        )
        assert result.ticker == "TEST"
        assert len(result.ar) == 16  # -5 to +10 inclusive
        assert isinstance(result.car, float)
        assert isinstance(result.beta, float)

    def test_ar_index_is_relative_time(self, synthetic_data):
        stock, market = synthetic_data
        event_date = stock.index[200]
        result = run_event_study(
            stock, market, event_date,
            estimation_window=100, estimation_gap=10,
            event_window_pre=5, event_window_post=10,
        )
        assert list(result.ar.index) == list(range(-5, 11))

    def test_event_too_early_raises(self, synthetic_data):
        stock, market = synthetic_data
        # Event on day 10 — not enough pre-event data
        event_date = stock.index[10]
        with pytest.raises(ValueError, match="Not enough pre-event data"):
            run_event_study(
                stock, market, event_date,
                estimation_window=120, estimation_gap=30,
            )


# =============================================================
# CAAR aggregation
# =============================================================
class TestCAAR:
    def test_caar_shape(self, synthetic_data):
        stock, market = synthetic_data
        event_date = stock.index[200]
        # Build 3 stocks with same market
        results = []
        for i in range(3):
            s = stock + pd.Series(
                np.random.default_rng(seed=i).normal(0, 0.005, len(stock)),
                index=stock.index,
            )
            s.name = f"TEST_{i}"
            results.append(run_event_study(
                s, market, event_date,
                estimation_window=100, estimation_gap=10,
            ))
        caar_df = compute_caar(results)
        assert set(caar_df.columns) == {"relative_day", "aar", "caar"}
        assert len(caar_df) == 16

    def test_results_to_dataframe(self, synthetic_data):
        stock, market = synthetic_data
        event_date = stock.index[200]
        result = run_event_study(
            stock, market, event_date,
            estimation_window=100, estimation_gap=10,
        )
        df = results_to_dataframe([result])
        assert "ticker" in df.columns
        assert "car" in df.columns
        assert "p_value" in df.columns
        assert len(df) == 1
