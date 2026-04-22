"""
Unit tests for src/risk_metrics.py
Run with:  pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from src.risk_metrics import (
    compute_beta,
    compute_sharpe_ratio,
    compute_rolling_volatility,
    compute_realised_volatility,
    build_risk_summary,
)


# =============================================================
# Fixtures
# =============================================================
@pytest.fixture
def sample_returns():
    """Generate reproducible synthetic returns for tests."""
    rng = np.random.default_rng(seed=42)
    dates = pd.date_range("2024-01-01", periods=252, freq="B")
    market = pd.Series(rng.normal(0.0005, 0.01, 252), index=dates, name="market")
    # Stock = 1.5 * market + noise -> true beta ≈ 1.5
    stock = 1.5 * market + pd.Series(rng.normal(0, 0.005, 252), index=dates)
    stock.name = "stock"
    return stock, market


# =============================================================
# Beta tests
# =============================================================
class TestBeta:
    def test_beta_recovers_true_value(self, sample_returns):
        """Beta estimate should be close to the true slope of 1.5."""
        stock, market = sample_returns
        beta = compute_beta(stock, market)
        assert 1.3 < beta < 1.7, f"Expected beta near 1.5, got {beta}"

    def test_beta_of_market_with_itself_is_one(self, sample_returns):
        """Beta of market regressed on itself must equal 1."""
        _, market = sample_returns
        beta = compute_beta(market, market)
        assert beta == pytest.approx(1.0, abs=1e-10)

    def test_beta_raises_on_too_few_observations(self):
        """Should raise ValueError for fewer than 30 observations."""
        short = pd.Series([0.01, 0.02, -0.01])
        with pytest.raises(ValueError):
            compute_beta(short, short)


# =============================================================
# Sharpe ratio tests
# =============================================================
class TestSharpeRatio:
    def test_sharpe_returns_float(self, sample_returns):
        stock, _ = sample_returns
        sharpe = compute_sharpe_ratio(stock)
        assert isinstance(sharpe, float)

    def test_sharpe_annualization(self, sample_returns):
        """Annualized Sharpe should be sqrt(252) times the daily Sharpe."""
        stock, _ = sample_returns
        daily = compute_sharpe_ratio(stock, annualize=False)
        annual = compute_sharpe_ratio(stock, annualize=True)
        assert annual == pytest.approx(daily * np.sqrt(252), rel=1e-6)

    def test_sharpe_of_constant_returns_is_nan(self):
        """Zero-variance returns should yield NaN Sharpe."""
        constant = pd.Series([0.01] * 50)
        assert np.isnan(compute_sharpe_ratio(constant))


# =============================================================
# Volatility tests
# =============================================================
class TestVolatility:
    def test_realised_volatility_positive(self, sample_returns):
        stock, _ = sample_returns
        vol = compute_realised_volatility(stock)
        assert vol > 0

    def test_rolling_volatility_length(self, sample_returns):
        stock, _ = sample_returns
        vol = compute_rolling_volatility(stock, window=30)
        # First 29 values should be NaN due to min_periods
        assert vol.iloc[:29].isna().all()
        assert vol.iloc[29:].notna().all()


# =============================================================
# Summary table tests
# =============================================================
class TestRiskSummary:
    def test_summary_shape(self, sample_returns):
        stock, market = sample_returns
        df = pd.DataFrame({"STK1": stock, "STK2": stock * 0.8})
        summary = build_risk_summary(df, market)
        assert summary.shape == (2, 6)
        assert list(summary.index) == ["STK1", "STK2"]

    def test_summary_columns(self, sample_returns):
        stock, market = sample_returns
        df = pd.DataFrame({"STK": stock})
        summary = build_risk_summary(df, market)
        expected_cols = {
            "beta", "alpha_daily", "alpha_annualized",
            "sharpe_ratio", "volatility_annualized",
            "mean_return_annualized",
        }
        assert set(summary.columns) == expected_cols
