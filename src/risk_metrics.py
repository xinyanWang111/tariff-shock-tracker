"""
risk_metrics.py
===============
Standard finance risk metrics: CAPM beta/alpha, Sharpe ratio, volatility.

All functions accept pandas Series (indexed by date) and return floats
or Series. No side effects — each function is a pure transformation,
making them trivial to unit-test.

References
----------
- CAPM : Sharpe (1964), Lintner (1965)
- Sharpe Ratio : Sharpe (1994), "The Sharpe Ratio", JPM

Author: Yiying
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Typical number of trading days per year (used to annualise)
TRADING_DAYS_PER_YEAR = 252


# =============================================================
# CAPM / Beta
# =============================================================
def compute_beta(
    stock_returns: pd.Series,
    market_returns: pd.Series,
) -> float:
    """
    Compute CAPM beta = Cov(R_i, R_m) / Var(R_m).

    Parameters
    ----------
    stock_returns : pd.Series
        Daily returns of the stock.
    market_returns : pd.Series
        Daily returns of the market index.

    Returns
    -------
    float
        Beta coefficient.
    """
    df = pd.concat([stock_returns, market_returns], axis=1).dropna()
    if len(df) < 30:
        raise ValueError(f"Need at least 30 observations, got {len(df)}.")

    cov = df.cov().iloc[0, 1]
    var_m = df.iloc[:, 1].var()
    return float(cov / var_m)


def compute_alpha(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Compute Jensen's alpha from the CAPM:
        alpha = E[R_i] - [R_f + beta * (E[R_m] - R_f)]

    Parameters
    ----------
    stock_returns : pd.Series
        Daily returns of the stock.
    market_returns : pd.Series
        Daily returns of the market index.
    risk_free_rate : float
        Daily risk-free rate (already de-annualised).

    Returns
    -------
    float
        Jensen's alpha (daily).
    """
    beta = compute_beta(stock_returns, market_returns)
    expected = risk_free_rate + beta * (market_returns.mean() - risk_free_rate)
    return float(stock_returns.mean() - expected)


# =============================================================
# Risk-adjusted returns
# =============================================================
def compute_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualize: bool = True,
) -> float:
    """
    Compute the Sharpe ratio: (mean(R) - R_f) / std(R).

    Parameters
    ----------
    returns : pd.Series
        Daily returns.
    risk_free_rate : float
        Daily risk-free rate (already de-annualised if annualize=True).
    annualize : bool
        If True, multiply by sqrt(252) to annualise.

    Returns
    -------
    float
        Sharpe ratio.
    """
    excess = returns.dropna() - risk_free_rate
    std = excess.std()
    if std == 0 or np.isnan(std):
        return np.nan
    sharpe = excess.mean() / std
    return float(sharpe * np.sqrt(TRADING_DAYS_PER_YEAR)) if annualize else float(sharpe)


# =============================================================
# Volatility
# =============================================================
def compute_rolling_volatility(
    returns: pd.Series,
    window: int = 30,
    annualize: bool = True,
) -> pd.Series:
    """
    Compute rolling-window volatility of returns.

    Parameters
    ----------
    returns : pd.Series
        Daily returns.
    window : int
        Rolling window in trading days.
    annualize : bool
        If True, multiply by sqrt(252).

    Returns
    -------
    pd.Series
        Rolling volatility series.
    """
    vol = returns.rolling(window=window).std()
    return vol * np.sqrt(TRADING_DAYS_PER_YEAR) if annualize else vol


def compute_realised_volatility(
    returns: pd.Series,
    annualize: bool = True,
) -> float:
    """
    Compute realised (sample) volatility of a return series.
    """
    std = returns.dropna().std()
    return float(std * np.sqrt(TRADING_DAYS_PER_YEAR)) if annualize else float(std)


# =============================================================
# Summary table for reporting
# =============================================================
def build_risk_summary(
    returns: pd.DataFrame,
    market_returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Build a per-stock summary of risk metrics.

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of daily stock returns (columns = tickers).
    market_returns : pd.Series
        Daily market returns.
    risk_free_rate : float
        Daily risk-free rate.

    Returns
    -------
    pd.DataFrame
        Index: ticker; columns: beta, alpha, sharpe, volatility, mean_return.
    """
    rows = []
    for ticker in returns.columns:
        r = returns[ticker].dropna()
        try:
            beta = compute_beta(r, market_returns)
            alpha = compute_alpha(r, market_returns, risk_free_rate)
            sharpe = compute_sharpe_ratio(r, risk_free_rate)
            vol = compute_realised_volatility(r)
            rows.append({
                "ticker": ticker,
                "beta": beta,
                "alpha_daily": alpha,
                "alpha_annualized": alpha * TRADING_DAYS_PER_YEAR,
                "sharpe_ratio": sharpe,
                "volatility_annualized": vol,
                "mean_return_annualized": r.mean() * TRADING_DAYS_PER_YEAR,
            })
        except ValueError as e:
            # Skip stocks with insufficient data
            rows.append({
                "ticker": ticker,
                "beta": np.nan,
                "alpha_daily": np.nan,
                "alpha_annualized": np.nan,
                "sharpe_ratio": np.nan,
                "volatility_annualized": np.nan,
                "mean_return_annualized": np.nan,
            })

    return pd.DataFrame(rows).set_index("ticker")
