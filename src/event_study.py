"""
event_study.py
==============
Event study methodology for measuring abnormal returns around tariff events.

Implements the classic market-model approach (MacKinlay, 1997):
    1. Estimate alpha and beta on a quiet pre-event estimation window.
    2. Use the estimated coefficients to predict expected returns during
       the event window.
    3. The residual (actual - expected) is the Abnormal Return (AR).
    4. Sum AR over the event window to get the Cumulative Abnormal Return (CAR).

Author: xinyan.Wang
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# =============================================================
# Data container
# =============================================================
@dataclass
class EventStudyResult:
    """Container for the results of a single stock event study."""

    ticker: str
    event_date: pd.Timestamp
    alpha: float
    beta: float
    ar: pd.Series           # daily abnormal returns over the event window
    car: float              # cumulative abnormal return over the event window
    t_stat: float           # CAR t-statistic
    p_value: float          # two-sided p-value
    event_window: tuple[int, int]


# =============================================================
# Core estimation
# =============================================================
def estimate_market_model(
    stock_returns: pd.Series,
    market_returns: pd.Series,
) -> tuple[float, float, float]:
    """
    Fit the market model R_i = alpha + beta * R_m + epsilon by OLS.

    Parameters
    ----------
    stock_returns : pd.Series
        Returns of the stock over the estimation window.
    market_returns : pd.Series
        Returns of the market index over the same window.

    Returns
    -------
    alpha : float
        Intercept.
    beta : float
        Market-beta coefficient.
    sigma : float
        Residual standard deviation (used for t-stats later).
    """
    # Align and drop any rows with NaN
    df = pd.concat([stock_returns, market_returns], axis=1).dropna()
    df.columns = ["stock", "market"]

    if len(df) < 30:
        raise ValueError(
            f"Estimation window too short ({len(df)} obs). Need at least 30."
        )

    x = df["market"].values
    y = df["stock"].values

    # Closed-form OLS (numerically stable enough for this use case)
    x_mean, y_mean = x.mean(), y.mean()
    beta = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    alpha = y_mean - beta * x_mean

    residuals = y - (alpha + beta * x)
    sigma = np.sqrt(np.sum(residuals ** 2) / (len(df) - 2))

    return float(alpha), float(beta), float(sigma)


# =============================================================
# Full event study for one stock
# =============================================================
def run_event_study(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    event_date: str | pd.Timestamp,
    estimation_window: int = 120,
    estimation_gap: int = 30,
    event_window_pre: int = 5,
    event_window_post: int = 10,
) -> EventStudyResult:
    """
    Run a complete single-stock event study.

    Parameters
    ----------
    stock_returns : pd.Series
        Daily returns of the stock (indexed by date).
    market_returns : pd.Series
        Daily returns of the market index (indexed by date).
    event_date : str or pd.Timestamp
        The event date (e.g., "2025-04-02").
    estimation_window : int
        Number of trading days used to estimate alpha/beta.
    estimation_gap : int
        Gap (in trading days) between the end of the estimation window
        and the start of the event window. Prevents information leakage.
    event_window_pre : int
        Days before the event to include in the event window.
    event_window_post : int
        Days after the event to include in the event window.

    Returns
    -------
    EventStudyResult
        Structured result with alpha, beta, AR series, CAR, t-stat, p-value.
    """
    ticker = stock_returns.name or "Unknown"
    event_ts = pd.Timestamp(event_date)

    # Align both series on the same trading-day index
    combined = pd.concat([stock_returns, market_returns], axis=1).dropna()
    combined.columns = ["stock", "market"]

    # Find the event index in the trading calendar
    # (use .get_indexer with method='bfill' to handle non-trading event dates)
    all_dates = combined.index
    if event_ts not in all_dates:
        # Snap to the next available trading day
        pos = all_dates.searchsorted(event_ts)
        if pos >= len(all_dates):
            raise ValueError(f"Event date {event_ts.date()} is after the data range.")
        event_ts = all_dates[pos]
        logger.info("Snapping event to next trading day: %s", event_ts.date())

    event_idx = all_dates.get_loc(event_ts)

    # Define windows
    est_end = event_idx - event_window_pre - estimation_gap
    est_start = est_end - estimation_window
    evt_start = event_idx - event_window_pre
    evt_end = event_idx + event_window_post + 1  # +1 because iloc end is exclusive

    if est_start < 0:
        raise ValueError(
            f"Not enough pre-event data for {ticker}. "
            f"Need {estimation_window + estimation_gap + event_window_pre} days before event."
        )

    est_window = combined.iloc[est_start:est_end]
    evt_window = combined.iloc[evt_start:evt_end]

    # Estimate market model on the pre-event window
    alpha, beta, sigma = estimate_market_model(
        est_window["stock"], est_window["market"]
    )

    # Compute abnormal returns over the event window
    expected = alpha + beta * evt_window["market"]
    ar = evt_window["stock"] - expected
    ar.index = range(-event_window_pre, event_window_post + 1)  # relative time

    car = ar.sum()

    # Standard error of CAR under the null of no abnormal performance
    se_car = sigma * np.sqrt(len(ar))
    t_stat = car / se_car if se_car > 0 else np.nan
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(est_window) - 2)) if not np.isnan(t_stat) else np.nan

    return EventStudyResult(
        ticker=ticker,
        event_date=event_ts,
        alpha=alpha,
        beta=beta,
        ar=ar,
        car=car,
        t_stat=t_stat,
        p_value=p_value,
        event_window=(-event_window_pre, event_window_post),
    )


# =============================================================
# Portfolio / aggregate statistics
# =============================================================
def compute_caar(results: list[EventStudyResult]) -> pd.DataFrame:
    """
    Compute the Cumulative Average Abnormal Return (CAAR) across a list
    of event studies.

    Parameters
    ----------
    results : list[EventStudyResult]
        Output from running `run_event_study` on multiple stocks.

    Returns
    -------
    pd.DataFrame
        Columns: ["relative_day", "aar", "caar"]
    """
    ar_matrix = pd.DataFrame({r.ticker: r.ar for r in results})
    aar = ar_matrix.mean(axis=1)        # Average AR across stocks per day
    caar = aar.cumsum()                 # Cumulative AAR

    return pd.DataFrame({
        "relative_day": aar.index,
        "aar": aar.values,
        "caar": caar.values,
    })


def results_to_dataframe(results: list[EventStudyResult]) -> pd.DataFrame:
    """
    Flatten a list of `EventStudyResult` into a summary DataFrame for
    reporting and visualisation.
    """
    return pd.DataFrame([
        {
            "ticker": r.ticker,
            "event_date": r.event_date.date(),
            "alpha": r.alpha,
            "beta": r.beta,
            "car": r.car,
            "t_stat": r.t_stat,
            "p_value": r.p_value,
            "significant_5pct": r.p_value < 0.05 if not np.isnan(r.p_value) else False,
        }
        for r in results
    ])
