"""
data_loader.py
==============
Download and clean market data for the Tariff Shock Tracker.

This module handles all data acquisition from Yahoo Finance via `yfinance`.
All functions are deterministic given the same inputs and are unit-tested
in tests/test_data_loader.py.

Author: xinyan,Wang
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml
import yfinance as yf

# Configure logging so progress is visible but not overwhelming
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================
# Configuration loading
# =============================================================
def load_config(config_path: str | Path = "config/events.yaml") -> dict:
    """
    Load the YAML configuration file.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML config file (default: config/events.yaml)

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info("Loaded config from %s", config_path)
    return config


def extract_all_tickers(config: dict) -> list[str]:
    """
    Walk the nested `stocks` section of the config and return a flat
    list of all unique tickers.

    Parameters
    ----------
    config : dict
        Loaded config dictionary.

    Returns
    -------
    list[str]
        Deduplicated list of tickers.
    """
    tickers: set[str] = set()
    for country_data in config["stocks"].values():
        for sector_stocks in country_data.values():
            for stock in sector_stocks:
                tickers.add(stock["ticker"])
    return sorted(tickers)


# =============================================================
# Data download
# =============================================================
def download_prices(
    tickers: Iterable[str],
    start_date: str,
    end_date: str,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Download adjusted closing prices for a list of tickers.

    Parameters
    ----------
    tickers : iterable of str
        List of ticker symbols (e.g. ["AAPL", "NVDA"]).
    start_date : str
        Start date in "YYYY-MM-DD" format.
    end_date : str
        End date in "YYYY-MM-DD" format.
    auto_adjust : bool
        Whether to use split/dividend-adjusted prices (default: True).

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date, columns are tickers, values are close prices.

    Notes
    -----
    yfinance occasionally returns an empty DataFrame for invalid tickers.
    Those are dropped silently with a warning in the logs.
    """
    tickers = list(tickers)
    logger.info("Downloading %d tickers from %s to %s", len(tickers), start_date, end_date)

    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="ticker",
    )

    # yfinance returns different column structures depending on # of tickers
    if len(tickers) == 1:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        # Multi-ticker: use the "Close" column from each ticker's sub-frame
        prices = pd.DataFrame(
            {t: raw[t]["Close"] for t in tickers if t in raw.columns.get_level_values(0)}
        )

    # Drop columns that are entirely NaN (invalid tickers)
    empty_cols = prices.columns[prices.isna().all()].tolist()
    if empty_cols:
        logger.warning("Dropping tickers with no data: %s", empty_cols)
        prices = prices.drop(columns=empty_cols)

    logger.info("Successfully downloaded %d tickers", prices.shape[1])
    return prices


# =============================================================
# Cleaning & transformation
# =============================================================
def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """
    Compute daily returns from a price DataFrame.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of prices indexed by date.
    method : {"log", "simple"}
        - "log"    : continuously compounded returns = ln(P_t / P_{t-1})
        - "simple" : arithmetic returns = P_t / P_{t-1} - 1

    Returns
    -------
    pd.DataFrame
        Returns with the same shape as `prices`, first row dropped.
    """
    if method == "log":
        import numpy as np
        returns = (prices / prices.shift(1)).apply(lambda x: np.log(x))
    elif method == "simple":
        returns = prices.pct_change()
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'log' or 'simple'.")

    return returns.dropna(how="all")


def save_to_processed(
    df: pd.DataFrame,
    filename: str,
    out_dir: str | Path = "data/processed",
) -> Path:
    """
    Persist a cleaned DataFrame to `data/processed/` as CSV.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    filename : str
        Output filename (e.g., "us_prices.csv").
    out_dir : str or Path
        Output directory.

    Returns
    -------
    Path
        Full path to the saved file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    df.to_csv(out_path)
    logger.info("Saved %s (%d rows × %d cols)", out_path, df.shape[0], df.shape[1])
    return out_path


# =============================================================
# CLI entry point
# =============================================================
if __name__ == "__main__":
    config = load_config()
    tickers = extract_all_tickers(config)
    # Add market indices
    tickers.extend(config["indices"].values())

    prices = download_prices(
        tickers=tickers,
        start_date=config["data_range"]["start_date"],
        end_date=config["data_range"]["end_date"],
    )
    returns = compute_returns(prices, method="log")

    save_to_processed(prices, "prices.csv")
    save_to_processed(returns, "returns.csv")

    print("\n✅ Data acquisition complete.")
    print(f"   Prices shape : {prices.shape}")
    print(f"   Returns shape: {returns.shape}")
