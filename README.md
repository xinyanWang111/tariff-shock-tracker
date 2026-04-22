# 📊 Tariff Shock Tracker

> Quantifying the market impact of the 2025 US tariff announcements on US–China tariff-sensitive sectors using event study methodology and CAPM-based abnormal returns.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/YOUR_USERNAME/tariff-shock-tracker/actions/workflows/tests.yml/badge.svg)](https://github.com/YOUR_USERNAME/tariff-shock-tracker/actions)

---

## 🎯 Project Overview

When the US government announced new tariffs on Chinese imports in 2025, markets reacted in minutes — but **by how much, for whom, and for how long?** This project answers that question quantitatively.

**Target audience**: retail investors, finance students, and policy analysts who want evidence-based, reproducible measurements of how policy shocks translate into stock price movements.

**Analytical problem**: Quantify the abnormal returns and volatility shifts experienced by tariff-sensitive US and Chinese listed firms (semiconductors, automotive, consumer electronics, retail) around the 2025 tariff announcement dates, and compare the magnitude and persistence of the market reaction across the two countries.

---

## 🔍 Key Features

- **Event study methodology** — computes Cumulative Abnormal Returns (CAR) around event dates using the market model
- **CAPM-based risk metrics** — beta, alpha, Sharpe ratio, and rolling volatility for every stock
- **Cross-country comparison** — side-by-side US vs. China sector reaction dashboard
- **Interactive Streamlit dashboard** — explore any stock, any event window, any sector
- **Reproducible pipeline** — one-command data refresh, fully tested code

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- Internet connection (for downloading market data via `yfinance`)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/tariff-shock-tracker.git
cd tariff-shock-tracker

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Analysis

```bash
# Option 1: Open the main notebook
jupyter notebook notebooks/04_final_analysis.ipynb

# Option 2: Launch the interactive dashboard
streamlit run dashboard/streamlit_app.py
```

---

## 📁 Repository Structure

```
tariff-shock-tracker/
├── README.md                    # You are here
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
├── config/
│   └── events.yaml              # Event dates and stock tickers
├── data/
│   ├── raw/                     # Raw data from yfinance
│   └── processed/               # Cleaned datasets
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_event_study.ipynb
│   ├── 03_risk_metrics.ipynb
│   └── 04_final_analysis.ipynb  # Main narrative notebook
├── src/
│   ├── data_loader.py           # Download & clean market data
│   ├── event_study.py           # CAR / AAR calculations
│   ├── risk_metrics.py          # Beta, Sharpe, CAPM
│   └── visualizations.py        # Chart generation
├── tests/                       # Unit tests (pytest)
├── dashboard/
│   └── streamlit_app.py         # Interactive dashboard
├── reports/
│   └── reflection_report.md     # 500–800 word reflection
└── .github/workflows/
    └── tests.yml                # CI pipeline
```

---

## 📊 Methodology

### 1. Event Study (Market Model)

For each stock _i_ and event date _t_, we estimate expected returns using the market model over an estimation window of 120 trading days ending 30 days before the event:

$$R_{i,t} = \alpha_i + \beta_i R_{m,t} + \varepsilon_{i,t}$$

Abnormal returns (AR) are then computed during the event window [-5, +10]:

$$AR_{i,t} = R_{i,t} - (\hat{\alpha}_i + \hat{\beta}_i R_{m,t})$$

Cumulative Abnormal Returns (CAR) are aggregated across the event window and tested for statistical significance using the standard _t_-test.

### 2. Risk Metrics

- **Beta** — sensitivity to market movements (CAPM)
- **Sharpe Ratio** — risk-adjusted return vs. 3-month US Treasury
- **Rolling 30-day Volatility** — annualized standard deviation of returns

### 3. Sector Comparison

Equal-weighted portfolios are built for each (country × sector) bucket, and their CAR series are compared statistically and visually.

---

## 📈 Key Findings

_(To be filled in after final analysis — see `notebooks/04_final_analysis.ipynb` for full results.)_

---

## 🗂️ Data Sources

| Dataset | Source | Access Date |
|---|---|---|
| US & Chinese stock prices | Yahoo Finance via `yfinance` | April 2026 |
| Market index (S&P 500, SSE Composite) | Yahoo Finance via `yfinance` | April 2026 |
| Risk-free rate (^IRX) | Yahoo Finance via `yfinance` | April 2026 |
| Tariff event dates | Compiled from public news sources | April 2026 |

All data used are publicly available and used solely for educational purposes.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📝 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙋 Author

**Yiying [Your Full Name]**  
BA Accounting, Year 2 · Xi'an Jiaotong-Liverpool University  
Course: ACC102 Business Analytics with Python  
Submission date: April 2026

---

## 📚 Acknowledgements

Built for the ACC102 Mini Assignment (Track 2 — GitHub Data Analysis Project). See `reports/reflection_report.md` for the full methodological reflection and AI disclosure.
