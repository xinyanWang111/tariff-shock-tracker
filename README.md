# 📊 Tariff Shock Tracker

> Quantifying the equity market impact of the Biden administration’s
> May 14, 2024 tariff announcement on US and Chinese stocks using
> event study methodology and CAPM-based risk metrics.

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Data: WRDS](https://img.shields.io/badge/Data-WRDS%20%7C%20CRSP%20%7C%20CSMAR-green.svg)](https://wrds-www.wharton.upenn.edu/)

-----

## 🎯 Research Question

When the Biden administration announced sweeping tariff increases
on Chinese imports on **May 14, 2024** — including 100% on electric
vehicles, 50% on solar panels, and 25% on lithium batteries — how
did US and Chinese stocks in affected sectors respond?

**Target audience**: Retail investors and finance students seeking
evidence-based, quantitative analysis of how trade policy
announcements translate into measurable equity market movements.

-----

## 🔍 Key Findings

|Stock         |Country|Sector              |CAR [-5,+10]|Significant?|
|--------------|-------|--------------------|------------|------------|
|NVIDIA        |🇺🇸 US   |Semiconductors      |+14.33%     |❌ No        |
|Walmart       |🇺🇸 US   |Retail              |+7.05%      |❌ No        |
|Apple         |🇺🇸 US   |Consumer Electronics|+5.53%      |❌ No        |
|Tesla         |🇺🇸 US   |Automotive          |+2.24%      |❌ No        |
|BYD 比亚迪       |🇨🇳 China|Electric Vehicles   |-4.60%      |❌ No        |
|CATL 宁德时代     |🇨🇳 China|Batteries           |-4.04%      |❌ No        |
|Midea 美的      |🇨🇳 China|Appliances          |-5.46%      |❌ No        |
|General Motors|🇺🇸 US   |Automotive          |-8.57%      |❌ No        |

**Key insight**: US tech stocks broadly gained while Chinese
EV/battery stocks declined — consistent with markets pricing
the tariff as a competitive advantage for US firms and a
headwind for Chinese counterparts. No result achieved
statistical significance at p<0.05, suggesting partial
anticipation or concurrent confounding events.

-----

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- WRDS account with CRSP and CSMAR access
- Internet connection

### Installation

```bash
# Clone the repository
git clone https://github.com/xinyanWang111/tariff-shock-tracker.git
cd tariff-shock-tracker

# Install dependencies
pip install -r requirements.txt
```

### Run the Analysis

```bash
# Step 1: Download data (requires WRDS credentials)
jupyter notebook notebooks/01_data_acquisition.ipynb

# Step 2: Run event study
jupyter notebook notebooks/02_event_study.ipynb

# Step 3: Compute risk metrics
jupyter notebook notebooks/03_risk_metrics.ipynb

# Step 4: Launch interactive dashboard
streamlit run dashboard/streamlit_app.py
```

-----

## 📁 Repository Structure

```
tariff-shock-tracker/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── LICENSE                          # MIT License
├── config/
│   └── events.yaml                  # Event dates & stock universe
├── data/
│   └── raw/                         # CRSP & CSMAR data (CSV)
│       ├── us_stocks_crsp.csv
│       ├── us_market_index_crsp.csv
│       ├── cn_stocks_csmar.csv
│       └── cn_market_index_csmar.csv
├── notebooks/
│   ├── 01_data_acquisition.ipynb    # Data download & EDA
│   ├── 02_event_study.ipynb         # CAR analysis
│   └── 03_risk_metrics.ipynb        # Beta, Sharpe, volatility
├── src/
│   ├── data_loader.py               # WRDS query utilities
│   ├── event_study.py               # Market model functions
│   └── risk_metrics.py              # Risk calculation functions
├── tests/
│   ├── test_event_study.py          # Unit tests
│   └── test_risk_metrics.py         # Unit tests
├── dashboard/
│   └── streamlit_app.py             # Interactive Streamlit app
├── assets/
│   └── screenshots/                 # Chart outputs
└── reports/
    └── reflection_report.md         # 500-800 word reflection
```

-----

## 📊 Methodology

### 1. Event Study (Market Model)

Following MacKinlay (1997), for each stock *i* and event date *t*:

**Step 1** — Estimate market model over 90-day estimation window:

$$R_{i,t} = \alpha_i + \beta_i R_{m,t} + \varepsilon_{i,t}$$

**Step 2** — Compute Abnormal Returns (AR) over [-5, +10] window:

$$AR_{i,t} = R_{i,t} - (\hat{\alpha}_i + \hat{\beta}*i R*{m,t})$$

**Step 3** — Cumulate to get CAR and test for significance:

$$CAR_i = \sum_{t=-5}^{+10} AR_{i,t}$$

### 2. Risk Metrics

- **Beta** — CAPM market sensitivity: Cov(Ri, Rm) / Var(Rm)
- **Sharpe Ratio** — Annualised risk-adjusted return
- **Volatility** — Annualised standard deviation, pre vs post event

### 3. Stock Universe

**US stocks** (CRSP permno identifiers):

|Ticker|permno|Sector              |
|------|------|--------------------|
|AAPL  |14593 |Consumer Electronics|
|NVDA  |86580 |Semiconductors      |
|TSLA  |93436 |Automotive          |
|GM    |12369 |Automotive          |
|WMT   |55976 |Retail              |

**Chinese stocks** (CSMAR stkcd identifiers):

|Code  |Company           |Sector                     |
|------|------------------|---------------------------|
|300750|CATL 宁德时代         |Batteries                  |
|002594|BYD 比亚迪           |Electric Vehicles          |
|000333|Midea 美的          |Appliances                 |
|002475|Luxshare 立讯       |Supply Chain               |
|600519|Kweichow Moutai 茅台|Domestic Consumer (Control)|


> **Methodological note**: Kweichow Moutai serves as a **control
> stock** — as a domestically-focused company with negligible export
> exposure, it should show minimal abnormal returns if the methodology
> correctly isolates the tariff effect.

-----

## 🗂️ Data Sources

|Dataset                  |Source        |Table            |Coverage           |
|-------------------------|--------------|-----------------|-------------------|
|US stock prices & returns|CRSP via WRDS |`crsp.dsf`       |Oct 2023 – Aug 2024|
|US market index          |CRSP via WRDS |`crsp.dsi`       |Oct 2023 – Aug 2024|
|Chinese stock returns    |CSMAR via WRDS|`csmar.trd_dalyr`|Oct 2023 – Aug 2024|
|Chinese market index     |CSMAR via WRDS|`csmar.trd_index`|Oct 2023 – Aug 2024|

All data accessed via WRDS institutional subscription
(Xi’an Jiaotong-Liverpool University), April 2026.

**Why WRDS over free alternatives (e.g. yfinance)?**

- CRSP is survivor-bias-free with full dividend adjustment
- CSMAR is the authoritative source for Chinese equity data
- Both are used in top-tier academic journals (JF, RFS, JFE)
- Institutional access ensures data reliability and reproducibility

-----

## 🧪 Running Tests

```bash
pytest tests/ -v
```

-----

## ⚠️ Technical Notes

**NVIDIA Stock Split**: NVIDIA completed a 10:1 stock split on
June 10, 2024. All price performance charts use **cumulative
returns** rather than raw prices to correctly handle this
corporate action.

**Chinese Trading Calendar**: China has fewer trading days per
year than the US due to national holidays (Chinese New Year,
National Day etc.). The estimation window for Chinese stocks was
shortened to 90 days (vs 120 for US stocks) to ensure sufficient
pre-event data.

**CRSP Data Coverage**: CRSP data covers through December 2024.
The 2025 Liberation Day tariff events fall outside this window
and represent a direction for future research.

-----

## 📈 Interactive Dashboard

Launch the Streamlit dashboard for interactive exploration:

```bash
streamlit run dashboard/streamlit_app.py
```

**Features:**

- 📈 Interactive stock selector for price performance comparison
- 🔬 CAR time series with statistical significance indicators
- ⚖️ Risk profile comparison (Beta, Sharpe Ratio, volatility)
- 📊 Pre vs post-event volatility analysis
- 📋 Full results table with t-statistics and p-values

-----

## 📝 License

This project is released under the MIT License.
See <LICENSE> for details.

-----

## 👤 Author

**Xinyan.Wang** (xinyanWang111)  
Module: ACC102 Business Analytics with Python  
Submission: April 2026

-----

## 📚 References

- MacKinlay, A.C. (1997). Event Studies in Economics and Finance.
  *Journal of Economic Literature*, 35(1), 13–39.
- Sharpe, W.F. (1994). The Sharpe Ratio.
  *Journal of Portfolio Management*, 21(1), 49–58.
- Center for Research in Security Prices (CRSP),
  University of Chicago Booth School of Business.
- China Stock Market & Accounting Research (CSMAR) Database.
