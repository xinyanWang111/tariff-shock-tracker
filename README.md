# 📈 Tech Stock Performance Analysis: US & Chinese Markets (2023–2024)

**ACC102 Individual Coursework — Track 3: Data Analysis Agent**  
**Author:** Yiying | **Institution:** Xi’an Jiaotong-Liverpool University  
**Data Source:** WRDS/CRSP | **Target Audience:** Retail Investors

-----

## 🔍 Project Overview

This project analyses the stock performance of **four major US tech companies** and **four Chinese tech ADRs** (American Depositary Receipts) listed on US exchanges, covering **January 2023 – December 2024**.

|Group      |Tickers                |Exchange     |
|-----------|-----------------------|-------------|
|US Tech    |AAPL, MSFT, GOOGL, TSLA|NASDAQ / NYSE|
|CN Tech ADR|BABA, BIDU, JD, NTES   |NYSE / NASDAQ|

**Research Questions:**

1. How did US and Chinese tech stocks perform in 2023–2024?
1. Which stocks outperformed or underperformed the S&P 500 benchmark?
1. What are the risk-return profiles for retail investors considering both markets?

-----

## 📊 Key Findings

|Metric                             |Best Performer|Value                    |
|-----------------------------------|--------------|-------------------------|
|Sharpe Ratio (risk-adjusted return)|**AAPL**      |1.4312                   |
|Lowest Volatility                  |**AAPL**      |0.2151 (annualised)      |
|Lowest Max Drawdown                |**MSFT**      |-15.49%                  |
|Highest Cumulative Return          |**TSLA**      |High return, highest risk|

**Investor Recommendations:**

- 🟢 **Conservative:** AAPL, MSFT (positive alpha + Sharpe > 1.0 + low drawdown)
- 🟡 **Balanced:** Mix of stable US tech + selective CN ADR exposure
- 🔴 **Aggressive:** TSLA, CN ADRs (higher potential upside, significantly higher risk)
- 📦 **Diversification:** Low US–CN cross-correlation suggests portfolio diversification benefit

-----

## 🗂️ Project Structure

```
ACC102-Tech-Stock-Analysis/
│
├── ACC102_Tech_Stock_Analysis.ipynb   # Main analysis notebook (Cells A–M)
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Files excluded from version control
├── reflection_report.md                # AI disclosure & methodology reflection
└── README.md                           # This file
```

-----

## 📈 Analysis Sections

|Section|Content                                                                              |
|-------|-------------------------------------------------------------------------------------|
|1      |Project Overview                                                                     |
|2      |Data Acquisition (WRDS/CRSP)                                                         |
|3      |Data Cleaning & Preparation                                                          |
|4      |US Tech Stock Analysis (price trends, volatility, cumulative return)                 |
|5      |Upgraded Metrics — Sharpe Ratio, Max Drawdown, Correlation, Box Plot, Monthly Heatmap|
|6      |Chinese Tech ADR Analysis                                                            |
|7      |Benchmark Comparison vs S&P 500 (Alpha Analysis)                                     |
|8      |US vs CN Cross-Market Comparison                                                     |
|9      |Key Findings & Investor Recommendations                                              |

-----

## 🗃️ Data Sources

|Dataset                                |Source              |Period   |
|---------------------------------------|--------------------|---------|
|US tech stock daily prices & returns   |WRDS/CRSP `crsp.dsf`|2023–2024|
|Chinese tech ADR daily prices & returns|WRDS/CRSP `crsp.dsf`|2023–2024|
|S&P 500 market index returns           |WRDS/CRSP `crsp.dsi`|2023–2024|


> ⚠️ **WRDS Access Required:** This notebook connects to the Wharton Research Data Services (WRDS) database. You will need a valid WRDS account to run the data acquisition cells. All data is used solely for educational purposes.

-----

## ⚙️ Setup & Usage

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ACC102-Tech-Stock-Analysis.git
cd ACC102-Tech-Stock-Analysis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

```bash
jupyter notebook ACC102_Tech_Stock_Analysis.ipynb
```

> Run cells **in order from top to bottom**. The notebook is structured so each section depends on variables defined in earlier cells.

-----

## 🤖 AI Disclosure

This project was completed as part of ACC102 coursework at XJTLU. AI tools (including Claude by Anthropic) were used to assist with:

- Code debugging and optimisation
- Visualisation design suggestions
- README and documentation drafting

All analytical decisions, data interpretation, and final write-up reflect the author’s own understanding. See `reflection_report.md` for full AI disclosure.

-----

## 📝 License

This project is for educational purposes only. Data accessed via WRDS is subject to WRDS Terms of Use.

-----

*ACC102 Business Analytics with Python | Xi’an Jiaotong-Liverpool University | April 2026*
