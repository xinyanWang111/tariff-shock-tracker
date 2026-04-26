# =============================================================
# Tariff Shock Tracker — Streamlit Dashboard
# Fixed: Using cumulative returns instead of raw prices
# =============================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import os

st.set_page_config(
    page_title="Tariff Shock Tracker",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data():
    base = os.path.expanduser(
        '~/Desktop/tariff-shock-tracker/data/raw')
    us_df     = pd.read_csv(
        f'{base}/us_stocks_crsp.csv',
        parse_dates=['date'])
    mkt_df    = pd.read_csv(
        f'{base}/us_market_index_crsp.csv',
        parse_dates=['date'])
    cn_df     = pd.read_csv(
        f'{base}/cn_stocks_csmar.csv',
        parse_dates=['date'])
    cn_mkt_df = pd.read_csv(
        f'{base}/cn_market_index_csmar.csv',
        parse_dates=['date'])
    cn_df['stkcd'] = (cn_df['stkcd']
                      .astype(str).str.zfill(6))
    return us_df, mkt_df, cn_df, cn_mkt_df

us_df, mkt_df, cn_df, cn_mkt_df = load_data()

EVENT_DATE = pd.Timestamp('2024-05-14')

cn_labels = {
    '000333': 'Midea 美的',
    '002475': 'Luxshare 立讯',
    '002594': 'BYD 比亚迪',
    '300750': 'CATL 宁德时代',
    '600519': 'Moutai 茅台'
}

us_colors = {
    'AAPL':'#1f77b4', 'NVDA':'#ff7f0e',
    'TSLA':'#e41a1c', 'GM':'#2ca02c',
    'WMT':'#9467bd'
}
cn_colors = {
    '300750':'#d62728', '002594':'#9467bd',
    '000333':'#8c564b', '002475':'#17becf',
    '600519':'#bcbd22'
}

# ---- Header ----
st.title("📊 Tariff Shock Tracker")
st.markdown("""
**Research Question**: How did the Biden administration's
**May 14, 2024 tariff announcement** (100% on Chinese EVs,
50% on solar panels, 25% on lithium batteries) affect
US and Chinese stocks?

*Data: CRSP & CSMAR via WRDS | 
Method: Event Study (Market Model)*
""")
st.divider()

# ---- Event Study Function ----
def run_event_study(stock_ret, mkt_ret,
                    event_date,
                    est_window=90,
                    est_gap=20,
                    pre=5, post=10):
    combined = pd.concat(
        [stock_ret, mkt_ret], axis=1
    ).dropna()
    combined.columns = ['s', 'm']
    dates = combined.index
    if event_date not in dates:
        future = dates[dates >= event_date]
        if len(future) == 0:
            return None
        event_date = future[0]
    ei  = dates.get_loc(event_date)
    es  = ei - pre - est_gap - est_window
    ee  = ei - pre - est_gap
    if es < 0:
        return None
    ed   = combined.iloc[es:ee]
    beta, alpha = np.polyfit(
        ed['m'].values, ed['s'].values, 1)
    sigma = np.std(
        ed['s'] - (alpha + beta * ed['m']),
        ddof=2)
    evd  = combined.iloc[ei-pre:ei+post+1]
    AR   = evd['s'] - (alpha + beta * evd['m'])
    AR.index = range(-pre, post+1)
    CAR  = AR.sum()
    t    = (CAR / (sigma * np.sqrt(len(AR)))
            if sigma > 0 else np.nan)
    p    = (2*(1-stats.t.cdf(abs(t), df=len(ed)-2))
            if not np.isnan(t) else np.nan)
    return {'AR':AR, 'CAR':CAR,
            'beta':beta, 't':t, 'p':p}

# Pre-compute event study results
us_mkt = mkt_df.set_index('date')['sprtrn']
cn_mkt = (cn_mkt_df[cn_mkt_df['indexcd']==300]
          .set_index('date')['retindex'].dropna())

@st.cache_data
def compute_all_results():
    us_res, cn_res = {}, {}
    for t in us_df['ticker'].unique():
        r = (us_df[us_df['ticker']==t]
             .set_index('date')['ret'].dropna())
        res = run_event_study(r, us_mkt, EVENT_DATE)
        if res:
            us_res[t] = res
    for s in cn_df['stkcd'].unique():
        r = (cn_df[cn_df['stkcd']==s]
             .set_index('date')['ret'].dropna())
        res = run_event_study(r, cn_mkt, EVENT_DATE)
        if res:
            cn_res[s] = res
    return us_res, cn_res

us_res, cn_res = compute_all_results()

# ============================================================
tab1, tab2, tab3 = st.tabs([
    "📈 Price Performance",
    "🔬 Event Study (CAR)",
    "⚖️ Risk Profile"
])

# ============================================================
# TAB 1: Price Performance
# ============================================================
with tab1:
    st.subheader(
        "Indexed Price Performance (Base = 100)")
    st.markdown(
        "Select stocks to compare performance "
        "around the tariff announcement.  \n"
        "*(Uses cumulative returns — "
        "automatically adjusts for stock splits)*")

    col1, col2 = st.columns(2)
    with col1:
        us_selected = st.multiselect(
            "🇺🇸 US Stocks",
            options=sorted(
                us_df['ticker'].unique()),
            default=['AAPL', 'NVDA', 'TSLA'])
    with col2:
        cn_selected = st.multiselect(
            "🇨🇳 Chinese Stocks",
            options=list(cn_labels.keys()),
            default=['300750', '002594'],
            format_func=lambda x: cn_labels[x])

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 5),
        facecolor='#FAFAFA')

    # US plot — cumulative returns
    ax1 = axes[0]
    for ticker in (us_selected or ['AAPL']):
        grp = (us_df[us_df['ticker']==ticker]
               .sort_values('date')
               .dropna(subset=['ret']).copy())
        
        grp['idx'] = (
            (1 + grp['ret']).cumprod() * 100)
        ax1.plot(grp['date'], grp['idx'],
                 label=ticker,
                 color=us_colors.get(
                     ticker, 'gray'),
                 linewidth=2)
    ax1.axvline(EVENT_DATE, color='red',
            linestyle='--', linewidth=1.5)
    ax1.text(EVENT_DATE + pd.Timedelta(days=3),
         ax1.get_ylim()[1] * 0.95,
         'Tariff\n2024-05-14',
         fontsize=8, color='red',
         fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.2',
                   facecolor='lightyellow',
                   edgecolor='red', alpha=0.8))

    ax1.axhline(100, color='gray',
                linestyle=':', alpha=0.5)
    ax1.set_title('🇺🇸 US Stocks',
                  fontweight='bold')
    ax1.set_ylabel(
        'Cumulative Return Index (Base=100)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(
        mdates.DateFormatter('%b %Y'))
    plt.setp(
        ax1.xaxis.get_majorticklabels(),
        rotation=30)

    # CN plot — cumulative returns
    ax2 = axes[1]
    for stkcd in (cn_selected or ['300750']):
        grp = (cn_df[cn_df['stkcd']==stkcd]
               .sort_values('date')
               .dropna(subset=['ret']).copy())
        if len(grp) == 0:
            continue
        # ✅ 用累积收益率
        grp['idx'] = (
            (1 + grp['ret']).cumprod() * 100)
        ax2.plot(grp['date'], grp['idx'],
                 label=cn_labels.get(
                     stkcd, stkcd),
                 color=cn_colors.get(
                     stkcd, 'gray'),
                 linewidth=2)
    ax2.axvline(EVENT_DATE, color='red',
            linestyle='--', linewidth=1.5)
    ax2.text(EVENT_DATE + pd.Timedelta(days=3),
         ax2.get_ylim()[1] * 0.95,
         'Tariff\n2024-05-14',
         fontsize=8, color='red',
         fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.2',
                   facecolor='lightyellow',
                   edgecolor='red', alpha=0.8))

    ax2.axhline(100, color='gray',
                linestyle=':', alpha=0.5)
    ax2.set_title('🇨🇳 Chinese Stocks',
                  fontweight='bold')
    ax2.set_ylabel(
        'Cumulative Return Index (Base=100)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(
        mdates.DateFormatter('%b %Y'))
    plt.setp(
        ax2.xaxis.get_majorticklabels(),
        rotation=30)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Key stats
    st.subheader("📊 Key Statistics")
    col1, col2, col3, col4 = st.columns(4)
    nvda_total = (
        (1 + us_df[us_df['ticker']=='NVDA']
         ['ret'].dropna()).prod() - 1) * 100
    byd_total = (
        (1 + cn_df[cn_df['stkcd']=='002594']
         ['ret'].dropna()).prod() - 1) * 100
    catl_total = (
        (1 + cn_df[cn_df['stkcd']=='300750']
         ['ret'].dropna()).prod() - 1) * 100
    gm_total = (
        (1 + us_df[us_df['ticker']=='GM']
         ['ret'].dropna()).prod() - 1) * 100

    col1.metric("NVDA Total Return",
                f"{nvda_total:+.1f}%",
                "Best US performer")
    col2.metric("GM Total Return",
                f"{gm_total:+.1f}%",
                "Worst US performer")
    col3.metric("CATL Total Return",
                f"{catl_total:+.1f}%",
                "Direct tariff target")
    col4.metric("BYD Total Return",
                f"{byd_total:+.1f}%",
                "EV tariff target")

# ============================================================
# TAB 2: Event Study (CAR)
# ============================================================
with tab2:
    st.subheader(
        "Cumulative Abnormal Returns (CAR)")
    st.markdown("""
CAR measures how much each stock **deviated from its 
expected return** around the tariff announcement.

> **AR** = Actual Return − (α + β × Market Return)  
> **CAR** = Cumulative AR over [−5, +10] trading days  
> Solid line = significant (p<0.05) | 
> Dashed = not significant
""")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5),
        facecolor='#FAFAFA')
    ew = range(-5, 11)

    for ticker, r in us_res.items():
        ls = '-' if r['p'] < 0.05 else '--'
        ax1.plot(
            list(ew),
            (r['AR'].cumsum()*100).values,
            label=(f"{ticker} "
                   f"({r['CAR']*100:+.1f}%)"),
            linewidth=2,
            color=us_colors.get(ticker,'gray'),
            linestyle=ls)
    ax1.axvline(0, color='red',
                linestyle='--', linewidth=1.5,
                label='Event Day')
    ax1.axhline(0, color='black',
                linestyle=':', alpha=0.4)
    ax1.fill_betweenx(
        [-25, 25], -5, 0,
        alpha=0.05, color='gray')
    ax1.set_title('🇺🇸 US Stocks CAR',
                  fontweight='bold')
    ax1.set_xlabel(
        'Days Relative to Event')
    ax1.set_ylabel('CAR (%)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-5, 10)
    ax1.set_facecolor('#F8F9FA')

    cn_col2 = {
        '000333':'#8c564b',
        '002475':'#17becf',
        '002594':'#9467bd',
        '300750':'#d62728',
        '600519':'#bcbd22'}
    for s, r in cn_res.items():
        ls = '-' if r['p'] < 0.05 else '--'
        ax2.plot(
            list(ew),
            (r['AR'].cumsum()*100).values,
            label=(f"{cn_labels.get(s,s)} "
                   f"({r['CAR']*100:+.1f}%)"),
            linewidth=2,
            color=cn_col2.get(s,'gray'),
            linestyle=ls)
    ax2.axvline(0, color='red',
                linestyle='--', linewidth=1.5,
                label='Event Day')
    ax2.axhline(0, color='black',
                linestyle=':', alpha=0.4)
    ax2.fill_betweenx(
        [-25, 25], -5, 0,
        alpha=0.05, color='gray')
    ax2.set_title('🇨🇳 Chinese Stocks CAR',
                  fontweight='bold')
    ax2.set_xlabel(
        'Days Relative to Event')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-5, 10)
    ax2.set_facecolor('#F8F9FA')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Results table
    st.subheader("📋 Full Results Table")
    rows = []
    for t, r in us_res.items():
        rows.append({
            'Country': '🇺🇸 US',
            'Stock': t,
            'Beta': f"{r['beta']:.3f}",
            'CAR (%)': f"{r['CAR']*100:+.2f}%",
            't-stat': f"{r['t']:+.3f}",
            'p-value': f"{r['p']:.4f}",
            'Significant': (
                '✅ Yes' if r['p'] < 0.05
                else '❌ No')
        })
    for s, r in cn_res.items():
        rows.append({
            'Country': '🇨🇳 China',
            'Stock': cn_labels.get(s, s),
            'Beta': f"{r['beta']:.3f}",
            'CAR (%)': f"{r['CAR']*100:+.2f}%",
            't-stat': f"{r['t']:+.3f}",
            'p-value': f"{r['p']:.4f}",
            'Significant': (
                '✅ Yes' if r['p'] < 0.05
                else '❌ No')
        })
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True)

    # Insight box
    st.info("""
**💡 Key Insight**: No stock achieved statistical 
significance (p<0.05). This suggests either 
(a) the tariff was partially anticipated by markets, 
or (b) concurrent events during the [-5, +10] window 
diluted the tariff signal. Despite insignificance, 
the directional pattern is clear: US tech stocks 
gained while Chinese EV/battery stocks declined.
""")

# ============================================================
# TAB 3: Risk Profile
# ============================================================
with tab3:
    st.subheader(
        "Risk Metrics: Beta, Sharpe & Volatility")
    st.markdown(
        "Comparing risk profiles of US and "
        "Chinese stocks around the tariff event.")

    def get_metrics(ret, mkt, rf=0.0001):
        r = ret.dropna()
        m = mkt.reindex(r.index).dropna()
        r = r.reindex(m.index)
        if len(r) < 30:
            return None
        beta = (np.cov(r, m)[0,1]
                / np.var(m))
        sharpe = (((r.mean()-rf)
                   / r.std())
                  * np.sqrt(252))
        vol = r.std() * np.sqrt(252) * 100
        pre  = r[r.index <  EVENT_DATE]
        post = r[r.index >= EVENT_DATE]
        v_pre  = (pre.std()
                  * np.sqrt(252) * 100
                  if len(pre) > 5 else np.nan)
        v_post = (post.std()
                  * np.sqrt(252) * 100
                  if len(post) > 5 else np.nan)
        d_vol = ((v_post - v_pre) / v_pre * 100
                 if v_pre else np.nan)
        return {
            'Beta': round(beta, 3),
            'Sharpe': round(sharpe, 3),
            'Vol (%)': round(vol, 1),
            'Pre-Vol (%)': round(v_pre, 1),
            'Post-Vol (%)': round(v_post, 1),
            'ΔVol (%)': round(d_vol, 1)
        }

    rows = []
    for t in sorted(us_df['ticker'].unique()):
        r = (us_df[us_df['ticker']==t]
             .set_index('date')['ret'])
        m = get_metrics(r, us_mkt)
        if m:
            rows.append({
                'Country': '🇺🇸',
                'Stock': t, **m})
    for s in sorted(cn_df['stkcd'].unique()):
        r = (cn_df[cn_df['stkcd']==s]
             .set_index('date')['ret'])
        m = get_metrics(r, cn_mkt)
        if m:
            rows.append({
                'Country': '🇨🇳',
                'Stock': cn_labels.get(s,s),
                **m})

    risk_df = pd.DataFrame(rows)
    st.dataframe(
        risk_df,
        use_container_width=True)

    # Highlight cards
    st.subheader("🏆 Highlights")
    col1, col2, col3 = st.columns(3)

    max_b = risk_df.loc[
        risk_df['Beta'].idxmax()]
    max_s = risk_df.loc[
        risk_df['Sharpe'].idxmax()]
    max_v = risk_df.loc[
        risk_df['ΔVol (%)'].idxmax()]

    col1.metric(
        "Highest Beta (Most Sensitive)",
        max_b['Stock'],
        f"β = {max_b['Beta']}")
    col2.metric(
        "Best Risk-Adjusted Return",
        max_s['Stock'],
        f"Sharpe = {max_s['Sharpe']}")
    col3.metric(
        "Biggest Volatility Jump",
        max_v['Stock'],
        f"ΔVol = +{max_v['ΔVol (%)']:.1f}%")

    # Beta vs Sharpe scatter
    st.subheader(
        "📊 Risk-Return Profile")
    fig, ax = plt.subplots(
        figsize=(10, 6),
        facecolor='#FAFAFA')
    all_colors = (
        [us_colors.get(t,'gray')
         for t in sorted(
             us_df['ticker'].unique())] +
        [cn_colors.get(s,'gray')
         for s in sorted(
             cn_df['stkcd'].unique())])
    for i, row in risk_df.iterrows():
        ax.scatter(
            row['Beta'], row['Sharpe'],
            color=all_colors[i],
            s=150, zorder=5, alpha=0.85)
        ax.annotate(
            row['Stock'],
            (row['Beta'], row['Sharpe']),
            textcoords='offset points',
            xytext=(8, 4), fontsize=8)
    ax.axhline(0, color='black',
               linestyle='--', alpha=0.3)
    ax.axvline(1, color='red',
               linestyle='--', alpha=0.3,
               label='β=1 (Market)')
    ax.set_xlabel(
        'Beta (Market Sensitivity)',
        fontsize=10)
    ax.set_ylabel(
        'Sharpe Ratio', fontsize=10)
    ax.set_title(
        'Risk-Return Profile: '
        'Beta vs Sharpe Ratio',
        fontweight='bold', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#F8F9FA')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ---- Footer ----
st.divider()
st.markdown("""
<small>
**Data Sources**: CRSP (US stocks & S&P 500) and 
CSMAR (Chinese stocks & CSI 300) via WRDS, 
accessed April 2026.  
**Methodology**: Market Model Event Study, 
Estimation Window: 90 trading days.  
**Note**: Price performance chart uses cumulative 
returns to adjust for corporate actions 
(e.g. NVIDIA 10:1 stock split on June 10, 2024).  
**Author**: Xinyan.Wang | ACC102 Mini Assignment, 
XJTLU, 2026
</small>
""", unsafe_allow_html=True)
