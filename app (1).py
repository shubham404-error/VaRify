"""
Value at Risk (VaR) Calculator
Portfolio Risk Analytics Tool | Built with Streamlit + yfinance
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="VaR Calculator",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main layout */
    .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1rem;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 700;
    }

    /* Section headers */
    .section-header {
        font-size: 1rem;
        font-weight: 600;
        color: rgba(255,255,255,0.75);
        border-bottom: 2px solid rgba(255,255,255,0.12);
        padding-bottom: 6px;
        margin-bottom: 1rem;
        margin-top: 1.5rem;
    }

    /* Info box */
    .info-box {
        background: #e8f4fd;
        border-left: 4px solid #1f77b4;
        border-radius: 4px;
        padding: 0.75rem 1rem;
        font-size: 0.875rem;
        margin: 0.5rem 0;
        color: #1a1a2e !important;
    }
    .info-box * { color: #1a1a2e !important; }

    /* Warning box */
    .warn-box {
        background: #fff8e1;
        border-left: 4px solid #ff9800;
        border-radius: 4px;
        padding: 0.75rem 1rem;
        font-size: 0.875rem;
        margin: 0.5rem 0;
        color: #1a1a2e !important;
    }
    .warn-box * { color: #1a1a2e !important; }

    /* Tag badges */
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
    }
    .badge-green  { background: #d4edda; color: #155724; }
    .badge-red    { background: #f8d7da; color: #721c24; }
    .badge-yellow { background: #fff3cd; color: #856404; }
    .badge-blue   { background: #d1ecf1; color: #0c5460; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #1a1a2e; }
    [data-testid="stSidebar"] * { color: #e8e8e8 !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stSlider label { color: #adb5bd !important; font-size: 0.85rem; }

    /* Footer */
    .footer { text-align: center; color: #adb5bd; font-size: 0.8rem; margin-top: 3rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_stock_list(country: str) -> pd.DataFrame:
    base = os.path.dirname(__file__)
    if country == "India (NSE)":
        path = os.path.join(base, "data", "nifty500.csv")
    else:
        path = os.path.join(base, "data", "sp500.csv")
    return pd.read_csv(path)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_prices(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices for given tickers."""
    data = yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data[["Close"]] if "Close" in data.columns else data
    close = close.dropna(how="all")
    return close


# ─────────────────────────────────────────────
# VaR ENGINE
# ─────────────────────────────────────────────
def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns."""
    return np.log(prices / prices.shift(1)).dropna()


def parametric_var(returns: pd.Series, conf: float, horizon: int) -> dict:
    mu = returns.mean()
    sigma = returns.std()
    z = stats.norm.ppf(1 - conf)
    # Scale to horizon
    mu_h = mu * horizon
    sigma_h = sigma * np.sqrt(horizon)
    var_pct = -(mu_h + z * sigma_h)
    # CVaR (Expected Shortfall) for normal distribution
    cvar_pct = -(mu_h - sigma_h * stats.norm.pdf(z) / (1 - conf))
    return {
        "var_pct": var_pct,
        "cvar_pct": cvar_pct,
        "mu": mu,
        "sigma": sigma,
        "mu_h": mu_h,
        "sigma_h": sigma_h,
        "z": z,
    }


def historical_var(returns: pd.Series, conf: float, horizon: int) -> dict:
    # Scale single-day returns to horizon via square root of time
    scaled = returns * np.sqrt(horizon)
    var_pct = -np.percentile(scaled, (1 - conf) * 100)
    tail = scaled[scaled <= -var_pct]
    cvar_pct = -tail.mean() if len(tail) > 0 else var_pct
    return {"var_pct": var_pct, "cvar_pct": cvar_pct}


def monte_carlo_var(returns: pd.Series, conf: float, horizon: int, n_sim: int = 10000) -> dict:
    mu = returns.mean()
    sigma = returns.std()
    # Simulate n_sim paths of `horizon` daily returns, sum = cumulative log return
    rng = np.random.default_rng(42)
    sim_daily = rng.normal(mu, sigma, size=(n_sim, horizon))
    sim_cumulative = sim_daily.sum(axis=1)  # log return over horizon
    var_pct = -np.percentile(sim_cumulative, (1 - conf) * 100)
    tail = sim_cumulative[sim_cumulative <= -var_pct]
    cvar_pct = -tail.mean() if len(tail) > 0 else var_pct
    return {"var_pct": var_pct, "cvar_pct": cvar_pct, "sim": sim_cumulative}


def portfolio_var(returns: pd.DataFrame, weights: np.ndarray, conf: float, horizon: int) -> dict:
    """Portfolio VaR using correlation matrix."""
    mu_vec = returns.mean().values
    cov = returns.cov().values
    port_mu = weights @ mu_vec
    port_var_daily = weights @ cov @ weights
    port_sigma = np.sqrt(port_var_daily)
    # Scale to horizon
    port_mu_h = port_mu * horizon
    port_sigma_h = port_sigma * np.sqrt(horizon)
    z = stats.norm.ppf(1 - conf)
    var_pct = -(port_mu_h + z * port_sigma_h)
    cvar_pct = -(port_mu_h - port_sigma_h * stats.norm.pdf(z) / (1 - conf))
    # Naive VaR (no correlation) for diversification benefit
    individual_vars = []
    for i, col in enumerate(returns.columns):
        r = returns[col]
        v = -(r.mean() * horizon + z * r.std() * np.sqrt(horizon))
        individual_vars.append(weights[i] * v)
    naive_var = sum(individual_vars)
    div_benefit = naive_var - var_pct
    return {
        "var_pct": var_pct,
        "cvar_pct": cvar_pct,
        "port_mu": port_mu,
        "port_sigma": port_sigma,
        "port_mu_h": port_mu_h,
        "port_sigma_h": port_sigma_h,
        "naive_var_pct": naive_var,
        "div_benefit_pct": div_benefit,
        "z": z,
    }


def distribution_stats(returns: pd.Series) -> dict:
    """Skewness, kurtosis, normality test."""
    skew = float(returns.skew())
    kurt = float(returns.kurtosis())  # excess kurtosis (0 = normal)
    jb_stat, jb_p = stats.jarque_bera(returns.dropna())
    return {
        "skewness": skew,
        "excess_kurtosis": kurt,
        "jb_stat": jb_stat,
        "jb_pvalue": jb_p,
        "is_normal": jb_p > 0.05,
    }


def backtest_var(returns: pd.Series, var_pct: float, conf: float) -> dict:
    """Count breaches and compute Kupiec POF test."""
    breaches = (returns < -var_pct).sum()
    total = len(returns)
    breach_rate = breaches / total
    expected_rate = 1 - conf
    # Kupiec likelihood ratio test
    if breach_rate == 0 or breach_rate == 1:
        pof_pvalue = np.nan
    else:
        lr = -2 * (
            total * np.log(1 - expected_rate)
            + breaches * np.log(expected_rate)
            - total * np.log(1 - breach_rate)
            - breaches * np.log(breach_rate)
        )
        pof_pvalue = 1 - stats.chi2.cdf(lr, df=1)
    return {
        "breaches": int(breaches),
        "total": total,
        "breach_rate": breach_rate,
        "expected_rate": expected_rate,
        "pof_pvalue": pof_pvalue,
    }


# ─────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────
COLORS = {
    "primary":   "#1f77b4",
    "danger":    "#d62728",
    "warning":   "#ff7f0e",
    "success":   "#2ca02c",
    "muted":     "#adb5bd",
    "tail_fill": "rgba(214,39,40,0.18)",
    "dist_fill": "rgba(31,119,180,0.12)",
}


def plot_distribution(returns: pd.Series, var_pct: float, cvar_pct: float,
                       conf: float, method: str, ticker_label: str) -> go.Figure:
    """Normal distribution + histogram with VaR cutoff."""
    r = returns.values
    mu, sigma = r.mean(), r.std()
    x_min, x_max = mu - 4.5 * sigma, mu + 4.5 * sigma
    x_range = np.linspace(x_min, x_max, 400)
    pdf_vals = stats.norm.pdf(x_range, mu, sigma)
    cutoff = -var_pct

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Histogram of actual returns (secondary y)
    fig.add_trace(go.Histogram(
        x=r,
        nbinsx=60,
        name="Actual returns",
        marker_color=COLORS["primary"],
        opacity=0.35,
        showlegend=True,
    ), secondary_y=True)

    # Normal PDF curve
    fig.add_trace(go.Scatter(
        x=x_range, y=pdf_vals,
        mode="lines",
        name="Normal fit",
        line=dict(color=COLORS["primary"], width=2.5),
        fill="tozeroy",
        fillcolor=COLORS["dist_fill"],
    ), secondary_y=False)

    # Shaded tail
    x_tail = x_range[x_range <= cutoff]
    y_tail = stats.norm.pdf(x_tail, mu, sigma)
    fig.add_trace(go.Scatter(
        x=np.concatenate([[x_tail[0]], x_tail, [x_tail[-1]]]),
        y=np.concatenate([[0], y_tail, [0]]),
        fill="toself",
        fillcolor=COLORS["tail_fill"],
        line=dict(width=0),
        name=f"Loss tail ({(1-conf)*100:.0f}%)",
        showlegend=True,
    ), secondary_y=False)

    # VaR cutoff line
    fig.add_vline(
        x=cutoff,
        line_dash="dash",
        line_color=COLORS["danger"],
        line_width=2,
        annotation_text=f"  VaR cutoff<br>  {cutoff*100:.2f}%",
        annotation_font=dict(color=COLORS["danger"], size=12),
        annotation_position="top right",
    )

    # CVaR line
    fig.add_vline(
        x=-cvar_pct,
        line_dash="dot",
        line_color=COLORS["warning"],
        line_width=1.5,
        annotation_text=f"  CVaR<br>  {-cvar_pct*100:.2f}%",
        annotation_font=dict(color=COLORS["warning"], size=11),
        annotation_position="bottom right",
    )

    fig.update_layout(
        title=dict(text=f"Return distribution — {ticker_label} ({method})", font_size=14),
        xaxis_title="Daily log return",
        yaxis_title="Probability density",
        yaxis2_title="Frequency",
        legend=dict(orientation="h", y=1.08),
        height=420,
        margin=dict(t=70, b=40, l=50, r=30),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(tickformat=".1%", gridcolor="#f0f0f0"),
        yaxis=dict(gridcolor="#f0f0f0"),
    )
    # Format x-axis as percentage
    fig.update_xaxes(tickformat=".2%")
    return fig


def plot_price_with_var(prices: pd.Series, returns: pd.Series,
                         var_pct: float, position_size: float,
                         ticker: str) -> go.Figure:
    """Price chart with rolling VaR bands (30-day window)."""
    roll = returns.rolling(30)
    roll_mu = roll.mean()
    roll_sigma = roll.std()
    z = stats.norm.ppf(0.05)  # 95% conf implied
    roll_var = -(roll_mu + z * roll_sigma)

    # Align with prices index
    aligned = roll_var.reindex(prices.index).bfill()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=prices.index, y=prices,
        mode="lines",
        name=ticker,
        line=dict(color=COLORS["primary"], width=2),
    ))

    # Rolling VaR floor
    var_floor = prices * (1 - aligned)
    fig.add_trace(go.Scatter(
        x=prices.index, y=var_floor,
        mode="lines",
        name="Rolling VaR floor (95%)",
        line=dict(color=COLORS["danger"], width=1.5, dash="dash"),
        fill=None,
    ))

    fig.add_trace(go.Scatter(
        x=prices.index, y=prices,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        fill="tonexty",
        fillcolor="rgba(214,39,40,0.07)",
    ))

    fig.update_layout(
        title=dict(text=f"{ticker} — Price with rolling VaR floor", font_size=14),
        xaxis_title="Date",
        yaxis_title="Price",
        height=360,
        margin=dict(t=60, b=40, l=60, r=30),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(gridcolor="#f0f0f0"),
        yaxis=dict(gridcolor="#f0f0f0"),
        legend=dict(orientation="h", y=1.08),
    )
    return fig


def plot_correlation_heatmap(returns: pd.DataFrame) -> go.Figure:
    corr = returns.corr()
    labels = corr.columns.tolist()
    z = corr.values.round(3)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        colorscale=[
            [0.0, "#d62728"],
            [0.5, "#ffffff"],
            [1.0, "#1f77b4"],
        ],
        zmin=-1, zmax=1,
        text=z,
        texttemplate="%{text}",
        textfont=dict(size=12),
        showscale=True,
        colorbar=dict(title="ρ", len=0.8),
    ))

    fig.update_layout(
        title=dict(text="Correlation matrix (daily log returns)", font_size=14),
        height=max(300, 80 * len(labels) + 80),
        margin=dict(t=60, b=40, l=80, r=40),
        xaxis=dict(side="bottom"),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def plot_mc_histogram(sim: np.ndarray, var_pct: float, cvar_pct: float,
                       conf: float) -> go.Figure:
    cutoff = -var_pct
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=sim,
        nbinsx=80,
        marker_color=COLORS["primary"],
        opacity=0.6,
        name="Simulated returns",
    ))
    fig.add_vline(x=cutoff, line_color=COLORS["danger"], line_dash="dash",
                  line_width=2,
                  annotation_text=f"  VaR {cutoff*100:.2f}%",
                  annotation_font_color=COLORS["danger"])
    fig.add_vline(x=-cvar_pct, line_color=COLORS["warning"], line_dash="dot",
                  line_width=1.5,
                  annotation_text=f"  CVaR {-cvar_pct*100:.2f}%",
                  annotation_font_color=COLORS["warning"])
    fig.update_layout(
        title=dict(text=f"Monte Carlo simulation — 10,000 paths ({conf*100:.0f}% conf.)", font_size=14),
        xaxis_title="Simulated cumulative return",
        yaxis_title="Frequency",
        height=380,
        margin=dict(t=60, b=40, l=60, r=30),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(tickformat=".2%", gridcolor="#f0f0f0"),
        yaxis=dict(gridcolor="#f0f0f0"),
    )
    return fig


def plot_backtest(returns: pd.Series, var_pct: float, ticker: str) -> go.Figure:
    cutoff = -var_pct
    colors = ["red" if r < cutoff else COLORS["primary"] for r in returns]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=returns.index,
        y=returns.values,
        marker_color=colors,
        name="Daily return",
        showlegend=False,
    ))
    fig.add_hline(y=cutoff, line_color=COLORS["danger"], line_dash="dash",
                  line_width=2,
                  annotation_text=f"  VaR threshold {cutoff*100:.2f}%",
                  annotation_font_color=COLORS["danger"])
    fig.update_layout(
        title=dict(text=f"{ticker} — VaR breach backtesting", font_size=14),
        xaxis_title="Date",
        yaxis_title="Daily log return",
        height=340,
        margin=dict(t=60, b=40, l=60, r=30),
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(tickformat=".2%", gridcolor="#f0f0f0"),
        xaxis=dict(gridcolor="#f0f0f0"),
    )
    return fig


def plot_weights_pie(weights: list, labels: list) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=labels,
        values=weights,
        hole=0.5,
        textinfo="label+percent",
        marker=dict(colors=px.colors.qualitative.Set2),
    ))
    fig.update_layout(
        title=dict(text="Portfolio weights", font_size=13),
        height=300,
        margin=dict(t=50, b=10, l=10, r=10),
        paper_bgcolor="white",
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📉 VaR Calculator")
    st.markdown("---")

    # Country
    country = st.selectbox(
        "Market",
        ["India (NSE)", "US (S&P 500)"],
        index=0,
    )

    stock_df = load_stock_list(country)
    suffix = ".NS" if country == "India (NSE)" else ""

    # Build search options: "SYMBOL — Name (Sector)"
    stock_df["display"] = (
        stock_df["symbol"] + " — " + stock_df["name"] + " (" + stock_df["sector"] + ")"
    )
    display_to_symbol = dict(zip(stock_df["display"], stock_df["symbol"]))

    st.markdown("#### Stock selection")
    selected_displays = st.multiselect(
        "Search and select stocks",
        options=stock_df["display"].tolist(),
        default=[stock_df["display"].iloc[0]],
        help="Type to search by name, symbol, or sector",
    )

    if not selected_displays:
        st.warning("Select at least one stock.")
        st.stop()

    selected_symbols = [display_to_symbol[d] for d in selected_displays]
    tickers_with_suffix = [s + suffix for s in selected_symbols]
    n_stocks = len(selected_symbols)

    # Portfolio weights (only for multi-stock)
    weights = []
    if n_stocks > 1:
        st.markdown("#### Portfolio weights (%)")
        equal = round(100 / n_stocks, 1)
        total_w = 0
        for sym in selected_symbols:
            w = st.number_input(
                sym,
                min_value=0.0,
                max_value=100.0,
                value=equal,
                step=0.5,
                format="%.1f",
                key=f"w_{sym}",
            )
            weights.append(w)
            total_w += w
        if abs(total_w - 100) > 0.5:
            st.error(f"Weights sum to {total_w:.1f}% — must equal 100%.")
            st.stop()
        weights = [w / 100 for w in weights]
    else:
        weights = [1.0]

    st.markdown("#### Parameters")
    position_size = st.number_input(
        "Position size (₹)" if country == "India (NSE)" else "Position size ($)",
        min_value=10_000,
        max_value=100_000_000,
        value=1_000_000,
        step=50_000,
        format="%d",
    )

    conf_opts = {"90%": 0.90, "95%": 0.95, "99%": 0.99, "99.9%": 0.999}
    conf_label = st.selectbox("Confidence level", list(conf_opts.keys()), index=1)
    conf = conf_opts[conf_label]

    horizon = st.slider("Holding period (days)", 1, 30, 1)

    lookback_opts = {"1 Year": 252, "2 Years": 504, "3 Years": 756, "5 Years": 1260}
    lookback_label = st.selectbox("Historical lookback", list(lookback_opts.keys()), index=0)
    lookback_days = lookback_opts[lookback_label]

    method = st.radio(
        "VaR method",
        ["Parametric", "Historical", "Monte Carlo"],
        index=0,
        help="Parametric: assumes normality | Historical: uses actual returns | Monte Carlo: simulates 10k paths",
    )

    st.markdown("---")
    run = st.button("▶  Calculate VaR", type="primary", use_container_width=True)


# ─────────────────────────────────────────────
# MAIN PANEL
# ─────────────────────────────────────────────
st.markdown("# 📉 Value at Risk Calculator")
st.markdown(
    f"**{country}** &nbsp;|&nbsp; "
    f"**{', '.join(selected_symbols)}** &nbsp;|&nbsp; "
    f"Conf: **{conf_label}** &nbsp;|&nbsp; "
    f"Horizon: **{horizon}d** &nbsp;|&nbsp; "
    f"Method: **{method}**"
)
st.markdown("---")

if not run:
    st.markdown("""
    <div class="info-box">
        ℹ️ Configure parameters in the sidebar and click <strong>Calculate VaR</strong> to run the analysis.
        <br><br>
        <strong>What this tool computes:</strong><br>
        • <strong>Parametric VaR</strong> — uses mean and standard deviation assuming a normal distribution<br>
        • <strong>Historical VaR</strong> — uses the actual historical return distribution (no normality assumption)<br>
        • <strong>Monte Carlo VaR</strong> — simulates 10,000 price paths using GBM<br>
        • <strong>CVaR / Expected Shortfall</strong> — expected loss when VaR is breached<br>
        • <strong>Breach backtesting</strong> — Kupiec test to validate your VaR model<br>
        • <strong>Correlation matrix</strong> — shows diversification effect for portfolios
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────
# DATA FETCH
# ─────────────────────────────────────────────
end_date   = datetime.today()
start_date = end_date - timedelta(days=int(lookback_days * 1.4))  # buffer for weekends/holidays

with st.spinner("Fetching price data from Yahoo Finance…"):
    prices_raw = fetch_prices(
        tuple(tickers_with_suffix),
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
    )

# Handle single vs multi ticker column structure
if prices_raw is None or prices_raw.empty:
    st.error("No data returned. Check ticker symbols or try a different date range.")
    st.stop()

# Clean up columns for single ticker
if n_stocks == 1:
    if isinstance(prices_raw.columns, pd.MultiIndex):
        prices_raw = prices_raw.droplevel(0, axis=1) if prices_raw.columns.nlevels > 1 else prices_raw
        prices_raw.columns = [selected_symbols[0]]
    elif len(prices_raw.columns) == 1:
        prices_raw.columns = [selected_symbols[0]]
else:
    if isinstance(prices_raw.columns, pd.MultiIndex):
        prices_raw.columns = prices_raw.columns.get_level_values(1)
        prices_raw.columns = [c.replace(suffix, "") for c in prices_raw.columns]

# Keep only last `lookback_days` trading days
prices_raw = prices_raw.dropna(how="all").tail(lookback_days)

if prices_raw.empty or len(prices_raw) < 30:
    st.error(f"Insufficient data ({len(prices_raw)} rows). Try a longer lookback or check the ticker.")
    st.stop()

returns_df = compute_returns(prices_raw)

# ─────────────────────────────────────────────
# COMPUTE VaR
# ─────────────────────────────────────────────
currency = "₹" if country == "India (NSE)" else "$"

if n_stocks == 1:
    sym = selected_symbols[0]
    rets = returns_df[sym].dropna() if sym in returns_df.columns else returns_df.iloc[:, 0].dropna()
    prices_s = prices_raw[sym] if sym in prices_raw.columns else prices_raw.iloc[:, 0]

    if method == "Parametric":
        result = parametric_var(rets, conf, horizon)
    elif method == "Historical":
        result = historical_var(rets, conf, horizon)
    else:
        result = monte_carlo_var(rets, conf, horizon)

    var_pct  = result["var_pct"]
    cvar_pct = result["cvar_pct"]
    var_amt  = var_pct  * position_size
    cvar_amt = cvar_pct * position_size
    cutoff_price = position_size * (1 - var_pct)
    ann_vol  = rets.std() * np.sqrt(252)
    ann_ret  = rets.mean() * 252
    dist     = distribution_stats(rets)
    bt       = backtest_var(rets, var_pct / np.sqrt(horizon), conf)  # daily VaR for backtest

else:
    weights_arr = np.array(weights)
    rets_aligned = returns_df[selected_symbols].dropna()
    result = portfolio_var(rets_aligned, weights_arr, conf, horizon)
    var_pct  = result["var_pct"]
    cvar_pct = result["cvar_pct"]
    var_amt  = var_pct  * position_size
    cvar_amt = cvar_pct * position_size
    cutoff_price = position_size * (1 - var_pct)
    port_ret_series = rets_aligned @ weights_arr
    ann_vol  = port_ret_series.std() * np.sqrt(252)
    ann_ret  = port_ret_series.mean() * 252
    dist     = distribution_stats(port_ret_series)
    bt       = backtest_var(port_ret_series, var_pct / np.sqrt(horizon), conf)


# ─────────────────────────────────────────────
# METRIC CARDS
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Risk summary</div>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label=f"VaR ({conf_label}, {horizon}d)",
        value=f"{currency}{var_amt:,.0f}",
        delta=f"{var_pct*100:.2f}% of position",
        delta_color="inverse",
    )
with col2:
    st.metric(
        label=f"CVaR / ES ({conf_label})",
        value=f"{currency}{cvar_amt:,.0f}",
        delta=f"{cvar_pct*100:.2f}% of position",
        delta_color="inverse",
    )
with col3:
    st.metric(
        label="VaR cutoff value",
        value=f"{currency}{cutoff_price:,.0f}",
        delta=f"Floor for {horizon}d hold",
        delta_color="off",
    )
with col4:
    st.metric(
        label="Annualised volatility",
        value=f"{ann_vol*100:.1f}%",
        delta="252-day scaled",
        delta_color="off",
    )
with col5:
    st.metric(
        label="Annualised return (hist.)",
        value=f"{ann_ret*100:.1f}%",
        delta=f"Based on {len(returns_df)} trading days",
        delta_color="normal",
    )


# ─────────────────────────────────────────────
# DISTRIBUTION STATS (single stock)
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Distribution diagnostics</div>', unsafe_allow_html=True)

ds_col1, ds_col2, ds_col3, ds_col4, ds_col5 = st.columns(5)

skew_color  = "badge-yellow" if abs(dist["skewness"]) > 0.5 else "badge-green"
kurt_color  = "badge-red"    if dist["excess_kurtosis"] > 3  else "badge-green"
norm_color  = "badge-green"  if dist["is_normal"] else "badge-red"
norm_label  = "Normal ✓"     if dist["is_normal"] else "Non-normal ✗"

with ds_col1:
    st.markdown(f"""
    <div style="text-align:center">
        <div style="font-size:0.8rem;color:#6c757d;">Skewness</div>
        <div style="font-size:1.5rem;font-weight:700">{dist['skewness']:.3f}</div>
        <span class="badge {skew_color}">{'Negative skew' if dist['skewness'] < -0.5 else 'Positive skew' if dist['skewness'] > 0.5 else 'Approx. symmetric'}</span>
    </div>
    """, unsafe_allow_html=True)

with ds_col2:
    st.markdown(f"""
    <div style="text-align:center">
        <div style="font-size:0.8rem;color:#6c757d;">Excess kurtosis</div>
        <div style="font-size:1.5rem;font-weight:700">{dist['excess_kurtosis']:.3f}</div>
        <span class="badge {kurt_color}">{'Fat tails ⚠' if dist['excess_kurtosis'] > 3 else 'Normal tails'}</span>
    </div>
    """, unsafe_allow_html=True)

with ds_col3:
    st.markdown(f"""
    <div style="text-align:center">
        <div style="font-size:0.8rem;color:#6c757d;">Jarque-Bera stat</div>
        <div style="font-size:1.5rem;font-weight:700">{dist['jb_stat']:.1f}</div>
        <span class="badge {norm_color}">{norm_label}</span>
    </div>
    """, unsafe_allow_html=True)

with ds_col4:
    breach_color = "badge-green" if bt["breach_rate"] <= bt["expected_rate"] * 1.5 else "badge-red"
    st.markdown(f"""
    <div style="text-align:center">
        <div style="font-size:0.8rem;color:#6c757d;">VaR breaches (hist.)</div>
        <div style="font-size:1.5rem;font-weight:700">{bt['breaches']} / {bt['total']}</div>
        <span class="badge {breach_color}">{bt['breach_rate']*100:.1f}% vs {bt['expected_rate']*100:.0f}% expected</span>
    </div>
    """, unsafe_allow_html=True)

with ds_col5:
    if not np.isnan(bt["pof_pvalue"]):
        pof_color = "badge-green" if bt["pof_pvalue"] > 0.05 else "badge-red"
        pof_label = "Model valid ✓" if bt["pof_pvalue"] > 0.05 else "Model rejected ✗"
    else:
        pof_color = "badge-yellow"
        pof_label = "Insufficient data"
    pof_display = f"{bt['pof_pvalue']:.3f}" if not np.isnan(bt['pof_pvalue']) else "N/A"
    st.markdown(f"""
    <div style="text-align:center">
        <div style="font-size:0.8rem;color:#6c757d;">Kupiec test (p-value)</div>
        <div style="font-size:1.5rem;font-weight:700">{pof_display}</div>
        <span class="badge {pof_color}">{pof_label}</span>
    </div>
    """, unsafe_allow_html=True)

# Normality warning
if not dist["is_normal"]:
    st.markdown("""
    <div class="warn-box">
        ⚠️ <strong>Normality rejected (Jarque-Bera p &lt; 0.05)</strong> — The return distribution has fat tails or significant skewness.
        Parametric VaR may <strong>underestimate</strong> actual tail risk. Consider using Historical or Monte Carlo VaR instead.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CHARTS — Row 1
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Distribution & price charts</div>', unsafe_allow_html=True)
chart_col1, chart_col2 = st.columns([1.1, 1])

with chart_col1:
    label = " + ".join(selected_symbols) if n_stocks > 1 else selected_symbols[0]
    if method == "Parametric":
        dist_ret = rets if n_stocks == 1 else port_ret_series
        fig_dist = plot_distribution(dist_ret, var_pct, cvar_pct, conf, method, label)
    elif method == "Historical":
        dist_ret = rets if n_stocks == 1 else port_ret_series
        fig_dist = plot_distribution(dist_ret, var_pct, cvar_pct, conf, method, label)
    else:
        dist_ret = rets if n_stocks == 1 else port_ret_series
        fig_dist = plot_mc_histogram(result["sim"], var_pct, cvar_pct, conf)
    st.plotly_chart(fig_dist, use_container_width=True)

with chart_col2:
    if n_stocks == 1:
        fig_price = plot_price_with_var(prices_s, rets, var_pct, position_size, selected_symbols[0])
        st.plotly_chart(fig_price, use_container_width=True)
    else:
        # Portfolio weights pie
        st.plotly_chart(plot_weights_pie(weights, selected_symbols), use_container_width=True)


# ─────────────────────────────────────────────
# PORTFOLIO: Correlation + Diversification
# ─────────────────────────────────────────────
if n_stocks > 1:
    st.markdown('<div class="section-header">Portfolio analytics</div>', unsafe_allow_html=True)
    corr_col, div_col = st.columns([1.4, 1])

    with corr_col:
        st.plotly_chart(plot_correlation_heatmap(rets_aligned), use_container_width=True)

    with div_col:
        st.markdown("#### Diversification benefit")
        naive  = result["naive_var_pct"]
        actual = result["var_pct"]
        benefit_pct = result["div_benefit_pct"]
        benefit_amt = benefit_pct * position_size

        st.metric("Naive VaR (no correlation)",
                  f"{currency}{naive * position_size:,.0f}",
                  f"{naive*100:.2f}%")
        st.metric("Correlation-adjusted VaR",
                  f"{currency}{actual * position_size:,.0f}",
                  f"{actual*100:.2f}%",
                  delta_color="inverse")
        st.metric("Diversification benefit",
                  f"{currency}{benefit_amt:,.0f}",
                  f"{benefit_pct*100:.2f}% saved by diversification",
                  delta_color="normal")

        st.markdown("""
        <div class="info-box">
            <strong>How to read this:</strong> Naive VaR assumes zero correlation between stocks.
            The actual VaR accounts for real-world co-movement. The difference is your
            <em>diversification benefit</em> — how much risk you've reduced by not holding a single stock.
        </div>
        """, unsafe_allow_html=True)

        # Per-stock individual VaR
        st.markdown("#### Individual stock VaR")
        rows = []
        for i, sym in enumerate(selected_symbols):
            r = rets_aligned[sym]
            pv = parametric_var(r, conf, horizon)
            alloc = weights[i] * position_size
            rows.append({
                "Ticker": sym,
                "Weight": f"{weights[i]*100:.1f}%",
                "Allocation": f"{currency}{alloc:,.0f}",
                "Ind. VaR %": f"{pv['var_pct']*100:.2f}%",
                "Ind. VaR ₹/$": f"{currency}{pv['var_pct']*alloc:,.0f}",
                "Ann. Vol": f"{r.std()*np.sqrt(252)*100:.1f}%",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# BACKTEST CHART
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">VaR breach backtesting</div>', unsafe_allow_html=True)

bt_ret = rets if n_stocks == 1 else port_ret_series
bt_sym = selected_symbols[0] if n_stocks == 1 else "Portfolio"
# daily VaR for backtest (horizon=1 always)
daily_var = parametric_var(bt_ret, conf, 1)["var_pct"]

fig_bt = plot_backtest(bt_ret, daily_var, bt_sym)
st.plotly_chart(fig_bt, use_container_width=True)

breach_summary = backtest_var(bt_ret, daily_var, conf)
b_col1, b_col2, b_col3 = st.columns(3)
with b_col1:
    st.metric("Total breach days", breach_summary["breaches"])
with b_col2:
    st.metric("Observed breach rate", f"{breach_summary['breach_rate']*100:.2f}%",
              delta=f"Expected: {breach_summary['expected_rate']*100:.1f}%",
              delta_color="off")
with b_col3:
    pv = breach_summary["pof_pvalue"]
    validity = "✓ Valid" if (not np.isnan(pv) and pv > 0.05) else "✗ Rejected"
    st.metric("Kupiec test", validity,
              delta=f"p = {pv:.3f}" if not np.isnan(pv) else "N/A",
              delta_color="off")


# ─────────────────────────────────────────────
# ALL-METHODS COMPARISON TABLE
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Method comparison</div>', unsafe_allow_html=True)

compare_rows = []
ret_for_compare = rets if n_stocks == 1 else port_ret_series

for m_name, m_func in [
    ("Parametric", lambda r, c, h: parametric_var(r, c, h)),
    ("Historical", lambda r, c, h: historical_var(r, c, h)),
    ("Monte Carlo", lambda r, c, h: monte_carlo_var(r, c, h)),
]:
    res = m_func(ret_for_compare, conf, horizon)
    vp = res["var_pct"]
    cp = res["cvar_pct"]
    compare_rows.append({
        "Method": m_name,
        f"VaR % ({conf_label})": f"{vp*100:.3f}%",
        f"VaR {currency}": f"{currency}{vp*position_size:,.0f}",
        f"CVaR % ({conf_label})": f"{cp*100:.3f}%",
        f"CVaR {currency}": f"{currency}{cp*position_size:,.0f}",
        "Assumes normality": "Yes" if m_name == "Parametric" else "No",
    })

st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# ROLLING VaR CHART (for multi-stock: portfolio returns)
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Rolling 30-day VaR over time</div>', unsafe_allow_html=True)

roll_ret = rets if n_stocks == 1 else port_ret_series
window = 30
roll_mu_s    = roll_ret.rolling(window).mean()
roll_sig_s   = roll_ret.rolling(window).std()
z_conf       = stats.norm.ppf(1 - conf)
rolling_var  = -(roll_mu_s * horizon + z_conf * roll_sig_s * np.sqrt(horizon))
rolling_var_amt = rolling_var * position_size

fig_roll = go.Figure()
fig_roll.add_trace(go.Scatter(
    x=rolling_var_amt.index,
    y=rolling_var_amt.values,
    mode="lines",
    name=f"Rolling VaR ({conf_label})",
    line=dict(color=COLORS["danger"], width=2),
    fill="tozeroy",
    fillcolor="rgba(214,39,40,0.08)",
))
fig_roll.add_hline(y=var_amt, line_dash="dash", line_color=COLORS["primary"],
                   annotation_text=f"  Full-period VaR: {currency}{var_amt:,.0f}",
                   annotation_font_color=COLORS["primary"])
fig_roll.update_layout(
    title=dict(text=f"Rolling {window}-day parametric VaR — {bt_sym}", font_size=14),
    xaxis_title="Date",
    yaxis_title=f"VaR ({currency})",
    height=340,
    margin=dict(t=60, b=40, l=70, r=30),
    plot_bgcolor="white", paper_bgcolor="white",
    xaxis=dict(gridcolor="#f0f0f0"),
    yaxis=dict(gridcolor="#f0f0f0", tickformat=",.0f"),
)
st.plotly_chart(fig_roll, use_container_width=True)


# ─────────────────────────────────────────────
# DATA TABLE + CSV EXPORT
# ─────────────────────────────────────────────
with st.expander("📋 Raw price & returns data"):
    tab1, tab2 = st.tabs(["Prices", "Log returns"])
    with tab1:
        st.dataframe(prices_raw.tail(50).style.format("{:.2f}"), use_container_width=True)
    with tab2:
        st.dataframe(returns_df.tail(50).style.format("{:.4f}"), use_container_width=True)

st.markdown('<div class="section-header">Export results</div>', unsafe_allow_html=True)

# Build summary export dataframe
export_rows = []
ret_for_export = rets if n_stocks == 1 else port_ret_series
for m_name, m_func in [
    ("Parametric", lambda r, c, h: parametric_var(r, c, h)),
    ("Historical", lambda r, c, h: historical_var(r, c, h)),
    ("Monte Carlo", lambda r, c, h: monte_carlo_var(r, c, h)),
]:
    res = m_func(ret_for_export, conf, horizon)
    export_rows.append({
        "Ticker(s)": " + ".join(selected_symbols),
        "Method": m_name,
        "Confidence": conf_label,
        "Horizon (days)": horizon,
        "Position Size": position_size,
        "VaR (%)": round(res["var_pct"] * 100, 4),
        f"VaR ({currency})": round(res["var_pct"] * position_size, 2),
        "CVaR (%)": round(res["cvar_pct"] * 100, 4),
        f"CVaR ({currency})": round(res["cvar_pct"] * position_size, 2),
        "Ann. Volatility (%)": round(ret_for_export.std() * np.sqrt(252) * 100, 4),
        "Ann. Return (%)": round(ret_for_export.mean() * 252 * 100, 4),
        "Skewness": round(dist["skewness"], 4),
        "Excess Kurtosis": round(dist["excess_kurtosis"], 4),
        "JB p-value": round(dist["jb_pvalue"], 4),
        "Normal Distribution": "Yes" if dist["is_normal"] else "No",
        "VaR Breaches": bt["breaches"],
        "Breach Rate (%)": round(bt["breach_rate"] * 100, 2),
        "Kupiec p-value": round(bt["pof_pvalue"], 4) if not np.isnan(bt["pof_pvalue"]) else "N/A",
        "Lookback": lookback_label,
        "Market": country,
        "Run Date": datetime.today().strftime("%Y-%m-%d"),
    })

export_df = pd.DataFrame(export_rows)
csv_bytes = export_df.to_csv(index=False).encode("utf-8")

exp_col1, exp_col2 = st.columns(2)
with exp_col1:
    st.download_button(
        label="⬇️ Download VaR summary (CSV)",
        data=csv_bytes,
        file_name=f"var_summary_{'_'.join(selected_symbols)}_{datetime.today().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True,
    )
with exp_col2:
    returns_csv = returns_df.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download returns data (CSV)",
        data=returns_csv,
        file_name=f"returns_{'_'.join(selected_symbols)}_{datetime.today().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
    VaR Calculator · Built with Streamlit + yfinance + Plotly<br>
    For educational and portfolio analysis purposes only. Not financial advice.
</div>
""", unsafe_allow_html=True)
