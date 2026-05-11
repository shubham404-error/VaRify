# 📉 VaR Calculator — Portfolio Risk Analytics Tool

A professional **Value at Risk (VaR)** calculator built with **Streamlit**, **yfinance**, and **Plotly**. Supports single stocks and multi-stock portfolios with three VaR methodologies, distribution diagnostics, backtesting, and CSV export.

---

## 🖥️ Live Demo

> Deploy to **Streamlit Community Cloud** in one click — see [Deployment](#deployment) below.

---

## ✨ Features

### Risk Methods
| Method | Description |
|---|---|
| **Parametric VaR** | Assumes normal distribution. Uses μ and σ scaled by √T. Fast and CFA-standard. |
| **Historical Simulation** | Uses the actual historical return distribution. No normality assumption. |
| **Monte Carlo** | Simulates 10,000 GBM paths over the holding period. Best for stress testing. |

### Outputs
- **VaR (₹/$)** and **CVaR / Expected Shortfall** — at 90%, 95%, 99%, or 99.9% confidence
- **VaR cutoff price** — the portfolio floor for the chosen holding period
- **Annualised volatility** and **expected return**
- **Normal distribution chart** with shaded loss tail, VaR and CVaR cutoff lines
- **Price chart** with rolling 30-day VaR floor bands
- **Rolling VaR over time** — shows how tail risk evolved across the lookback period
- **All-methods comparison table** — Parametric vs Historical vs Monte Carlo side by side
- **Backtest chart** — every trading day coloured red if actual loss exceeded VaR
- **Kupiec POF test** — formal statistical validity test for the VaR model

### Distribution Diagnostics
- Skewness, Excess Kurtosis
- Jarque-Bera normality test with pass/fail badge
- Fat tail warning when parametric VaR may underestimate risk

### Portfolio Mode (2+ stocks)
- **Custom weights** with live validation (must sum to 100%)
- **Correlation matrix heatmap** (Plotly, colour-coded ρ values)
- **Diversification benefit** — naive VaR vs correlation-adjusted VaR in ₹/$ terms
- Per-stock individual VaR breakdown table

### Markets Supported
- 🇮🇳 **India (NSE)** — Nifty 500 universe, `.NS` suffix auto-appended
- 🇺🇸 **US (S&P 500)** — S&P 500 universe

### Export
- Download VaR summary as **CSV** (all 3 methods × all metrics)
- Download raw **returns data** as CSV

---

## 📁 Project Structure

```
var_app/
├── app.py              ← Single-file Streamlit application
├── requirements.txt    ← Python dependencies
├── .gitignore
├── README.md
└── data/
    ├── nifty500.csv    ← NSE stock universe (symbol, name, sector)
    └── sp500.csv       ← US stock universe (symbol, name, sector)
```

---

## 🚀 Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/var-calculator.git
cd var-calculator
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 📦 Dependencies

```
streamlit>=1.32.0
yfinance>=0.2.38
pandas>=2.0.0
numpy>=1.26.0
scipy>=1.12.0
plotly>=5.20.0
```

---

## 🧮 Methodology

### Parametric VaR
Assumes log returns follow a normal distribution:

```
VaR = -(μ·T + z_{α} · σ · √T)
```

Where:
- `μ` = mean daily log return
- `σ` = standard deviation of daily log returns
- `T` = holding period in days
- `z_{α}` = inverse normal at `(1 - confidence)` — e.g. −1.645 for 95%

### CVaR / Expected Shortfall (Parametric)
```
CVaR = -(μ·T - σ·√T · φ(z_{α}) / (1 - α))
```
Where `φ` is the standard normal PDF.

### Historical VaR
Returns are scaled by √T, then:
```
VaR = -percentile(returns · √T, (1-α) · 100)
```
CVaR = mean of returns in the tail beyond the VaR threshold.

### Monte Carlo VaR
10,000 paths of `T` daily returns are simulated from `N(μ, σ²)`, summed to cumulative log returns:
```
VaR = -percentile(simulated_paths, (1-α) · 100)
```

### Portfolio VaR (with correlation)
```
σ_portfolio = √(wᵀ Σ w)
VaR_portfolio = -(μ_p·T + z_{α} · σ_portfolio · √T)
```
Where `Σ` is the full covariance matrix estimated from historical returns.

### Kupiec POF Test
Likelihood ratio test that the observed breach rate is statistically consistent with the model's expected breach rate `(1 - α)`. A p-value > 0.05 means the model is not rejected at 5% significance.

---

## Deployment

### Streamlit Community Cloud (free)
1. Push the repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account → select this repo → set `app.py` as the entry point
4. Click **Deploy**

The app auto-installs `requirements.txt` — no configuration needed.

---

## ⚠️ Disclaimer

This tool is for **educational and portfolio analysis purposes only**. It is not financial advice. VaR models have known limitations — they assume historical patterns repeat, underestimate tail risk during market crises, and do not account for liquidity or concentration risk. Always use multiple risk measures and consult a qualified professional before making investment decisions.

---

## 🛠️ Built With

- [Streamlit](https://streamlit.io) — UI framework
- [yfinance](https://pypi.org/project/yfinance/) — Yahoo Finance price data
- [Plotly](https://plotly.com/python/) — interactive charts
- [SciPy](https://scipy.org) — statistical distributions and tests
- [pandas](https://pandas.pydata.org) / [NumPy](https://numpy.org) — data processing

---

## 📄 License

MIT License — free to use, modify, and distribute.
