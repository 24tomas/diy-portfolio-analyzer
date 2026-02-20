"""
Professional Portfolio Analysis Dashboard
==========================================
Tech Stack: Streamlit Â· yfinance Â· pandas Â· NumPy Â· Plotly Â· QuantStats Â· PyPortfolioOpt
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import quantstats as qs
import time
import random
import json
import os
from datetime import date, timedelta
try:
    import pypfopt
    HAS_PYPFOPT = True
except Exception:
    pypfopt = None
    HAS_PYPFOPT = False
try:
    from pypfopt import risk_models
    HAS_RISK_MODELS = True
except Exception:
    risk_models = None
    HAS_RISK_MODELS = False
try:
    import statsmodels.api as sm_api
    HAS_STATSMODELS = True
except ImportError:
    sm_api = None
    HAS_STATSMODELS = False
try:
    from scipy.linalg import cholesky as sp_cholesky
    HAS_SCIPY = True
except Exception:
    sp_cholesky = None
    HAS_SCIPY = False
try:
    import scipy.optimize as sp_opt
    HAS_SCIPY_OPT = True
except Exception:
    sp_opt = None
    HAS_SCIPY_OPT = False

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. PAGE CONFIG
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.set_page_config(
    page_title="Portfolio Analyzer",
    page_icon="đź“Š",
    layout="wide",
    initial_sidebar_state="expanded",
)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. PORTFOLIO STORAGE (JSON file on disk)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
PORTFOLIO_FILE = "saved_portfolios.json"
STRESS_START_DATE = date(2007, 1, 1)

def load_saved_portfolios() -> dict:
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    return {}

def save_portfolios_to_disk(portfolios: dict):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolios, f, indent=2)

if "saved_portfolios" not in st.session_state:
    st.session_state["saved_portfolios"] = load_saved_portfolios()


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. SIDEBAR
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with st.sidebar:

    # â”€â”€ 3a. Portfolio manager â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.header("đź’ľ Portfolio Manager")
    saved = st.session_state["saved_portfolios"]
    portfolio_names = list(saved.keys())

    # Load a saved portfolio
    if portfolio_names:
        load_choice = st.selectbox(
            "Load saved portfolio",
            options=["â€” New portfolio â€”"] + portfolio_names,
            key="load_choice",
        )
        if load_choice != "â€” New portfolio â€”":
            if st.button(f"đź“‚ Load '{load_choice}'", width='stretch'):
                data = saved[load_choice]
                st.session_state["holdings"] = pd.DataFrame(data["holdings"])
                st.rerun()

        # Delete
        del_choice = st.selectbox(
            "Delete a portfolio", options=["â€”"] + portfolio_names,
            key="del_choice",
        )
        if del_choice != "â€”":
            if st.button(f"đź—‘ď¸Ź Delete '{del_choice}'", width='stretch'):
                del st.session_state["saved_portfolios"][del_choice]
                save_portfolios_to_disk(st.session_state["saved_portfolios"])
                st.rerun()
    else:
        st.caption("No saved portfolios yet.")

    st.divider()

    # â”€â”€ 3b. Holdings editor â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.header("đź“ Portfolio Holdings")
    st.caption("Enter tickers and how many shares you own.")

    if "holdings" not in st.session_state:
        st.session_state["holdings"] = pd.DataFrame(
            {"Ticker": ["AAPL", "MSFT", "GOOGL"], "Shares": [10, 15, 8]}
        )

    edited_holdings = st.data_editor(
        st.session_state["holdings"],
        num_rows="dynamic",
        width='stretch',
        column_config={
            "Ticker": st.column_config.TextColumn(
                "Ticker", help="Stock symbol, e.g. AAPL", required=True),
            "Shares": st.column_config.NumberColumn(
                "Shares", help="Number of shares you own",
                min_value=0.0001, required=True, format="%.4g"),
        },
        hide_index=True,
        key="holdings_editor",
    )

    # Save current portfolio
    save_name = st.text_input("Save current portfolio as", key="save_name_input")
    if st.button("đź’ľ Save Portfolio", width='stretch'):
        name = save_name.strip()
        if not name:
            st.warning("Enter a name first.")
        else:
            h = edited_holdings.dropna(subset=["Ticker", "Shares"])
            st.session_state["saved_portfolios"][name] = {
                "holdings": h.to_dict(orient="list"),
            }
            save_portfolios_to_disk(st.session_state["saved_portfolios"])
            st.success(f"Saved '{name}'!")

    st.divider()

    # â”€â”€ 3c. Compare with another portfolio â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.header("đź”€ Compare Portfolios")
    compare_options = ["None (benchmark only)"] + portfolio_names
    compare_choice = st.selectbox("Compare with", options=compare_options, key="compare_sel")

    st.divider()

    # â”€â”€ 3d. Date range, benchmark, rf â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date",
                                   value=date.today() - timedelta(days=3*365))
    with col2:
        end_date = st.date_input("End Date", value=date.today())

    benchmark_ticker = st.text_input("Benchmark", value="SPY",
                                     help="Used for comparison in charts and factor analysis")

    risk_free_rate = st.number_input("Risk-Free Rate (%)", min_value=0.0,
                                     max_value=20.0, value=4.5, step=0.1) / 100

    fetch_sector_data = st.checkbox(
        "Fetch sector/industry metadata",
        value=False,
        help="Adds extra Yahoo requests per ticker. Keep off to reduce rate-limit risk.",
    )

    run_btn = st.button("đźš€  Analyze Portfolio", type="primary", width='stretch')

    if st.button("Clear cached data", width='stretch'):
        st.cache_data.clear()
        st.success("Cleared Streamlit cached data.")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. HELPERS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def validate_holdings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().dropna(subset=["Ticker", "Shares"])
    df["Ticker"] = df["Ticker"].str.strip().str.upper()
    df = df[df["Ticker"].str.len() > 0]
    df = df[df["Shares"] > 0]
    df = df.groupby("Ticker", as_index=False)["Shares"].sum()
    if df.empty:
        st.error("âš ď¸Ź  Add at least one valid holding.")
        st.stop()
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=6 * 3600, persist="disk", max_entries=128)
def download_prices(tickers: tuple, start, end, max_retries=5):
    tickers = list(tickers)
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                timeout=30,
                threads=False,
            )
            if raw.empty:
                raise RuntimeError("Yahoo returned an empty response.")
            break
        except Exception as e:
            last_error = e
        if attempt < max_retries:
            wait = min(90, (2 ** attempt) + random.uniform(0.5, 1.5))
            time.sleep(wait)
    else:
        raise RuntimeError(
            f"Yahoo download failed after {max_retries} attempts. Last error: {last_error}"
        )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])
    else:
        prices = raw[["Close"]].copy()
        prices.columns = [tickers[0]]

    prices = prices.ffill().bfill()
    prices = prices.dropna(axis=1, how="all")
    if prices.empty:
        raise RuntimeError("Yahoo returned no usable closing price columns.")
    return prices


def compute_weights_and_returns(prices, holdings):
    valid = holdings[holdings["Ticker"].isin(prices.columns)].copy()
    if valid.empty:
        st.error("âťŚ  None of the tickers have price data.")
        st.stop()
    latest = prices.iloc[-1]
    valid["Price"] = valid["Ticker"].map(latest)
    valid["Value"] = valid["Shares"] * valid["Price"]
    total = valid["Value"].sum()
    valid["Weight"] = valid["Value"] / total
    tickers = valid["Ticker"].tolist()
    w = valid["Weight"].values
    ind_ret = prices[tickers].pct_change().dropna()
    port_ret = ind_ret.dot(w)
    port_ret.name = "Portfolio"
    return port_ret, ind_ret, valid.set_index("Ticker")["Weight"]


@st.cache_data(show_spinner="đź“ˇ Downloading Fama-French factors â€¦", ttl=86400)
def download_fama_french() -> pd.DataFrame:
    """
    Download the Fama-French 3-Factor daily dataset directly from
    Kenneth French's website. Returns a DataFrame with columns
    ['Mkt-RF', 'SMB', 'HML', 'RF'] in decimal form (not percentage).
    Cached for 24 hours.
    """
    import urllib.request, zipfile, io

    url = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
           "ftp/F-F_Research_Data_Factors_daily_CSV.zip")
    resp = urllib.request.urlopen(url, timeout=30)
    z = zipfile.ZipFile(io.BytesIO(resp.read()))
    csv_name = z.namelist()[0]

    with z.open(csv_name) as f:
        raw = f.read().decode("utf-8")

    # Parse only lines that start with a digit (YYYYMMDD format)
    data_lines = [
        line.strip().replace("\r", "")
        for line in raw.strip().split("\n")
        if line.strip() and line.strip()[0].isdigit() and "," in line
    ]

    from io import StringIO
    ff_df = pd.read_csv(
        StringIO("\n".join(data_lines)),
        header=None,
        names=["Date", "Mkt-RF", "SMB", "HML", "RF"],
    )
    ff_df["Date"] = pd.to_datetime(ff_df["Date"], format="%Y%m%d")
    ff_df = ff_df.set_index("Date")
    ff_df = ff_df.apply(pd.to_numeric, errors="coerce").dropna()
    ff_df = ff_df / 100  # percentage â†’ decimal

    return ff_df


@st.cache_data(show_spinner="đźŹ˘ Looking up sector data â€¦", ttl=7 * 86400, persist="disk", max_entries=256)
def get_sector_map(tickers: tuple) -> dict:
    """
    For each ticker, query yf.Ticker(t).info to retrieve sector and industry.
    Returns dict: {ticker: {"sector": ..., "industry": ...}}.
    Cached for 7 days. Accepts a tuple (not list) so Streamlit can hash it.
    """
    result = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            result[t] = {
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
            }
        except Exception:
            result[t] = {"sector": "Unknown", "industry": "Unknown"}
        time.sleep(0.8)  # slower to lower rate-limit risk for metadata calls
    return result


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5. MAIN â€” RUN ON BUTTON CLICK
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.title("đź“Š Portfolio Analysis Dashboard")

if not run_btn and "port_ret" not in st.session_state:
    st.info("đź‘ Enter your holdings in the sidebar, then click **Analyze Portfolio**.")
    st.stop()

if run_btn:
    if end_date <= start_date:
        st.error("End Date must be after Start Date.")
        st.stop()

    holdings = validate_holdings(edited_holdings)
    st.session_state["holdings"] = holdings
    bench = benchmark_ticker.strip().upper()

    # Gather all tickers we need (portfolio + benchmark + comparison)
    all_tickers = list(dict.fromkeys(holdings["Ticker"].tolist() + [bench]))

    comp_holdings = None
    if compare_choice and compare_choice != "None (benchmark only)":
        comp_data = st.session_state["saved_portfolios"].get(compare_choice)
        if comp_data:
            comp_holdings = pd.DataFrame(comp_data["holdings"])
            comp_holdings = validate_holdings(comp_holdings)
            for t in comp_holdings["Ticker"]:
                if t not in all_tickers:
                    all_tickers.append(t)

    all_tickers = tuple(sorted(all_tickers))
    full_start = min(start_date, STRESS_START_DATE)

    with st.spinner("Downloading data from Yahoo Finance..."):
        try:
            all_prices = download_prices(all_tickers, full_start, end_date)
        except Exception as e:
            st.error(
                "Yahoo Finance download failed. This is usually a temporary "
                "rate-limit or upstream issue."
            )
            st.info(
                "Try again in 5-15 minutes. Keep ticker count small and leave "
                "'Fetch sector/industry metadata' disabled unless needed."
            )
            st.caption(f"Details: {e}")
            st.stop()

    prices = all_prices.loc[
        (all_prices.index >= pd.Timestamp(start_date))
        & (all_prices.index <= pd.Timestamp(end_date))
    ].copy()

    missing = [t for t in all_tickers if t not in all_prices.columns]
    if missing:
        st.warning(f"No data for: **{', '.join(missing)}**. Excluded.")

    if prices.empty:
        st.error("Yahoo Finance returned no data in the selected date range.")
        st.stop()

    port_ret, ind_ret, wt = compute_weights_and_returns(prices, holdings)

    if bench not in prices.columns:
        st.error(f"Benchmark **{bench}** has no data.")
        st.stop()
    bench_ret = prices[bench].pct_change().dropna()
    bench_ret.name = "Benchmark"

    # Comparison portfolio
    comp_ret = None
    comp_wt = None
    comp_name = None
    if comp_holdings is not None:
        comp_ret, _, comp_wt = compute_weights_and_returns(prices, comp_holdings)
        comp_ret.name = compare_choice
        comp_name = compare_choice

    # Align dates
    common = port_ret.index.intersection(bench_ret.index)
    port_ret = port_ret.loc[common]
    bench_ret = bench_ret.loc[common]
    ind_ret = ind_ret.loc[ind_ret.index.isin(common)]
    if comp_ret is not None:
        comp_ret = comp_ret.reindex(common).dropna()

    # Stress test data (2007+) reuses the already downloaded full history
    stress_prices = all_prices.loc[
        (all_prices.index >= pd.Timestamp(STRESS_START_DATE))
        & (all_prices.index <= pd.Timestamp(end_date))
    ].copy()
    if not stress_prices.empty:
        stress_port, _, _ = compute_weights_and_returns(stress_prices, holdings)
        stress_bench = stress_prices[bench].pct_change().dropna() if bench in stress_prices.columns else pd.Series(dtype=float)
    else:
        stress_port = pd.Series(dtype=float)
        stress_bench = pd.Series(dtype=float)

    # Fama-French factors â€” separate source (Ken French website), no Yahoo
    try:
        ff_factors = download_fama_french()
    except Exception as e:
        st.warning(f"âš ď¸Ź  Could not download Fama-French factors: {e}")
        ff_factors = pd.DataFrame()

    # Sector map â€” cached, only portfolio tickers (not benchmark)
    portfolio_tickers = tuple(sorted(holdings["Ticker"].tolist()))
    if fetch_sector_data:
        try:
            sector_map = get_sector_map(portfolio_tickers)
        except Exception:
            sector_map = {t: {"sector": "Unknown", "industry": "Unknown"} for t in portfolio_tickers}
    else:
        sector_map = {t: {"sector": "Unknown", "industry": "Unknown"} for t in portfolio_tickers}

    # Store everything
    for k, v in {
        "port_ret": port_ret, "bench_ret": bench_ret, "ind_ret": ind_ret,
        "weights": wt, "bench": bench, "rf": risk_free_rate,
        "holdings": holdings, "prices": prices,
        "stress_port": stress_port, "stress_bench": stress_bench,
        "comp_ret": comp_ret, "comp_wt": comp_wt, "comp_name": comp_name,
        "ff_factors": ff_factors, "sector_map": sector_map,
    }.items():
        st.session_state[k] = v

# â”€â”€ Retrieve â”€â”€
port_ret     = st.session_state["port_ret"]
bench_ret    = st.session_state["bench_ret"]
ind_ret      = st.session_state["ind_ret"]
wt           = st.session_state["weights"]
bench        = st.session_state["bench"]
rf           = st.session_state["rf"]
holdings     = st.session_state["holdings"]
prices       = st.session_state["prices"]
stress_port  = st.session_state["stress_port"]
stress_bench = st.session_state["stress_bench"]
comp_ret     = st.session_state.get("comp_ret")
comp_wt      = st.session_state.get("comp_wt")
comp_name    = st.session_state.get("comp_name")
ff_factors   = st.session_state.get("ff_factors", pd.DataFrame())
sector_map   = st.session_state.get("sector_map", {})


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 6. HOLDINGS SUMMARY & KEY METRICS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def annualized_return(r): return ((1 + r).prod() ** (252 / max(len(r), 1))) - 1
def annualized_vol(r):    return r.std() * np.sqrt(252)
def max_drawdown(r):      c = (1 + r).cumprod(); return ((c - c.cummax()) / c.cummax()).min()

st.subheader("Your Holdings")
disp = holdings.copy()
latest = prices.iloc[-1]
disp["Price"] = disp["Ticker"].map(latest)
disp["Value"] = disp["Shares"] * disp["Price"]
total_val = disp["Value"].sum()
disp["Weight"] = disp["Value"] / total_val
# Add sector column if available
if sector_map:
    disp["Sector"] = disp["Ticker"].map(lambda t: sector_map.get(t, {}).get("sector", "Unknown"))
disp_fmt = disp.copy()
disp_fmt["Price"]  = disp["Price"].map("${:,.2f}".format)
disp_fmt["Value"]  = disp["Value"].map("${:,.2f}".format)
disp_fmt["Weight"] = disp["Weight"].map("{:.1%}".format)
st.dataframe(disp_fmt, width='stretch', hide_index=True)

m1, m2, m3, m4, m5 = st.columns(5)
ann_ret = annualized_return(port_ret)
b_ann   = annualized_return(bench_ret)
m1.metric("Ann. Return",     f"{ann_ret:.2%}")
m2.metric("Ann. Volatility", f"{annualized_vol(port_ret):.2%}")
m3.metric("Sharpe Ratio",    f"{qs.stats.sharpe(port_ret, rf=rf):.2f}")
m4.metric("Max Drawdown",    f"{max_drawdown(port_ret):.2%}")
m5.metric("vs Benchmark",    f"{ann_ret - b_ann:+.2%}")

st.divider()


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 7. TABS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
tab_perf, tab_risk, tab_factor, tab_dd, tab_stress, tab_opt, tab_mc = st.tabs(
    ["\U0001F4C8 Performance", "\u2696\ufe0f Risk & Concentration", "\U0001F52C Factor Exposure",
     "\U0001F4C9 Drawdowns", "\U0001F9EA Stress Tests", "\U0001F3AF Optimization", "\U0001F3B2 Monte Carlo"]
)


# ==================== TAB 1 â€” PERFORMANCE ====================
with tab_perf:

    # Growth of $1
    st.subheader("Growth of Investment")
    cum_p = qs.stats.compsum(port_ret)
    cum_b = qs.stats.compsum(bench_ret)
    growth_p = 1 + cum_p
    growth_b = 1 + cum_b

    fig_g = go.Figure()
    fig_g.add_trace(go.Scatter(x=growth_p.index, y=growth_p.values,
                               name="Portfolio", line=dict(color="#636EFA", width=2.5)))
    fig_g.add_trace(go.Scatter(x=growth_b.index, y=growth_b.values,
                               name=f"Benchmark ({bench})",
                               line=dict(color="#EF553B", width=2, dash="dot"),
                               fill="tonexty", fillcolor="rgba(99,110,250,0.07)"))
    if comp_ret is not None:
        cum_c = qs.stats.compsum(comp_ret.reindex(growth_p.index).dropna())
        growth_c = 1 + cum_c
        fig_g.add_trace(go.Scatter(x=growth_c.index, y=growth_c.values,
                                   name=comp_name,
                                   line=dict(color="#00CC96", width=2, dash="dash")))
    fig_g.update_layout(yaxis_title="Growth of $1", yaxis_tickprefix="$",
                        yaxis_tickformat=".2f",
                        legend=dict(orientation="h", y=1.02, x=0),
                        margin=dict(l=50, r=20, t=40, b=40), height=460,
                        hovermode="x unified")
    st.plotly_chart(fig_g, width='stretch')

    # Metric cards
    st.subheader("Return Metrics")
    qs_total  = qs.stats.comp(port_ret)
    qs_cagr   = qs.stats.cagr(port_ret)
    qs_vol    = qs.stats.volatility(port_ret)
    qs_sharpe = qs.stats.sharpe(port_ret, rf=rf)
    qs_info   = qs.stats.information_ratio(port_ret, bench_ret)
    b_total   = qs.stats.comp(bench_ret)
    b_cagr    = qs.stats.cagr(bench_ret)
    b_vol     = qs.stats.volatility(bench_ret)

    k1, k2, k3 = st.columns(3)
    k1.metric("Total Return", f"{qs_total:.2%}", delta=f"{qs_total - b_total:+.2%} vs {bench}")
    k2.metric("CAGR",         f"{qs_cagr:.2%}",  delta=f"{qs_cagr - b_cagr:+.2%} vs {bench}")
    k3.metric("Ann. Vol.",    f"{qs_vol:.2%}",    delta=f"{qs_vol - b_vol:+.2%} vs {bench}",
              delta_color="inverse")
    k4, k5, k6 = st.columns(3)
    k4.metric("Sharpe Ratio",      f"{qs_sharpe:.2f}")
    k5.metric("Information Ratio", f"{qs_info:.2f}",
              help="> 0.5 is good, > 1.0 is excellent.")
    k6.metric("Sortino Ratio",     f"{qs.stats.sortino(port_ret):.2f}",
              help="Like Sharpe but only penalises downside volatility.")

    # Detailed stats table
    st.subheader("Detailed Statistics")
    def _fmt(fn, *a, fmt="{:.2f}"): return fmt.format(fn(*a))
    stats_rows = [
        ("Total Return",    f"{qs.stats.comp(port_ret):.2%}",    f"{qs.stats.comp(bench_ret):.2%}"),
        ("CAGR",            f"{qs.stats.cagr(port_ret):.2%}",    f"{qs.stats.cagr(bench_ret):.2%}"),
        ("Ann. Volatility", f"{qs.stats.volatility(port_ret):.2%}", f"{qs.stats.volatility(bench_ret):.2%}"),
        ("Sharpe",          f"{qs.stats.sharpe(port_ret, rf=rf):.2f}", f"{qs.stats.sharpe(bench_ret, rf=rf):.2f}"),
        ("Sortino",         f"{qs.stats.sortino(port_ret):.2f}",  f"{qs.stats.sortino(bench_ret):.2f}"),
        ("Calmar",          f"{qs.stats.calmar(port_ret):.2f}",   f"{qs.stats.calmar(bench_ret):.2f}"),
        ("Max Drawdown",    f"{qs.stats.max_drawdown(port_ret):.2%}", f"{qs.stats.max_drawdown(bench_ret):.2%}"),
        ("Best Day",        f"{qs.stats.best(port_ret):.2%}",     f"{qs.stats.best(bench_ret):.2%}"),
        ("Worst Day",       f"{qs.stats.worst(port_ret):.2%}",    f"{qs.stats.worst(bench_ret):.2%}"),
        ("Win Rate",        f"{qs.stats.win_rate(port_ret):.1%}", f"{qs.stats.win_rate(bench_ret):.1%}"),
        ("Profit Factor",   f"{qs.stats.profit_factor(port_ret):.2f}", f"{qs.stats.profit_factor(bench_ret):.2f}"),
        ("Skew",            f"{qs.stats.skew(port_ret):.2f}",     f"{qs.stats.skew(bench_ret):.2f}"),
        ("Kurtosis",        f"{qs.stats.kurtosis(port_ret):.2f}", f"{qs.stats.kurtosis(bench_ret):.2f}"),
    ]
    cols = {"Metric": [], "Portfolio": [], f"Benchmark ({bench})": []}
    if comp_ret is not None:
        cols[comp_name] = []
    for metric, pv, bv in stats_rows:
        cols["Metric"].append(metric)
        cols["Portfolio"].append(pv)
        cols[f"Benchmark ({bench})"].append(bv)
        if comp_ret is not None:
            fn_map = {
                "Total Return": lambda: f"{qs.stats.comp(comp_ret):.2%}",
                "CAGR": lambda: f"{qs.stats.cagr(comp_ret):.2%}",
                "Ann. Volatility": lambda: f"{qs.stats.volatility(comp_ret):.2%}",
                "Sharpe": lambda: f"{qs.stats.sharpe(comp_ret, rf=rf):.2f}",
                "Sortino": lambda: f"{qs.stats.sortino(comp_ret):.2f}",
                "Calmar": lambda: f"{qs.stats.calmar(comp_ret):.2f}",
                "Max Drawdown": lambda: f"{qs.stats.max_drawdown(comp_ret):.2%}",
                "Best Day": lambda: f"{qs.stats.best(comp_ret):.2%}",
                "Worst Day": lambda: f"{qs.stats.worst(comp_ret):.2%}",
                "Win Rate": lambda: f"{qs.stats.win_rate(comp_ret):.1%}",
                "Profit Factor": lambda: f"{qs.stats.profit_factor(comp_ret):.2f}",
                "Skew": lambda: f"{qs.stats.skew(comp_ret):.2f}",
                "Kurtosis": lambda: f"{qs.stats.kurtosis(comp_ret):.2f}",
            }
            cols[comp_name].append(fn_map.get(metric, lambda: "â€”")())
    st.dataframe(pd.DataFrame(cols).set_index("Metric"), width='stretch')

    # Monthly heatmap
    st.subheader("Monthly Returns Heatmap")
    monthly = port_ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    mdf = pd.DataFrame({"Year": monthly.index.year, "Month": monthly.index.month,
                         "Return": monthly.values})
    piv = mdf.pivot_table(index="Year", columns="Month", values="Return", aggfunc="first")
    lbl = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    piv.columns = lbl[:len(piv.columns)]
    yearly = port_ret.resample("YE").apply(lambda x: (1 + x).prod() - 1)
    piv["YTD"] = piv.index.map(dict(zip(yearly.index.year, yearly.values)))
    fig_hm = go.Figure(data=go.Heatmap(
        z=piv.values, x=piv.columns.tolist(), y=piv.index.astype(str),
        colorscale="RdYlGn", zmid=0,
        text=np.vectorize(lambda v: f"{v:.1%}" if not np.isnan(v) else "")(piv.values),
        texttemplate="%{text}",
        hovertemplate="Year %{y} â€” %{x}<br>Return: %{z:.2%}<extra></extra>"))
    fig_hm.update_layout(height=max(300, len(piv) * 40),
                         margin=dict(l=40, r=20, t=20, b=40))
    st.plotly_chart(fig_hm, width='stretch')

    # Return distribution
    st.subheader("Daily Return Distribution")
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=port_ret.values, nbinsx=80, name="Portfolio",
                                    marker_color="rgba(99,110,250,0.6)",
                                    histnorm="probability density"))
    fig_dist.add_trace(go.Histogram(x=bench_ret.values, nbinsx=80,
                                    name=f"Benchmark ({bench})",
                                    marker_color="rgba(239,85,59,0.35)",
                                    histnorm="probability density"))
    if comp_ret is not None:
        fig_dist.add_trace(go.Histogram(x=comp_ret.values, nbinsx=80,
                                        name=comp_name,
                                        marker_color="rgba(0,204,150,0.35)",
                                        histnorm="probability density"))
    fig_dist.update_layout(barmode="overlay", height=350,
                           xaxis_title="Daily Return", yaxis_title="Density",
                           xaxis_tickformat=".1%",
                           legend=dict(orientation="h", y=1.02, x=0),
                           margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig_dist, width='stretch')


# ==================== TAB 2 â€” RISK & CONCENTRATION ====================
with tab_risk:

    valid_tickers = [t for t in wt.index if t in prices.columns]
    w_arr = wt[valid_tickers].values

    # â”€â”€ 1. HHI (Herfindahl-Hirschman Index) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.subheader("Portfolio Concentration â€” HHI")
    hhi = float(np.sum(w_arr ** 2))
    hhi_norm = (hhi - 1 / len(w_arr)) / (1 - 1 / len(w_arr)) if len(w_arr) > 1 else 1.0

    hh1, hh2 = st.columns([1, 2])
    with hh1:
        if hhi_norm < 0.15:
            label, color = "Highly Diversified", "đźź˘"
        elif hhi_norm < 0.25:
            label, color = "Moderately Concentrated", "đźźˇ"
        elif hhi_norm < 0.50:
            label, color = "Concentrated", "đźź "
        else:
            label, color = "Highly Concentrated", "đź”´"
        st.metric("HHI", f"{hhi:.4f}", help="Range: 1/N (equal weight) to 1.0 (single stock)")
        st.metric("Normalized HHI", f"{hhi_norm:.2%}")
        st.markdown(f"**{color} {label}**")
    with hh2:
        st.caption(
            "The **Herfindahl-Hirschman Index** measures portfolio concentration. "
            "It's the sum of squared weights. A portfolio equally split among *N* "
            "assets has HHI = 1/N (the minimum); a single-stock portfolio has HHI = 1.\n\n"
            "The **Normalized HHI** rescales this to 0â€“100%, where 0% = perfectly "
            "equal-weighted and 100% = single asset.\n\n"
            "| Norm. HHI | Interpretation |\n"
            "|-----------|----------------|\n"
            "| < 15%     | đźź˘ Highly Diversified |\n"
            "| 15â€“25%    | đźźˇ Moderately Concentrated |\n"
            "| 25â€“50%    | đźź  Concentrated |\n"
            "| > 50%     | đź”´ Highly Concentrated |"
        )

    st.divider()

    # â”€â”€ 2. Weight allocation + Rolling vol â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Weight Allocation")
        # Sorted bar chart instead of pie â€” easier to read with many holdings
        wt_sorted = wt[valid_tickers].sort_values(ascending=True)
        fig_bar_w = go.Figure(go.Bar(
            x=wt_sorted.values, y=wt_sorted.index, orientation="h",
            text=[f"{v:.1%}" for v in wt_sorted.values],
            textposition="auto",
            marker_color=px.colors.qualitative.Plotly[:len(wt_sorted)],
        ))
        fig_bar_w.update_layout(
            xaxis_tickformat=".0%", xaxis_title="Weight",
            margin=dict(l=20, r=20, t=20, b=40),
            height=max(300, len(wt_sorted) * 36),
        )
        st.plotly_chart(fig_bar_w, width='stretch')

    with col_b:
        st.subheader("Rolling 60-Day Volatility")
        rv  = port_ret.rolling(60).std() * np.sqrt(252)
        brv = bench_ret.rolling(60).std() * np.sqrt(252)
        fig_rv = go.Figure()
        fig_rv.add_trace(go.Scatter(x=rv.index, y=rv.values, name="Portfolio"))
        fig_rv.add_trace(go.Scatter(x=brv.index, y=brv.values,
                                    name="Benchmark", line=dict(dash="dot")))
        if comp_ret is not None:
            crv = comp_ret.rolling(60).std() * np.sqrt(252)
            fig_rv.add_trace(go.Scatter(x=crv.index, y=crv.values,
                                        name=comp_name,
                                        line=dict(color="#00CC96", dash="dash")))
        fig_rv.update_layout(yaxis_tickformat=".0%",
                             height=max(300, len(wt_sorted) * 36),
                             margin=dict(l=40, r=20, t=20, b=40))
        st.plotly_chart(fig_rv, width='stretch')

    st.divider()

    # â”€â”€ 3. Sector Exposure (NEW) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.subheader("Sector Exposure")
    if sector_map:
        # Build a DataFrame: ticker â†’ weight â†’ sector
        sector_df = pd.DataFrame({
            "Ticker": valid_tickers,
            "Weight": wt[valid_tickers].values,
            "Sector": [sector_map.get(t, {}).get("sector", "Unknown") for t in valid_tickers],
            "Industry": [sector_map.get(t, {}).get("industry", "Unknown") for t in valid_tickers],
        })

        # Aggregate weight by sector
        sector_agg = sector_df.groupby("Sector", as_index=False)["Weight"].sum()
        sector_agg = sector_agg.sort_values("Weight", ascending=False)

        sec_col1, sec_col2 = st.columns([1, 1])

        with sec_col1:
            # Donut chart
            fig_sector = px.pie(
                sector_agg, names="Sector", values="Weight",
                hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_sector.update_traces(
                textposition="inside", textinfo="percent+label",
                hovertemplate="<b>%{label}</b><br>Weight: %{value:.1%}<extra></extra>",
            )
            fig_sector.update_layout(
                margin=dict(l=20, r=20, t=20, b=20), height=400,
                showlegend=False,
            )
            st.plotly_chart(fig_sector, width='stretch')

        with sec_col2:
            # Detailed breakdown table
            st.markdown("**Sector Breakdown**")
            sec_display = sector_agg.copy()
            sec_display["Weight"] = sec_display["Weight"].map("{:.1%}".format)
            st.dataframe(sec_display, width='stretch', hide_index=True)

            # Per-ticker detail
            st.markdown("**Holdings by Sector**")
            detail = sector_df[["Ticker", "Sector", "Industry", "Weight"]].copy()
            detail = detail.sort_values(["Sector", "Weight"], ascending=[True, False])
            detail["Weight"] = detail["Weight"].map("{:.1%}".format)
            st.dataframe(detail, width='stretch', hide_index=True)

        # Concentration insight
        top_sector = sector_agg.iloc[0]
        n_sectors = len(sector_agg)
        if top_sector["Weight"] > 0.5:
            st.warning(
                f"âš ď¸Ź **{top_sector['Sector']}** accounts for **{top_sector['Weight']:.1%}** "
                f"of your portfolio â€” over half your exposure is in a single sector. "
                f"Consider diversifying across more sectors."
            )
        elif n_sectors <= 2:
            st.info(
                f"đź“Š Your portfolio spans only **{n_sectors} sector(s)**. "
                "Broader sector exposure can help reduce concentration risk."
            )
        else:
            st.success(
                f"âś… Your portfolio spans **{n_sectors} sectors**, "
                f"with the largest (**{top_sector['Sector']}**) at **{top_sector['Weight']:.1%}**."
            )
    else:
        st.info("Sector data not available. Re-run analysis to fetch sector information.")

    st.divider()

    # â”€â”€ 4. Covariance matrix (PyPortfolioOpt) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.subheader("Annualized Covariance Matrix")

    asset_prices = prices[valid_tickers]
    if HAS_RISK_MODELS:
        st.caption("Computed with `pypfopt.risk_models.CovarianceShrinkage` (Ledoit-Wolf) â€” "
                   "a robust estimator that reduces estimation noise.")
        try:
            cov_matrix = risk_models.CovarianceShrinkage(asset_prices).ledoit_wolf()
        except Exception:
            cov_matrix = risk_models.sample_cov(asset_prices)
    else:
        st.caption("Computed from sample returns (annualized). "
                   "Install `pyportfolioopt` and `scikit-learn` for the Ledoit-Wolf shrinkage estimator.")
        cov_matrix = asset_prices.pct_change().dropna().cov() * 252

    fig_cov = go.Figure(data=go.Heatmap(
        z=cov_matrix.values,
        x=cov_matrix.columns, y=cov_matrix.index,
        colorscale="Blues", zmid=None,
        text=np.round(cov_matrix.values, 4), texttemplate="%{text}",
        hovertemplate="%{y} / %{x}<br>Cov: %{z:.4f}<extra></extra>",
    ))
    fig_cov.update_layout(height=max(350, len(valid_tickers) * 55),
                          margin=dict(l=40, r=20, t=20, b=40))
    st.plotly_chart(fig_cov, width='stretch')

    # Also show correlation
    st.subheader("Asset Correlation Matrix")
    corr = ind_ret[valid_tickers].corr()
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu_r", zmid=0,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        zmin=-1, zmax=1))
    fig_corr.update_layout(height=max(350, len(valid_tickers) * 55),
                           margin=dict(l=40, r=20, t=20, b=40))
    st.plotly_chart(fig_corr, width='stretch')

    st.divider()

    # â”€â”€ 5. Marginal Contribution to Risk (MCR) â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.subheader("Risk Contribution by Asset")
    st.caption(
        "**Marginal Contribution to Risk (MCR)** = how much each extra dollar in "
        "an asset increases portfolio volatility. **% of Risk** = what share of "
        "total portfolio risk each asset is responsible for."
    )

    cov_np = cov_matrix.values
    port_var = w_arr @ cov_np @ w_arr
    port_vol_ann = np.sqrt(port_var)
    mcr = (cov_np @ w_arr) / port_vol_ann                  # marginal contribution
    ctr = w_arr * mcr                                       # component contribution
    pct_ctr = ctr / ctr.sum()                               # percentage contribution

    risk_df = pd.DataFrame({
        "Ticker": valid_tickers,
        "Weight": w_arr,
        "MCR": mcr,
        "Risk Contribution": ctr,
        "% of Risk": pct_ctr,
    }).sort_values("% of Risk", ascending=True)

    # Bar chart â€” weight vs risk contribution side by side
    fig_mcr = go.Figure()
    fig_mcr.add_trace(go.Bar(
        y=risk_df["Ticker"], x=risk_df["Weight"], name="Weight",
        orientation="h", marker_color="#636EFA",
        text=[f"{v:.1%}" for v in risk_df["Weight"]], textposition="auto",
    ))
    fig_mcr.add_trace(go.Bar(
        y=risk_df["Ticker"], x=risk_df["% of Risk"], name="% of Risk",
        orientation="h", marker_color="#EF553B",
        text=[f"{v:.1%}" for v in risk_df["% of Risk"]], textposition="auto",
    ))
    fig_mcr.update_layout(
        barmode="group", xaxis_tickformat=".0%",
        xaxis_title="Proportion",
        legend=dict(orientation="h", y=1.02, x=0),
        height=max(320, len(risk_df) * 45),
        margin=dict(l=20, r=20, t=40, b=40),
    )
    st.plotly_chart(fig_mcr, width='stretch')

    # Detail table
    risk_display = risk_df.copy().sort_values("% of Risk", ascending=False)
    risk_display["Weight"]     = risk_display["Weight"].map("{:.2%}".format)
    risk_display["MCR"]        = risk_display["MCR"].map("{:.4f}".format)
    risk_display["Risk Contribution"] = risk_display["Risk Contribution"].map("{:.4f}".format)
    risk_display["% of Risk"]  = risk_display["% of Risk"].map("{:.1%}".format)
    st.dataframe(risk_display, width='stretch', hide_index=True)

    # Key insight
    top_risk = risk_df.sort_values("% of Risk", ascending=False).iloc[0]
    st.info(
        f"đź’ˇ **{top_risk['Ticker']}** contributes **{top_risk['% of Risk']:.1%}** "
        f"of total portfolio risk while having a **{top_risk['Weight']:.1%}** weight. "
        + ("This is disproportionately high â€” consider reducing exposure."
           if top_risk["% of Risk"] > top_risk["Weight"] * 1.5
           else "This is roughly proportional to its weight.")
    )

    st.metric("Portfolio Volatility (annualized, from cov matrix)", f"{port_vol_ann:.2%}")


# ==================== TAB 3 â€” FACTOR EXPOSURE (Fama-French 3-Factor) ====================
with tab_factor:
    # â”€â”€ Single-factor (CAPM) â€” always available â”€â”€â”€â”€â”€â”€â”€â”€
    st.subheader("Single-Factor Model (CAPM vs Benchmark)")

    from scipy import stats as sp_stats
    aligned_sf = pd.concat([port_ret, bench_ret], axis=1).dropna()
    slope_sf, intercept_sf, r_sf, p_sf, se_sf = sp_stats.linregress(
        aligned_sf.iloc[:, 1], aligned_sf.iloc[:, 0])

    sf1, sf2, sf3, sf4 = st.columns(4)
    sf1.metric("Beta (vs benchmark)", f"{slope_sf:.3f}")
    sf2.metric("Alpha (daily)", f"{intercept_sf:.4%}")
    sf3.metric("RÂ˛", f"{r_sf**2:.2%}")
    sf4.metric("Tracking Error",
               f"{(aligned_sf.iloc[:, 0] - aligned_sf.iloc[:, 1]).std() * np.sqrt(252):.2%}")

    fig_sf = go.Figure(data=go.Scatter(
        x=aligned_sf.iloc[:, 1], y=aligned_sf.iloc[:, 0],
        mode="markers", marker=dict(size=3, opacity=0.35, color="#636EFA")))
    xr = np.linspace(aligned_sf.iloc[:, 1].min(), aligned_sf.iloc[:, 1].max(), 100)
    fig_sf.add_trace(go.Scatter(x=xr, y=intercept_sf + slope_sf * xr, mode="lines",
                                name=f"Î˛ = {slope_sf:.3f}",
                                line=dict(color="red", width=2)))
    fig_sf.update_layout(xaxis_title=f"Benchmark ({bench}) Daily Return",
                         yaxis_title="Portfolio Daily Return",
                         xaxis_tickformat=".1%", yaxis_tickformat=".1%",
                         height=400, margin=dict(l=50, r=20, t=30, b=40),
                         showlegend=True,
                         legend=dict(orientation="h", y=1.02, x=0))
    st.plotly_chart(fig_sf, width='stretch')

    st.divider()

    # â”€â”€ Fama-French 3-Factor Model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.subheader("Fama-French 3-Factor Model")
    st.caption(
        "Regresses portfolio excess returns against three systematic risk factors "
        "from Kenneth French's data library: **Mkt-RF** (market risk premium), "
        "**SMB** (Small Minus Big â€” size factor), and **HML** (High Minus Low â€” value factor)."
    )

    if not HAS_STATSMODELS:
        st.warning(
            "`statsmodels` is not installed, so Fama-French regression is unavailable. "
            "Add `statsmodels` to `requirements.txt` and redeploy."
        )
    elif ff_factors is None or ff_factors.empty:
        st.warning("Fama-French factors could not be downloaded. "
                   "Check your internet connection and try again.")
    else:
        # Align portfolio returns with FF factors
        ff_common = port_ret.index.intersection(ff_factors.index)

        if len(ff_common) < 30:
            st.warning(f"Only {len(ff_common)} overlapping dates with FF data â€” "
                       "need at least 30 for a meaningful regression.")
        else:
            pr_ff = port_ret.loc[ff_common]
            factors = ff_factors.loc[ff_common]

            # Excess return = portfolio return - risk-free rate
            y = pr_ff - factors["RF"]
            X = factors[["Mkt-RF", "SMB", "HML"]]
            X = sm_api.add_constant(X)

            model = sm_api.OLS(y, X).fit()
            params = model.params
            pvals  = model.pvalues
            ci     = model.conf_int(alpha=0.05)
            ci.columns = ["Lower 95%", "Upper 95%"]

            alpha_daily = params["const"]
            alpha_annual = (1 + alpha_daily) ** 252 - 1
            mkt_beta = params["Mkt-RF"]
            smb_beta = params["SMB"]
            hml_beta = params["HML"]

            # â”€â”€ Metric cards â”€â”€
            st.markdown("##### Factor Loadings")
            ff1, ff2, ff3, ff4 = st.columns(4)
            ff1.metric(
                "Alpha (annualized)", f"{alpha_annual:.2%}",
                help=f"Daily alpha: {alpha_daily:.4%} | "
                     f"p-value: {pvals['const']:.4f}",
            )
            ff2.metric(
                "Market Beta (Mkt-RF)", f"{mkt_beta:.3f}",
                help=f"p-value: {pvals['Mkt-RF']:.4f} | "
                     f"95% CI: [{ci.loc['Mkt-RF', 'Lower 95%']:.3f}, {ci.loc['Mkt-RF', 'Upper 95%']:.3f}]",
            )
            ff3.metric(
                "Size Beta (SMB)", f"{smb_beta:.3f}",
                help=f"p-value: {pvals['SMB']:.4f} | "
                     f"Positive = small-cap tilt, Negative = large-cap tilt",
            )
            ff4.metric(
                "Value Beta (HML)", f"{hml_beta:.3f}",
                help=f"p-value: {pvals['HML']:.4f} | "
                     f"Positive = value tilt, Negative = growth tilt",
            )

            ff5, ff6, ff7, ff8 = st.columns(4)
            ff5.metric("RÂ˛", f"{model.rsquared:.2%}")
            ff6.metric("Adj. RÂ˛", f"{model.rsquared_adj:.2%}")
            ff7.metric("Observations", f"{int(model.nobs):,}")
            ff8.metric("Residual Vol (ann.)",
                       f"{model.resid.std() * np.sqrt(252):.2%}",
                       help="Idiosyncratic risk â€” the volatility not explained by the three factors.")

            st.divider()

            # â”€â”€ Factor loadings bar chart â”€â”€
            st.markdown("##### Factor Loadings â€” Visual")
            betas = pd.DataFrame({
                "Factor": ["Mkt-RF\n(Market)", "SMB\n(Size)", "HML\n(Value)"],
                "Beta": [mkt_beta, smb_beta, hml_beta],
                "Lower": [ci.loc["Mkt-RF", "Lower 95%"], ci.loc["SMB", "Lower 95%"], ci.loc["HML", "Lower 95%"]],
                "Upper": [ci.loc["Mkt-RF", "Upper 95%"], ci.loc["SMB", "Upper 95%"], ci.loc["HML", "Upper 95%"]],
                "p-value": [pvals["Mkt-RF"], pvals["SMB"], pvals["HML"]],
            })
            # Color: significant = bold blue/red, insignificant = gray
            colors = []
            for _, row in betas.iterrows():
                if row["p-value"] < 0.05:
                    colors.append("#636EFA" if row["Beta"] >= 0 else "#EF553B")
                else:
                    colors.append("#CCCCCC")

            fig_betas = go.Figure()
            fig_betas.add_trace(go.Bar(
                y=betas["Factor"], x=betas["Beta"], orientation="h",
                marker_color=colors,
                text=[f"{b:.3f}" for b in betas["Beta"]],
                textposition="outside",
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=(betas["Upper"] - betas["Beta"]).tolist(),
                    arrayminus=(betas["Beta"] - betas["Lower"]).tolist(),
                    color="#888", thickness=1.5, width=6,
                ),
            ))
            fig_betas.add_vline(x=0, line_dash="dash", line_color="#666", line_width=1)
            fig_betas.update_layout(
                xaxis_title="Factor Loading (Î˛)",
                height=280,
                margin=dict(l=100, r=60, t=20, b=40),
            )
            st.plotly_chart(fig_betas, width='stretch')

            st.caption(
                "**Colored bars** are statistically significant at the 5% level. "
                "**Gray bars** are not significant â€” the loading may be due to noise. "
                "Error bars show the 95% confidence interval."
            )

            st.divider()

            # â”€â”€ Interpretation helper â”€â”€
            st.markdown("##### What Do These Factors Mean?")
            interp_lines = []
            if pvals["Mkt-RF"] < 0.05:
                if mkt_beta > 1.05:
                    interp_lines.append(
                        f"đź”´ **Mkt-RF = {mkt_beta:.2f}**: Your portfolio is **aggressive** â€” "
                        "it amplifies market moves. A 1% market gain gives you ~{:.2f}% gain, "
                        "but losses are amplified too.".format(mkt_beta))
                elif mkt_beta < 0.95:
                    interp_lines.append(
                        f"đźź˘ **Mkt-RF = {mkt_beta:.2f}**: Your portfolio is **defensive** â€” "
                        "it dampens market swings.")
                else:
                    interp_lines.append(
                        f"âšŞ **Mkt-RF = {mkt_beta:.2f}**: Your portfolio moves roughly in line "
                        "with the market.")

            if pvals["SMB"] < 0.05:
                tilt = "small-cap" if smb_beta > 0 else "large-cap"
                interp_lines.append(
                    f"đź“ **SMB = {smb_beta:.2f}**: Significant **{tilt} tilt**. "
                    + ("Your returns benefit when smaller companies outperform." if smb_beta > 0
                       else "Your returns benefit when larger companies outperform."))
            else:
                interp_lines.append(
                    f"đź“ **SMB = {smb_beta:.2f}**: No significant size tilt (p = {pvals['SMB']:.2f}).")

            if pvals["HML"] < 0.05:
                tilt = "value" if hml_beta > 0 else "growth"
                interp_lines.append(
                    f"đź“Š **HML = {hml_beta:.2f}**: Significant **{tilt} tilt**. "
                    + ("Your returns benefit when cheaper (value) stocks outperform." if hml_beta > 0
                       else "Your returns benefit when expensive (growth) stocks outperform."))
            else:
                interp_lines.append(
                    f"đź“Š **HML = {hml_beta:.2f}**: No significant value/growth tilt (p = {pvals['HML']:.2f}).")

            if pvals["const"] < 0.05 and alpha_annual > 0:
                interp_lines.append(
                    f"âś¨ **Alpha = {alpha_annual:.2%}/yr**: Statistically significant positive alpha! "
                    "Your portfolio generates returns not explained by market, size, or value factors.")
            elif pvals["const"] < 0.05 and alpha_annual < 0:
                interp_lines.append(
                    f"âš ď¸Ź **Alpha = {alpha_annual:.2%}/yr**: Statistically significant negative alpha. "
                    "After accounting for factor exposures, the portfolio underperforms.")

            for line in interp_lines:
                st.markdown(line)

            st.divider()

            # â”€â”€ Full regression table â”€â”€
            st.markdown("##### Full Regression Output")
            reg_table = pd.DataFrame({
                "Coefficient": params.values,
                "Std Error": model.bse.values,
                "t-stat": model.tvalues.values,
                "p-value": pvals.values,
                "95% CI Lower": ci["Lower 95%"].values,
                "95% CI Upper": ci["Upper 95%"].values,
            }, index=["Alpha (const)", "Mkt-RF", "SMB", "HML"])

            # Format
            reg_display = reg_table.copy()
            reg_display["Coefficient"] = reg_display["Coefficient"].map("{:.6f}".format)
            reg_display["Std Error"]   = reg_display["Std Error"].map("{:.6f}".format)
            reg_display["t-stat"]      = reg_display["t-stat"].map("{:.3f}".format)
            reg_display["p-value"]     = reg_display["p-value"].map("{:.4f}".format)
            reg_display["95% CI Lower"] = reg_display["95% CI Lower"].map("{:.4f}".format)
            reg_display["95% CI Upper"] = reg_display["95% CI Upper"].map("{:.4f}".format)
            st.dataframe(reg_display, width='stretch')

            st.divider()

            # Factor Return Attribution (JP Morgan-style)
            st.markdown("##### Factor Return Attribution")
            st.caption(
                "Breaks annualized portfolio excess return into contributions from "
                "Alpha, Market (Mkt-RF), Size (SMB), and Value (HML)."
            )

            mkt_ann = annualized_return(factors["Mkt-RF"])
            smb_ann = annualized_return(factors["SMB"])
            hml_ann = annualized_return(factors["HML"])
            port_excess_ann = annualized_return(y)

            market_contrib = mkt_beta * mkt_ann
            size_contrib = smb_beta * smb_ann
            value_contrib = hml_beta * hml_ann
            alpha_contrib = alpha_annual
            total_excess_model = alpha_contrib + market_contrib + size_contrib + value_contrib

            wf_labels = [
                "Alpha",
                "Market Risk",
                "Size Factor",
                "Value Factor",
                "Total Excess Return",
            ]
            wf_measures = ["relative", "relative", "relative", "relative", "total"]
            wf_values = [alpha_contrib, market_contrib, size_contrib, value_contrib, 0.0]
            wf_text = [
                f"{alpha_contrib:+.2%}",
                f"{market_contrib:+.2%}",
                f"{size_contrib:+.2%}",
                f"{value_contrib:+.2%}",
                f"{total_excess_model:+.2%}",
            ]

            fig_attr = go.Figure(go.Waterfall(
                x=wf_labels,
                measure=wf_measures,
                y=wf_values,
                text=wf_text,
                textposition="outside",
                connector={"line": {"color": "#777"}},
                increasing={"marker": {"color": "#2CA02C"}},
                decreasing={"marker": {"color": "#D62728"}},
                totals={"marker": {"color": "#1F77B4"}},
            ))
            fig_attr.update_layout(
                height=430,
                margin=dict(l=40, r=20, t=20, b=40),
                yaxis_title="Annualized Contribution",
                yaxis_tickformat="+.1%",
            )
            st.plotly_chart(fig_attr, width='stretch')

            attr_df = pd.DataFrame({
                "Factor": [
                    "Alpha (Idiosyncratic)",
                    "Market (Mkt-RF)",
                    "Size (SMB)",
                    "Value (HML)",
                    "Total Excess Return (Model)",
                    "Portfolio Excess Return (Actual)",
                ],
                "Beta (Sensitivity)": [
                    np.nan,
                    mkt_beta,
                    smb_beta,
                    hml_beta,
                    np.nan,
                    np.nan,
                ],
                "Factor Return (Ann.)": [
                    np.nan,
                    mkt_ann,
                    smb_ann,
                    hml_ann,
                    np.nan,
                    np.nan,
                ],
                "Contribution to Portfolio (Ann.)": [
                    alpha_contrib,
                    market_contrib,
                    size_contrib,
                    value_contrib,
                    total_excess_model,
                    port_excess_ann,
                ],
            })

            attr_display = attr_df.copy()
            attr_display["Beta (Sensitivity)"] = attr_display["Beta (Sensitivity)"].map(
                lambda v: "â€”" if pd.isna(v) else f"{v:.3f}"
            )
            attr_display["Factor Return (Ann.)"] = attr_display["Factor Return (Ann.)"].map(
                lambda v: "â€”" if pd.isna(v) else f"{v:+.2%}"
            )
            attr_display["Contribution to Portfolio (Ann.)"] = attr_display[
                "Contribution to Portfolio (Ann.)"
            ].map(lambda v: f"{v:+.2%}")
            st.dataframe(attr_display, width='stretch', hide_index=True)

            gap = total_excess_model - port_excess_ann
            st.caption(
                f"Verification: model-implied excess return = **{total_excess_model:+.2%}**, "
                f"actual portfolio excess return = **{port_excess_ann:+.2%}** "
                f"(gap: **{gap:+.2%}**)."
            )


# ==================== TAB 4 â€” DRAWDOWNS & TAIL RISK ====================
with tab_dd:

    # â”€â”€ 1. Key risk metrics row â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.subheader("Drawdown & Tail-Risk Metrics")

    dd_series   = qs.stats.to_drawdown_series(port_ret)
    dd_bench    = qs.stats.to_drawdown_series(bench_ret)
    mdd_val     = qs.stats.max_drawdown(port_ret)
    mdd_bench   = qs.stats.max_drawdown(bench_ret)
    cvar_95     = qs.stats.cvar(port_ret, confidence=0.95)
    cvar_99     = qs.stats.cvar(port_ret, confidence=0.99)
    var_95      = qs.stats.value_at_risk(port_ret, confidence=0.95)
    skew_val    = qs.stats.skew(port_ret)
    kurt_val    = qs.stats.kurtosis(port_ret)
    calmar_val  = qs.stats.calmar(port_ret)
    ulcer_val   = qs.stats.ulcer_index(port_ret)
    recov_val   = qs.stats.recovery_factor(port_ret)
    tail_ratio  = qs.stats.tail_ratio(port_ret)

    dr1, dr2, dr3, dr4 = st.columns(4)
    dr1.metric("Max Drawdown", f"{mdd_val:.2%}",
               delta=f"{mdd_val - mdd_bench:+.2%} vs {bench}",
               delta_color="inverse",
               help="Largest peak-to-trough decline in portfolio value.")
    dr2.metric("CVaR (95%)", f"{cvar_95:.2%}",
               help="Expected Shortfall â€” average loss on the worst 5% of days.")
    dr3.metric("VaR (95%)", f"{var_95:.2%}",
               help="Value at Risk â€” threshold below which the worst 5% of days fall.")
    dr4.metric("Calmar Ratio", f"{calmar_val:.2f}",
               help="CAGR Ă· Max Drawdown. Higher = better risk-adjusted returns.")

    # CVaR plain-English explanation
    st.info(
        f"đź“‰ **On the worst 5% of trading days, your average loss is "
        f"{abs(cvar_95):.2%}.** This means roughly once a month you can expect "
        f"a daily loss of at least {abs(var_95):.2%}, and when those bad days "
        f"happen, the average hit is {abs(cvar_95):.2%}."
    )

    st.divider()

    # â”€â”€ 2. Underwater chart â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.subheader("Underwater Chart")
    st.caption("Shows how far the portfolio is below its all-time high at every point in time.")

    fig_uw = go.Figure()
    fig_uw.add_trace(go.Scatter(
        x=dd_series.index, y=dd_series.values,
        fill="tozeroy", fillcolor="rgba(220,50,50,0.18)",
        line=dict(color="#DC3232", width=1), name="Portfolio",
        hovertemplate="%{x|%b %d, %Y}<br>Drawdown: %{y:.2%}<extra></extra>",
    ))
    fig_uw.add_trace(go.Scatter(
        x=dd_bench.index, y=dd_bench.values,
        line=dict(color="#888", width=1, dash="dot"), name=f"Benchmark ({bench})",
        hovertemplate="%{x|%b %d, %Y}<br>Drawdown: %{y:.2%}<extra></extra>",
    ))
    if comp_ret is not None:
        dd_comp = qs.stats.to_drawdown_series(comp_ret)
        fig_uw.add_trace(go.Scatter(
            x=dd_comp.index, y=dd_comp.values,
            line=dict(color="#00CC96", width=1, dash="dash"), name=comp_name,
        ))
    fig_uw.update_layout(
        yaxis_tickformat=".0%", yaxis_title="Drawdown from Peak",
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=50, r=20, t=40, b=40), height=420,
        hovermode="x unified",
    )
    st.plotly_chart(fig_uw, width='stretch')

    st.divider()

    # â”€â”€ 3. Worst drawdown periods (quantstats) â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.subheader("Worst Drawdown Periods")

    dd_details = qs.stats.drawdown_details(dd_series)
    if not dd_details.empty:
        # Sort by worst drawdown, take top 10
        dd_details = dd_details.sort_values("max drawdown").head(10).reset_index(drop=True)
        dd_table = pd.DataFrame({
            "Rank": range(1, len(dd_details) + 1),
            "Start":    dd_details["start"].astype(str),
            "Trough":   dd_details["valley"].astype(str),
            "Recovery":  dd_details["end"].apply(lambda x: str(x) if pd.notna(x) and str(x) != "" else "Ongoing"),
            "Max DD":   dd_details["max drawdown"].apply(lambda x: f"{x / 100:.2%}"),
            "Duration (days)": dd_details["days"].astype(int),
        })
        st.dataframe(dd_table, width='stretch', hide_index=True)

        # Drawdown periods visualization
        st.subheader("Top 5 Drawdowns Highlighted")
        top5 = dd_details.head(5)
        fig_top = go.Figure()
        fig_top.add_trace(go.Scatter(
            x=dd_series.index, y=dd_series.values,
            line=dict(color="#ccc", width=1), name="All drawdowns",
            showlegend=False,
        ))
        colors = ["#DC3232", "#E8553B", "#F07848", "#F49D6E", "#F7C49A"]
        for i, (_, row) in enumerate(top5.iterrows()):
            s = pd.Timestamp(row["start"])
            e = pd.Timestamp(row["end"]) if pd.notna(row["end"]) and str(row["end"]) != "" else dd_series.index[-1]
            mask = (dd_series.index >= s) & (dd_series.index <= e)
            seg = dd_series[mask]
            fig_top.add_trace(go.Scatter(
                x=seg.index, y=seg.values,
                fill="tozeroy", fillcolor=f"rgba({','.join(str(int(colors[i][j:j+2], 16)) for j in (1, 3, 5))},0.25)",
                line=dict(color=colors[i], width=2),
                name=f"#{i + 1}: {row['max drawdown'] / 100:.1%}",
            ))
        fig_top.update_layout(
            yaxis_tickformat=".0%", yaxis_title="Drawdown",
            legend=dict(orientation="h", y=1.06, x=0),
            margin=dict(l=50, r=20, t=50, b=40), height=400,
        )
        st.plotly_chart(fig_top, width='stretch')

    st.divider()

    # â”€â”€ 4. Tail risk deep dive â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.subheader("Tail Risk Analysis")

    tc1, tc2, tc3 = st.columns(3)

    # Skewness
    with tc1:
        st.metric("Skewness", f"{skew_val:.3f}")
        if skew_val < -0.5:
            st.caption("âš ď¸Ź **Negatively skewed** â€” heavy left tail, "
                       "meaning extreme losses are more common than extreme gains.")
        elif skew_val > 0.5:
            st.caption("âś… **Positively skewed** â€” the right tail is heavier, "
                       "meaning extreme gains are more frequent than extreme losses.")
        else:
            st.caption("â†”ď¸Ź **Roughly symmetric** â€” gains and losses are "
                       "similarly distributed.")

    # Kurtosis
    with tc2:
        st.metric("Excess Kurtosis", f"{kurt_val:.3f}")
        if kurt_val > 1:
            st.caption("âš ď¸Ź **Leptokurtic** (fat tails) â€” extreme moves in "
                       "either direction happen more often than a normal "
                       "distribution would predict.")
        elif kurt_val < -1:
            st.caption("âś… **Platykurtic** (thin tails) â€” extreme moves "
                       "are less common than normal.")
        else:
            st.caption("â†”ď¸Ź **Near-normal tails** â€” tail behavior is close "
                       "to what a Gaussian distribution would predict.")

    # Tail ratio
    with tc3:
        st.metric("Tail Ratio", f"{tail_ratio:.2f}",
                  help="Right tail (gains) Ă· left tail (losses) at the 95th percentile. "
                       "> 1 = fatter right tail (good).")
        if tail_ratio > 1.0:
            st.caption("âś… **Right tail dominant** â€” extreme gains tend to "
                       "outsize extreme losses.")
        else:
            st.caption("âš ď¸Ź **Left tail dominant** â€” extreme losses tend to "
                       "outsize extreme gains.")

    st.divider()

    # â”€â”€ 5. Return distribution analysis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.subheader("Return Distribution Analysis")
    st.caption(
        "Overlayed daily return distributions for your portfolio and benchmark. "
        "Fatter tails indicate a higher frequency of extreme outcomes."
    )

    fig_dd_dist = go.Figure()
    fig_dd_dist.add_trace(go.Histogram(
        x=port_ret.values,
        name="Portfolio",
        nbinsx=90,
        histnorm="probability density",
        opacity=0.7,
        marker_color="rgba(99,110,250,0.7)",
    ))
    fig_dd_dist.add_trace(go.Histogram(
        x=bench_ret.values,
        name=f"Benchmark ({bench})",
        nbinsx=90,
        histnorm="probability density",
        opacity=0.7,
        marker_color="rgba(239,85,59,0.7)",
    ))
    fig_dd_dist.update_layout(
        barmode="overlay",
        xaxis_title="Daily Return",
        yaxis_title="Probability Density",
        xaxis_tickformat=".1%",
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=40, r=20, t=20, b=40),
        height=360,
    )
    st.plotly_chart(fig_dd_dist, width='stretch')

    st.divider()

    # â”€â”€ 6. Extended risk statistics table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.subheader("Comprehensive Risk Statistics")

    risk_stats = {
        "Metric": [
            "Max Drawdown", "Avg. Drawdown",
            "VaR (95%)", "VaR (99%)",
            "CVaR (95%)", "CVaR (99%)",
            "Skewness", "Excess Kurtosis", "Tail Ratio",
            "Calmar Ratio", "Ulcer Index", "Recovery Factor",
            "Gain/Pain Ratio",
        ],
        "Portfolio": [
            f"{mdd_val:.2%}",
            f"{dd_series[dd_series < 0].mean():.2%}" if (dd_series < 0).any() else "0.00%",
            f"{var_95:.2%}",
            f"{qs.stats.value_at_risk(port_ret, confidence=0.99):.2%}",
            f"{cvar_95:.2%}",
            f"{cvar_99:.2%}",
            f"{skew_val:.3f}",
            f"{kurt_val:.3f}",
            f"{tail_ratio:.2f}",
            f"{calmar_val:.2f}",
            f"{ulcer_val:.4f}",
            f"{recov_val:.2f}",
            f"{qs.stats.gain_to_pain_ratio(port_ret):.2f}",
        ],
        f"Benchmark ({bench})": [
            f"{mdd_bench:.2%}",
            f"{dd_bench[dd_bench < 0].mean():.2%}" if (dd_bench < 0).any() else "0.00%",
            f"{qs.stats.value_at_risk(bench_ret, confidence=0.95):.2%}",
            f"{qs.stats.value_at_risk(bench_ret, confidence=0.99):.2%}",
            f"{qs.stats.cvar(bench_ret, confidence=0.95):.2%}",
            f"{qs.stats.cvar(bench_ret, confidence=0.99):.2%}",
            f"{qs.stats.skew(bench_ret):.3f}",
            f"{qs.stats.kurtosis(bench_ret):.3f}",
            f"{qs.stats.tail_ratio(bench_ret):.2f}",
            f"{qs.stats.calmar(bench_ret):.2f}",
            f"{qs.stats.ulcer_index(bench_ret):.4f}",
            f"{qs.stats.recovery_factor(bench_ret):.2f}",
            f"{qs.stats.gain_to_pain_ratio(bench_ret):.2f}",
        ],
    }
    st.dataframe(pd.DataFrame(risk_stats).set_index("Metric"), width='stretch')


# ==================== TAB 5 â€” STRESS TESTS ====================
with tab_stress:
    st.subheader("Historical Stress Test Scenarios")
    st.caption("Uses separately downloaded data back to 2007.")

    scenarios = {
        "Global Financial Crisis (Oct 2007 â€“ Mar 2009)": ("2007-10-09", "2009-03-09"),
        "GFC â€” Lehman Phase (Sep â€“ Nov 2008)":           ("2008-09-12", "2008-11-20"),
        "European Debt Crisis (Apr â€“ Oct 2011)":         ("2011-04-29", "2011-10-03"),
        "China Deval. / Oil Crash (Aug 2015 â€“ Feb 2016)": ("2015-08-10", "2016-02-11"),
        "Vol-mageddon (Jan â€“ Feb 2018)":                 ("2018-01-26", "2018-02-08"),
        "Q4 2018 Selloff (Sep â€“ Dec 2018)":              ("2018-09-20", "2018-12-24"),
        "COVID Crash (Feb â€“ Mar 2020)":                  ("2020-02-19", "2020-03-23"),
        "2022 Rate Shock (Jan â€“ Oct 2022)":              ("2022-01-03", "2022-10-12"),
        "2023 Banking Crisis â€” SVB (Mar 2023)":          ("2023-03-08", "2023-03-15"),
        "Trump Tariffs Shock (Apr 2025)":                ("2025-04-02", "2025-04-08"),
        "Tariff Escalation (Apr 2 â€“ Apr 21, 2025)":     ("2025-04-02", "2025-04-21"),
    }

    sp = stress_port; sb = stress_bench
    has_stress = len(sp) > 0 and len(sb) > 0
    results = []
    for name, (s, e) in scenarios.items():
        if has_stress:
            mp = (sp.index >= s) & (sp.index <= e)
            mb = (sb.index >= s) & (sb.index <= e)
            if mp.sum() > 0 and mb.sum() > 0:
                pr = (1 + sp.loc[mp]).prod() - 1
                br = (1 + sb.loc[mb]).prod() - 1
                results.append({"Scenario": name, "Portfolio": f"{pr:.2%}",
                                "Benchmark": f"{br:.2%}", "Excess": f"{pr - br:+.2%}"})
                continue
        results.append({"Scenario": name, "Portfolio": "No data",
                        "Benchmark": "â€”", "Excess": "â€”"})
    st.dataframe(pd.DataFrame(results), width='stretch', hide_index=True)

    st.subheader("What-If: Uniform Market Shock")
    shock_pct = st.slider("Simulated benchmark drop (%)", -50, 0, -20, step=1)
    estimated_loss = slope_sf * (shock_pct / 100)
    st.metric(f"Estimated portfolio loss (Î˛ = {slope_sf:.2f})", f"{estimated_loss:.2%}")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# ==================== TAB 6 - OPTIMIZATION ====================
with tab_opt:
    st.subheader("Optimization: Robust Portfolio Construction")
    st.caption(
        "**Max Sharpe** (mean-variance), **Risk Parity (ERC)** (equal risk), "
        "**Black-Litterman** (market-implied robust returns), and "
        "**Min CVaR** (tail-risk minimization at 95%) compared against your current allocation."
    )

    if not HAS_PYPFOPT:
        st.warning(
            "PyPortfolioOpt is not available in this runtime. Ensure `PyPortfolioOpt` is in "
            "`requirements.txt`, redeploy, and check build logs for installation errors."
        )
    elif len(valid_tickers) < 2:
        st.info("Optimization needs at least 2 assets with valid price history.")
    else:
        asset_prices = prices[valid_tickers].copy()
        asset_returns = asset_prices.pct_change().dropna()

        # Reuse covariance from Risk tab, fallback to local estimate.
        try:
            cov_for_opt = cov_matrix.loc[valid_tickers, valid_tickers].copy()
        except Exception:
            if HAS_RISK_MODELS:
                try:
                    cov_for_opt = risk_models.CovarianceShrinkage(asset_prices).ledoit_wolf()
                except Exception:
                    cov_for_opt = asset_returns.cov() * 252
            else:
                cov_for_opt = asset_returns.cov() * 252

        cov_for_opt = cov_for_opt.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        cov_for_opt = 0.5 * (cov_for_opt + cov_for_opt.T)

        def _zero_weights() -> pd.Series:
            return pd.Series(0.0, index=valid_tickers, dtype=float)

        def _normalize_weights(w: pd.Series) -> pd.Series:
            w = w.reindex(valid_tickers).fillna(0.0).astype(float)
            s = float(w.sum())
            return (w / s) if s > 0 else w

        def calculate_equal_risk_contribution(cov_df: pd.DataFrame) -> pd.Series:
            """
            Equal Risk Contribution weights via SLSQP.
            Objective: sum_i ( w_i * (Sigma w)_i - target_risk )^2
            """
            if not HAS_SCIPY_OPT or sp_opt is None:
                raise RuntimeError("scipy.optimize is unavailable for ERC fallback.")

            sigma = np.asarray(cov_df, dtype=float)
            sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
            sigma = 0.5 * (sigma + sigma.T)
            n = sigma.shape[0]

            if n == 1:
                return pd.Series([1.0], index=valid_tickers, dtype=float)

            x0 = np.repeat(1.0 / n, n)
            bounds = [(0.0, 1.0)] * n
            constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

            def erc_objective(w):
                w = np.asarray(w, dtype=float)
                sigma_w = sigma @ w
                rc = w * sigma_w
                target_risk = float((w @ sigma_w) / n)
                return float(np.sum((rc - target_risk) ** 2))

            res = sp_opt.minimize(
                erc_objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 600, "ftol": 1e-12},
            )
            if (not res.success) or np.any(~np.isfinite(res.x)):
                raise RuntimeError(f"ERC optimizer failed: {res.message}")
            return pd.Series(res.x, index=valid_tickers, dtype=float)

        def calculate_min_semivariance_weights(returns_df: pd.DataFrame) -> pd.Series:
            """
            Tail-risk fallback when EfficientCVaR/cvxpy is unavailable.
            Minimizes historical downside semi-variance.
            """
            if not HAS_SCIPY_OPT or sp_opt is None:
                raise RuntimeError("scipy.optimize is unavailable for semi-variance fallback.")

            r = np.asarray(returns_df, dtype=float)
            r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
            n = r.shape[1]
            x0 = np.repeat(1.0 / n, n)
            bounds = [(0.0, 1.0)] * n
            constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

            def semivar_objective(w):
                pr = r @ w
                downside = np.minimum(pr, 0.0)
                return float(np.mean(downside ** 2))

            res = sp_opt.minimize(
                semivar_objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 600, "ftol": 1e-12},
            )
            if (not res.success) or np.any(~np.isfinite(res.x)):
                raise RuntimeError(f"Semi-variance optimizer failed: {res.message}")
            return pd.Series(res.x, index=valid_tickers, dtype=float)

        try:
            from pypfopt import expected_returns as expected_returns_mod
            exp_ret = expected_returns_mod.mean_historical_return(asset_prices, frequency=252)
        except Exception:
            exp_ret = asset_returns.mean() * 252
        exp_ret = pd.Series(exp_ret, index=valid_tickers, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        EfficientFrontier = None
        EfficientCVaR = None
        black_litterman = None
        try:
            from pypfopt.efficient_frontier import EfficientFrontier, EfficientCVaR
        except Exception:
            pass
        try:
            from pypfopt import black_litterman
        except Exception:
            pass

        # 1) Max Sharpe (Mean-Variance)
        max_sharpe_w = _zero_weights()
        if EfficientFrontier is not None:
            try:
                ef = EfficientFrontier(exp_ret, cov_for_opt)
                ef.max_sharpe(risk_free_rate=rf)
                max_sharpe_w = pd.Series(ef.clean_weights(), dtype=float)
            except Exception:
                try:
                    ef = EfficientFrontier(exp_ret, cov_for_opt)
                    ef.min_volatility()
                    max_sharpe_w = pd.Series(ef.clean_weights(), dtype=float)
                except Exception:
                    max_sharpe_w = _zero_weights()

        # 2) Risk Parity / Equal Risk Contribution
        # Flow: Library -> Scipy ERC -> 0.0
        risk_parity_w = _zero_weights()
        try:
            from pypfopt.hierarchical_portfolio import HRPOpt
            try:
                hrp = HRPOpt(returns=asset_returns)
                risk_parity_w = pd.Series(hrp.optimize(), dtype=float)
            except Exception:
                try:
                    risk_parity_w = calculate_equal_risk_contribution(cov_for_opt)
                except Exception:
                    risk_parity_w = _zero_weights()
        except Exception:
            try:
                risk_parity_w = calculate_equal_risk_contribution(cov_for_opt)
            except Exception:
                risk_parity_w = _zero_weights()

        # 3) Black-Litterman (market-implied prior returns only; no views/Q)
        bl_w = _zero_weights()
        if black_litterman is not None and EfficientFrontier is not None:
            try:
                bench_mu_ann = float(bench_ret.mean() * 252)
                bench_var_ann = float(bench_ret.var() * 252)
                delta = 2.5 if bench_var_ann <= 1e-12 else max((bench_mu_ann - rf) / bench_var_ann, 1e-6)

                eq_mcaps = pd.Series(1.0, index=valid_tickers)
                implied_rets = black_litterman.market_implied_prior_returns(
                    eq_mcaps,
                    delta,
                    cov_for_opt,
                    risk_free_rate=rf,
                )

                ef_bl = EfficientFrontier(implied_rets, cov_for_opt)
                ef_bl.max_sharpe(risk_free_rate=rf)
                bl_w = pd.Series(ef_bl.clean_weights(), dtype=float)
            except Exception:
                bl_w = max_sharpe_w.copy()
        else:
            bl_w = max_sharpe_w.copy()

        # 4) Min CVaR with silent fallback
        # Flow: Library -> Scipy Semi-Variance -> 0.0
        tail_col_name = "Min CVaR"
        tail_w = _zero_weights()
        if EfficientCVaR is not None:
            try:
                ecvar = EfficientCVaR(exp_ret, asset_returns)
                ecvar.min_cvar()
                tail_w = pd.Series(ecvar.clean_weights(), dtype=float)
            except Exception:
                try:
                    tail_w = calculate_min_semivariance_weights(asset_returns)
                    tail_col_name = "Min Semi-Variance (Fallback)"
                except Exception:
                    tail_w = _zero_weights()
        else:
            try:
                tail_w = calculate_min_semivariance_weights(asset_returns)
                tail_col_name = "Min Semi-Variance (Fallback)"
            except Exception:
                tail_w = _zero_weights()

        current_w = wt.reindex(valid_tickers).fillna(0.0).astype(float)
        max_sharpe_w = _normalize_weights(max_sharpe_w)
        risk_parity_w = _normalize_weights(risk_parity_w)
        bl_w = _normalize_weights(bl_w)
        tail_w = _normalize_weights(tail_w)

        opt_df = pd.DataFrame({
            "Ticker": valid_tickers,
            "Current Weight": current_w.values,
            "Max Sharpe": max_sharpe_w.values,
            "Risk Parity": risk_parity_w.values,
            "Black-Litterman": bl_w.values,
            tail_col_name: tail_w.values,
        }).sort_values("Current Weight", ascending=False)

        chart_cols = ["Current Weight", "Max Sharpe", "Risk Parity", "Black-Litterman", tail_col_name]
        chart_colors = {
            "Current Weight": "#636EFA",
            "Max Sharpe": "#EF553B",
            "Risk Parity": "#00CC96",
            "Black-Litterman": "#AB63FA",
            tail_col_name: "#FFA15A",
        }

        fig_opt = go.Figure()
        for col in chart_cols:
            fig_opt.add_trace(go.Bar(
                x=opt_df["Ticker"],
                y=opt_df[col],
                name=col,
                marker_color=chart_colors.get(col, "#888888"),
            ))

        fig_opt.update_layout(
            barmode="group",
            xaxis_title="Ticker",
            yaxis_title="Weight",
            yaxis_tickformat=".0%",
            legend=dict(orientation="h", y=1.06, x=0),
            margin=dict(l=40, r=20, t=40, b=40),
            height=max(560, 360 + len(opt_df) * 26),
            hovermode="x unified",
        )
        st.plotly_chart(fig_opt, width='stretch')

        st.markdown("##### Strategy Weights Heatmap")
        weights_matrix = opt_df.set_index("Ticker")[chart_cols]
        st.dataframe(
            weights_matrix.style.format("{:.1%}").background_gradient(cmap="YlGn", axis=1),
            width='stretch',
        )

        dominant = weights_matrix.idxmax(axis=1)
        dom_df = pd.DataFrame({"Ticker": dominant.index, "Dominant Allocation": dominant.values})
        st.dataframe(dom_df, width='stretch', hide_index=True)
# ==================== TAB 7 - MONTE CARLO SIMULATION ====================
with tab_mc:
    st.subheader("Monte Carlo Simulation (Correlated GBM)")

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        sim_count = st.slider("Simulation Count", min_value=500, max_value=5000, value=1000, step=100)
    with mc2:
        horizon_years = st.slider("Forecast Horizon (Years)", min_value=1, max_value=10, value=5, step=1)
    with mc3:
        initial_investment = st.number_input(
            "Initial Investment", min_value=100.0, value=10000.0, step=500.0, format="%.2f"
        )

    if len(valid_tickers) < 1:
        st.warning("No valid assets available for simulation.")
    else:
        try:
            w_mc = wt.reindex(valid_tickers).fillna(0.0).values.astype(float)
            if w_mc.sum() <= 0:
                raise ValueError("Current weights are not valid for simulation.")
            w_mc = w_mc / w_mc.sum()

            mu_ann = ind_ret[valid_tickers].mean().values.astype(float) * 252.0
            cov_ann_df = cov_matrix.loc[valid_tickers, valid_tickers].astype(float).copy()
            cov_ann = cov_ann_df.values
            cov_ann = np.nan_to_num(cov_ann, nan=0.0, posinf=0.0, neginf=0.0)
            cov_ann = 0.5 * (cov_ann + cov_ann.T)

            n_assets = len(valid_tickers)
            dt = 1.0 / 252.0
            steps = int(252 * horizon_years)
            sqrt_dt = np.sqrt(dt)
            var_diag = np.clip(np.diag(cov_ann), 0.0, None)
            drift = (mu_ann - 0.5 * var_diag) * dt

            eye = np.eye(n_assets)
            jitter = 1e-12
            chol = None
            for _ in range(8):
                try:
                    if HAS_SCIPY and sp_cholesky is not None:
                        chol = sp_cholesky(cov_ann + jitter * eye, lower=True, check_finite=False)
                    else:
                        chol = np.linalg.cholesky(cov_ann + jitter * eye)
                    break
                except Exception:
                    jitter *= 10

            if chol is None:
                raise ValueError("Cholesky decomposition failed for covariance matrix.")

            rng = np.random.default_rng(42)
            asset_values = (initial_investment * w_mc)[:, None] * np.ones((n_assets, sim_count), dtype=float)
            port_paths = np.empty((steps + 1, sim_count), dtype=float)
            port_paths[0, :] = initial_investment

            with st.spinner("Running correlated Monte Carlo simulation..."):
                for t in range(1, steps + 1):
                    z = rng.standard_normal((n_assets, sim_count))
                    corr_shocks = chol @ z
                    log_returns = drift[:, None] + sqrt_dt * corr_shocks
                    asset_values *= np.exp(log_returns)
                    port_paths[t, :] = asset_values.sum(axis=0)

            years_axis = np.arange(steps + 1) / 252.0
            p05 = np.percentile(port_paths, 5, axis=1)
            p50 = np.percentile(port_paths, 50, axis=1)
            p95 = np.percentile(port_paths, 95, axis=1)

            draw_n = min(50, sim_count)
            path_idx = rng.choice(sim_count, size=draw_n, replace=False)

            fig_mc = go.Figure()
            for idx in path_idx:
                fig_mc.add_trace(go.Scatter(
                    x=years_axis,
                    y=port_paths[:, idx],
                    mode="lines",
                    line=dict(color="rgba(140,140,140,0.20)", width=1),
                    hoverinfo="skip",
                    showlegend=False,
                ))

            fig_mc.add_trace(go.Scatter(
                x=years_axis, y=p05, mode="lines",
                name="Bear Case (5th Percentile)",
                line=dict(color="#D62728", width=3),
            ))
            fig_mc.add_trace(go.Scatter(
                x=years_axis, y=p50, mode="lines",
                name="Base Case (Median)",
                line=dict(color="#1F77B4", width=3),
            ))
            fig_mc.add_trace(go.Scatter(
                x=years_axis, y=p95, mode="lines",
                name="Bull Case (95th Percentile)",
                line=dict(color="#2CA02C", width=3),
            ))
            fig_mc.update_layout(
                title="Correlated Monte Carlo Forecast (GBM with Cholesky)",
                xaxis_title="Years Ahead",
                yaxis_title="Portfolio Value",
                yaxis_tickprefix="$",
                yaxis_tickformat=",.0f",
                margin=dict(l=50, r=20, t=50, b=40),
                legend=dict(orientation="h", y=1.02, x=0),
                height=520,
                hovermode="x unified",
            )
            st.plotly_chart(fig_mc, width='stretch')

            end_vals = port_paths[-1, :]
            bear_end = np.percentile(end_vals, 5)
            base_end = np.percentile(end_vals, 50)
            bull_end = np.percentile(end_vals, 95)

            boardroom = pd.DataFrame({
                "Scenario": ["Bear (5th Percentile)", "Base (Median)", "Bull (95th Percentile)"],
                "Ending Wealth": [bear_end, base_end, bull_end],
            })
            boardroom["CAGR"] = (boardroom["Ending Wealth"] / initial_investment) ** (1.0 / horizon_years) - 1.0

            boardroom_fmt = boardroom.copy()
            boardroom_fmt["Ending Wealth"] = boardroom["Ending Wealth"].map("${:,.0f}".format)
            boardroom_fmt["CAGR"] = boardroom["CAGR"].map("{:.2%}".format)

            st.subheader("Boardroom Summary")
            st.dataframe(boardroom_fmt, width='stretch', hide_index=True)

            prob_below_initial = float(np.mean(end_vals < initial_investment))
            st.metric(
                "Monte Carlo VaR: P(Ending Wealth < Initial Investment)",
                f"{prob_below_initial:.2%}",
            )

        except Exception as e:
            st.warning(f"Monte Carlo simulation failed: {e}")

st.divider()
st.caption(
    "Data from Yahoo Finance via `yfinance`. Past performance is not indicative "
    "of future results. For educational/analytical purposes only â€” not financial advice."
)



