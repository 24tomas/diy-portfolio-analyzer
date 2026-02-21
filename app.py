# -*- coding: utf-8 -*-
"""
Professional Portfolio Analysis Dashboard
==========================================
Tech Stack: Streamlit · yfinance · pandas · NumPy · Plotly · QuantStats · PyPortfolioOpt
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
import plotly.io as pio
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# Keep chart rendering on Streamlit's native dark theme.
_ORIG_ST_PLOTLY_CHART = st.plotly_chart
def _plotly_chart_no_streamlit_theme(*args, **kwargs):
    kwargs.setdefault("theme", "streamlit")
    return _ORIG_ST_PLOTLY_CHART(*args, **kwargs)
st.plotly_chart = _plotly_chart_no_streamlit_theme

# ??????????????????????????????????????????????
# 1. PAGE CONFIG
# ??????????????????????????????????????????????
st.set_page_config(
    page_title="Portfolio Analyzer",
    page_icon="\U0001F4CA",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# THEME SYSTEM  (dark / light + CSS + plotly)
# ──────────────────────────────────────────────
# Dark mode is enforced (light mode removed).
st.session_state["dark_mode"] = True


def _apply_theme():
    dm = st.session_state["dark_mode"]

    # ── colour palette ──
    if dm:
        bg      = "#0C0C0C"; sb     = "#111111"; card   = "#1A1A1A"
        border  = "#2A2A2A"; text   = "#EFEFEF"; muted  = "#888888"
        accent  = "#3B82F6"; plot_bg= "#161616"; paper  = "#0C0C0C"
        grid    = "#212121"; fc     = "#D0D0D0"
        cw = ["#60A5FA","#F87171","#34D399","#A78BFA",
              "#FBBF24","#F472B6","#22D3EE","#A3E635"]
    else:
        bg      = "#F6F3ED"; sb     = "#EEE9DF"; card   = "#FCFAF5"
        border  = "#DDD6C8"; text   = "#1E2430"; muted  = "#6B7280"
        accent  = "#2F5CD9"; plot_bg= "#FCFAF5"; paper  = "#F6F3ED"
        grid    = "#E8E2D6"; fc     = "#1E2430"
        cw = ["#2563EB","#DC2626","#16A34A","#7C3AED",
              "#D97706","#DB2777","#0891B2","#65A30D"]

    # ── plotly template ──
    pio.templates["portfolio"] = go.layout.Template(
        layout=go.Layout(
            paper_bgcolor=paper,
            plot_bgcolor=plot_bg,
            font=dict(family="Inter, -apple-system, sans-serif", color=fc, size=12),
            colorway=cw,
            xaxis=dict(
                gridcolor=grid, linecolor=border, zerolinecolor=grid,
                tickfont=dict(size=11, color=fc), title_font=dict(color=fc),
            ),
            yaxis=dict(
                gridcolor=grid, linecolor=border, zerolinecolor=grid,
                tickfont=dict(size=11, color=fc), title_font=dict(color=fc),
            ),
            legend=dict(
                bgcolor="rgba(0,0,0,0)", font=dict(size=12, color=fc),
                bordercolor=border,
            ),
            hoverlabel=dict(
                bgcolor=card,
                font=dict(family="Inter, sans-serif", size=12, color=text),
                bordercolor=border,
            ),
            margin=dict(l=50, r=22, t=42, b=40),
        )
    )
    pio.templates.default = "portfolio"
    px.defaults.template = "portfolio"
    px.defaults.color_discrete_sequence = cw

    # ── CSS injection ──
    st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400;1,700&family=Inter:wght@300;400;500;600;700&display=swap');

/* ── BASE ── */
html,body{{background:{bg}!important;color:{text}!important;}}
.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"],
section.main,.block-container{{background:{bg}!important;}}
[data-testid="stHeader"]{{background:{bg}!important;border-bottom:1px solid {border}!important;}}

/* ── TYPOGRAPHY ── */
*{{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif!important;}}
/* Preserve icon ligatures used by Streamlit widgets (expander arrows, +/- icons). */
.material-icons,
.material-symbols-outlined,
.material-symbols-rounded,
[class^="material-symbols"],
[class*=" material-symbols"]{{
  font-family:'Material Symbols Rounded','Material Symbols Outlined','Material Icons'!important;
  font-weight:normal!important;
  font-style:normal!important;
  letter-spacing:normal!important;
  text-transform:none!important;
}}
h1,h2,h3,.stMarkdown h1,.stMarkdown h2,.stMarkdown h3{{
  font-family:'Playfair Display',Georgia,serif!important;
  font-weight:700!important;letter-spacing:-0.025em!important;
  color:{text}!important;line-height:1.15!important;}}
h1{{font-size:2.6rem!important;margin-bottom:0.3rem!important;}}
h2{{font-size:1.75rem!important;}}
h3{{font-size:1.2rem!important;}}
p,li,label,span,[data-testid="stMarkdown"]{{color:{text}!important;}}

/* ── SIDEBAR ── */
[data-testid="stSidebar"]{{
  background:{sb}!important;border-right:1px solid {border}!important;}}
[data-testid="stSidebar"] *{{color:{text}!important;}}
[data-testid="stSidebarNav"]{{display:none!important;}}
[data-testid="stSidebarHeader"]{{
  display:none!important;min-height:0!important;padding:0!important;margin:0!important;}}
[data-testid="stSidebarHeader"] *{{display:none!important;}}
[data-testid="stSidebarHeader"] a{{display:none!important;}}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3{{
  font-family:'Inter',sans-serif!important;font-size:0.7rem!important;
  font-weight:700!important;text-transform:uppercase!important;
  letter-spacing:0.12em!important;color:{muted}!important;}}

/* ── METRIC CARDS ── */
[data-testid="metric-container"]{{
  background:{card}!important;border:1px solid {border}!important;
  border-radius:14px!important;padding:20px 22px!important;}}
[data-testid="stMetricLabel"]>div{{
  font-size:0.68rem!important;font-weight:600!important;
  text-transform:uppercase!important;letter-spacing:0.11em!important;
  color:{muted}!important;font-family:'Inter',sans-serif!important;}}
[data-testid="stMetricValue"]>div{{
  font-family:'Playfair Display',serif!important;
  font-size:1.9rem!important;font-weight:700!important;color:{text}!important;}}
[data-testid="stMetricDelta"]>div{{
  font-size:0.78rem!important;font-weight:500!important;}}

/* ── TABS ── */
[data-testid="stTabs"] [data-baseweb="tab-list"]{{
  background:transparent!important;
  border-bottom:2px solid {border}!important;gap:0!important;padding:0!important;}}
[data-testid="stTabs"] [data-baseweb="tab"]{{
  font-size:0.78rem!important;font-weight:500!important;color:{muted}!important;
  background:transparent!important;border:none!important;
  padding:10px 14px!important;letter-spacing:0.01em!important;
  transition:color 0.15s!important;}}
[data-testid="stTabs"] [aria-selected="true"]{{
  color:{accent}!important;border-bottom:2px solid {accent}!important;
  margin-bottom:-2px!important;font-weight:600!important;}}
[data-testid="stTabs"] [data-baseweb="tab-highlight"]{{
  background-color:{accent}!important;height:2px!important;}}

/* ── BUTTONS ── */
.stButton>button{{
  font-size:0.85rem!important;font-weight:500!important;
  border-radius:8px!important;border:1.5px solid {border}!important;
  background:transparent!important;color:{text}!important;
  transition:all 0.15s!important;letter-spacing:0.01em!important;}}
.stButton>button:hover{{border-color:{accent}!important;color:{accent}!important;}}
.stButton>button[kind="primary"]{{
  background:{accent}!important;border-color:{accent}!important;
  color:#fff!important;font-weight:600!important;}}
.stButton>button[kind="primary"]:hover{{opacity:0.88!important;color:#fff!important;}}

/* ── INPUTS ── */
.stTextInput>div>div>input,.stNumberInput>div>div>input{{
  background:{card}!important;border:1.5px solid {border}!important;
  border-radius:8px!important;color:{text}!important;font-size:0.875rem!important;}}
[data-baseweb="input"] input{{
  background:{card}!important;color:{text}!important;}}
.stTextInput>div>div>input::placeholder{{color:{muted}!important;}}
[data-baseweb="select"]>div{{
  background:{card}!important;border:1.5px solid {border}!important;
  border-radius:8px!important;color:{text}!important;}}
[data-baseweb="popover"],[data-baseweb="menu"]{{
  background:{card}!important;border:1px solid {border}!important;
  border-radius:10px!important;}}

/* ── SLIDER ── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"]{{
  background:{accent}!important;}}

/* ── PLOTLY CONTAINER ── */
[data-testid="stPlotlyChart"]{{
  border-radius:12px!important;border:1px solid {border}!important;
  overflow:hidden!important;background:{card}!important;
  box-shadow:0 2px 8px rgba(0,0,0,0.04)!important;}}

/* ── DATAFRAME ── */
.stDataFrame{{
  border-radius:12px!important;border:1px solid {border}!important;
  overflow:hidden!important;}}

/* ── ALERTS ── */
[data-testid="stAlert"]{{border-radius:10px!important;}}

/* ── DIVIDER ── */
hr{{border-color:{border}!important;margin:2.5rem 0!important;opacity:1!important;}}

/* ── CAPTION ── */
.stCaption p,[data-testid="stCaptionContainer"] p{{
  color:{muted}!important;font-size:0.78rem!important;}}

/* ── CHECKBOX / TOGGLE ── */
[data-testid="stToggle"] span{{color:{text}!important;}}

/* ── SCROLLBAR ── */
::-webkit-scrollbar{{width:5px;height:5px;}}
::-webkit-scrollbar-track{{background:transparent;}}
::-webkit-scrollbar-thumb{{background:{border};border-radius:3px;}}
::-webkit-scrollbar-thumb:hover{{background:{muted};}}
</style>""", unsafe_allow_html=True)


_apply_theme()

# ──────────────────────────────────────────────
# 2. PORTFOLIO STORAGE (JSON file on disk)
# ──────────────────────────────────────────────
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
if "analysis_ready" not in st.session_state:
    st.session_state["analysis_ready"] = False
if "input_mode" not in st.session_state:
    st.session_state["input_mode"] = "Shares"


# ??????????????????????????????????????????????
# 3. SIDEBAR
# ??????????????????????????????????????????????
with st.sidebar:
    st.divider()

    # ?? 3a. Portfolio manager ?????????????????????????
    st.header("\U0001F4BE Portfolio Manager")
    saved = st.session_state["saved_portfolios"]
    portfolio_names = list(saved.keys())

    # Load a saved portfolio
    if portfolio_names:
        load_choice = st.selectbox(
            "Load saved portfolio",
            options=["\u2014 New portfolio \u2014"] + portfolio_names,
            key="load_choice",
        )
        if load_choice != "\u2014 New portfolio \u2014":
            if st.button(f"\U0001F4C2 Load '{load_choice}'", width='stretch'):
                data = saved[load_choice]
                st.session_state["holdings"] = pd.DataFrame(data["holdings"])
                st.session_state["analysis_ready"] = False
                st.rerun()

        # Delete
        del_choice = st.selectbox(
            "Delete a portfolio", options=["\u2014"] + portfolio_names,
            key="del_choice",
        )
        if del_choice != "\u2014":
            if st.button(f"\U0001F5D1\ufe0f Delete '{del_choice}'", width='stretch'):
                del st.session_state["saved_portfolios"][del_choice]
                save_portfolios_to_disk(st.session_state["saved_portfolios"])
                st.rerun()
    else:
        st.caption("No saved portfolios yet.")

    st.divider()

    # ?? 3b. Holdings editor ???????????????????????????
    st.header("\U0001F4C1 Portfolio Holdings")
    st.caption("Enter positions using shares or target portfolio weights.")

    input_mode = st.radio(
        "Input Mode",
        options=["Shares", "Target Weights (%)"],
        horizontal=True,
        key="input_mode",
    )

    if "holdings" not in st.session_state:
        st.session_state["holdings"] = pd.DataFrame(
            {"Ticker": ["AAPL", "MSFT", "GOOGL"], "Shares": [10, 15, 8]}
        )

    holdings_df = st.session_state["holdings"].copy()
    if not holdings_df.empty:
        holdings_df = holdings_df.dropna(subset=["Ticker"])
        holdings_df["Ticker"] = holdings_df["Ticker"].astype(str).str.strip().str.upper()
        holdings_df = holdings_df[holdings_df["Ticker"] != ""]
    holdings_df = holdings_df.reset_index(drop=True)

    holdings_share_map = {}
    for _, row in holdings_df.iterrows():
        ticker = str(row["Ticker"]).strip().upper()
        shares_val = pd.to_numeric(row.get("Shares", np.nan), errors="coerce")
        holdings_share_map[ticker] = float(shares_val) if pd.notna(shares_val) else 10.0

    holdings_weight_map = {}
    share_sum = float(sum(max(v, 0.0) for v in holdings_share_map.values()))
    if share_sum > 0:
        for t, v in holdings_share_map.items():
            holdings_weight_map[t] = 100.0 * max(v, 0.0) / share_sum
    elif holdings_share_map:
        eq_w = 100.0 / len(holdings_share_map)
        for t in holdings_share_map:
            holdings_weight_map[t] = eq_w

    unique_tickers = list(holdings_share_map.keys())
    if "optim_shares" not in st.session_state:
        st.session_state["optim_shares"] = holdings_share_map.copy()
    else:
        synced_shares = {}
        for ticker in unique_tickers:
            if ticker in st.session_state["optim_shares"]:
                synced_shares[ticker] = float(st.session_state["optim_shares"][ticker])
            else:
                synced_shares[ticker] = float(holdings_share_map.get(ticker, 10.0))
        st.session_state["optim_shares"] = synced_shares

    if "optim_weights" not in st.session_state:
        st.session_state["optim_weights"] = holdings_weight_map.copy()
    else:
        synced_weights = {}
        for ticker in unique_tickers:
            if ticker in st.session_state["optim_weights"]:
                synced_weights[ticker] = float(st.session_state["optim_weights"][ticker])
            else:
                synced_weights[ticker] = float(holdings_weight_map.get(ticker, 0.0))
        st.session_state["optim_weights"] = synced_weights

    new_ticker = st.text_input("Add Ticker", key="new_ticker_input").strip().upper()
    if st.button("\u2795 Add", width='stretch'):
        if not new_ticker:
            st.warning("Enter a ticker first.")
        elif new_ticker in st.session_state["optim_shares"]:
            st.info(f"{new_ticker} is already in your portfolio.")
        else:
            st.session_state["optim_shares"][new_ticker] = 10.0
            st.session_state["optim_weights"][new_ticker] = 0.0
            st.session_state["holdings"] = pd.concat(
                [
                    holdings_df[["Ticker", "Shares"]] if not holdings_df.empty else pd.DataFrame(columns=["Ticker", "Shares"]),
                    pd.DataFrame([{"Ticker": new_ticker, "Shares": 10.0}]),
                ],
                ignore_index=True,
            )
            st.session_state["analysis_ready"] = False
            st.success(f"Added {new_ticker} with 10 shares.")
            st.rerun()

    active_key = "optim_shares" if input_mode == "Shares" else "optim_weights"
    if not st.session_state[active_key]:
        st.info("No tickers yet. Add one above to start building your portfolio.")
    else:
        def _sync_from_num(ticker_name: str, mode_key: str):
            prefix = "shr" if mode_key == "optim_shares" else "wt"
            num_key = f"num_{prefix}_{ticker_name}"
            slide_key = f"slide_{prefix}_{ticker_name}"
            v = float(st.session_state.get(num_key, 0.0))
            upper = 100.0 if mode_key == "optim_weights" else v
            v = min(max(0.0, v), upper)
            st.session_state[mode_key][ticker_name] = v
            st.session_state[slide_key] = v

        def _sync_from_slide(ticker_name: str, mode_key: str):
            prefix = "shr" if mode_key == "optim_shares" else "wt"
            num_key = f"num_{prefix}_{ticker_name}"
            slide_key = f"slide_{prefix}_{ticker_name}"
            v = float(st.session_state.get(slide_key, 0.0))
            upper = 100.0 if mode_key == "optim_weights" else v
            v = min(max(0.0, v), upper)
            st.session_state[mode_key][ticker_name] = v
            st.session_state[num_key] = v

        for ticker in list(st.session_state[active_key].keys()):
            with st.container():
                prefix = "shr" if active_key == "optim_shares" else "wt"
                num_key = f"num_{prefix}_{ticker}"
                slide_key = f"slide_{prefix}_{ticker}"
                default_val = 10.0 if active_key == "optim_shares" else 0.0
                base_val = float(st.session_state[active_key].get(ticker, default_val))
                base_val = max(0.0, base_val)
                if active_key == "optim_weights":
                    base_val = min(base_val, 100.0)

                if num_key not in st.session_state:
                    st.session_state[num_key] = base_val
                if slide_key not in st.session_state:
                    st.session_state[slide_key] = base_val

                sync_anchor = max(
                    base_val,
                    float(st.session_state.get(num_key, base_val)),
                    float(st.session_state.get(slide_key, base_val)),
                )
                slider_max = 100.0 if active_key == "optim_weights" else max(1000.0, sync_anchor * 2.0)

                c1, c2, c3 = st.columns([1.5, 2, 0.5])
                c1.number_input(
                    f"{ticker}",
                    min_value=0.0,
                    step=1.0,
                    format="%.2f",
                    key=num_key,
                    on_change=_sync_from_num,
                    args=(ticker, active_key),
                )
                c2.slider(
                    "Adjust",
                    min_value=0.0,
                    max_value=float(slider_max),
                    step=1.0,
                    key=slide_key,
                    on_change=_sync_from_slide,
                    args=(ticker, active_key),
                    label_visibility="collapsed",
                )
                if c3.button("\u2715", key=f"del_{ticker}", help=f"Remove {ticker}"):
                    st.session_state["optim_shares"].pop(ticker, None)
                    st.session_state["optim_weights"].pop(ticker, None)
                    st.session_state.pop(num_key, None)
                    st.session_state.pop(slide_key, None)
                    st.session_state["analysis_ready"] = False
                    st.rerun()

    if input_mode == "Target Weights (%)" and st.session_state["optim_weights"]:
        total_weight = float(sum(st.session_state["optim_weights"].values()))
        if not np.isclose(total_weight, 100.0, atol=1e-6):
            st.warning(
                f"Target weights currently sum to {total_weight:.2f}%. "
                "They will be automatically normalized to 100% during analysis."
            )
            if st.button("Normalize to 100%", width='stretch'):
                if total_weight > 0:
                    st.session_state["optim_weights"] = {
                        t: (v / total_weight) * 100.0
                        for t, v in st.session_state["optim_weights"].items()
                    }
                st.rerun()

    edited_holdings = pd.DataFrame(
        {
            "Ticker": list(st.session_state["optim_shares"].keys()),
            "Shares": [float(v) for v in st.session_state["optim_shares"].values()],
        },
        columns=["Ticker", "Shares"],
    )
    st.session_state["holdings"] = edited_holdings.copy()

    # Save current portfolio
    save_name = st.text_input("Save current portfolio as", key="save_name_input")
    if st.button("\U0001F4BE Save Portfolio", width='stretch'):
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

    # ?? 3c. Compare with another portfolio ????????????
    st.header("\U0001F500 Compare Portfolios")
    compare_options = ["None (benchmark only)"] + portfolio_names
    compare_choice = st.selectbox("Compare with", options=compare_options, key="compare_sel")

    st.divider()

    # ?? 3d. Benchmark, rf ?????????????????

    benchmark_ticker = st.text_input("Benchmark", value="SPY",
                                     help="Used for comparison in charts and factor analysis")

    use_dynamic_rf = st.checkbox(
        "Use Dynamic Risk-Free Rate (^IRX)", value=True,
        help="Downloads 13-week T-Bill yields to calculate daily dynamic "
             "risk-free rates. Falls back to the static rate below on failure.",
    )
    risk_free_rate = st.number_input(
        "Static Risk-Free Rate Fallback (%)", min_value=0.0,
        max_value=20.0, value=4.5, step=0.1) / 100

    rebalance_freq = st.selectbox(
        "Rebalancing Frequency",
        options=["Buy & Hold", "Monthly", "Quarterly", "Annually",
                 "Daily (Constant Weights)"],
        index=0,
        help="How often the portfolio is rebalanced back to target weights. "
             "'Buy & Hold' simulates no rebalancing after initial allocation; "
             "'Daily' assumes constant weights (original behavior).",
    )
    transaction_cost_rate = st.number_input(
        "Transaction Cost / Slippage (%)",
        min_value=0.0,
        max_value=10.0,
        value=0.10,
        step=0.01,
        help="Applied on each periodic rebalance based on portfolio turnover.",
    ) / 100

    fetch_sector_data = st.checkbox(
        "Fetch sector/industry metadata",
        value=False,
        help="Adds extra Yahoo requests per ticker. Keep off to reduce rate-limit risk.",
    )

    fetch_btn = st.button("⬇️ Fetch Market Data", type="primary", width='stretch')

    if st.button("Clear cached data", width='stretch'):
        st.cache_data.clear()
        st.success("Cleared Streamlit cached data.")


# ??????????????????????????????????????????????
# 4. HELPERS
# ??????????????????????????????????????????????
def validate_holdings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().dropna(subset=["Ticker", "Shares"])
    df["Ticker"] = df["Ticker"].str.strip().str.upper()
    df = df[df["Ticker"].str.len() > 0]
    df = df[df["Shares"] > 0]
    df = df.groupby("Ticker", as_index=False)["Shares"].sum()
    if df.empty:
        st.error("\u26a0\ufe0f  Add at least one valid holding.")
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

    first_valid_dates = prices.apply(lambda col: col.first_valid_index())

    prices = prices.ffill().bfill()
    prices = prices.dropna(axis=1, how="all")
    if prices.empty:
        raise RuntimeError("Yahoo returned no usable closing price columns.")

    # Keep metadata only for columns that survived cleaning
    first_valid_dates = first_valid_dates.reindex(prices.columns)
    return prices, first_valid_dates.to_dict()


def compute_weights_and_returns(
    prices,
    holdings,
    rebalance_freq="Daily (Constant Weights)",
    shares_dict=None,
    input_mode="Shares",
    weights_dict=None,
    transaction_cost=0.0,
):
    """Compute portfolio returns via a proper backtest simulation.

    Target weights are derived from either explicit share counts or user-defined
    target weights. Historical returns are then simulated by investing
    hypothetical capital ($10,000) according to the chosen rebalancing
    strategy, eliminating look-ahead bias from constant-weight assumptions.
    """
    if shares_dict is None:
        shares_dict = dict(zip(holdings["Ticker"], holdings["Shares"]))
    if weights_dict is None:
        weights_dict = {}

    INITIAL_CAPITAL = 10_000.0

    valid = holdings[holdings["Ticker"].isin(prices.columns)].copy()
    if valid.empty:
        st.error("\u274c  None of the tickers have price data.")
        st.stop()

    tickers = valid["Ticker"].tolist()
    latest = prices.iloc[-1]
    valid["Price"] = valid["Ticker"].map(latest)

    if input_mode == "Target Weights (%)":
        raw_weights = np.array([float(weights_dict.get(t, 0.0)) for t in tickers], dtype=float)
        raw_weights = np.nan_to_num(raw_weights, nan=0.0, posinf=0.0, neginf=0.0)
        raw_weights = np.clip(raw_weights, 0.0, None)
        total_weight = float(raw_weights.sum())
        if total_weight <= 0:
            st.error("\u274c  Target weights must sum to a positive value.")
            st.stop()
        w = raw_weights / total_weight
        valid["Shares_Count"] = np.nan
        valid["Value"] = w
        valid["Weight"] = w
    else:
        # Target weights from latest prices and shares
        valid["Shares_Count"] = valid["Ticker"].map(lambda t: float(shares_dict.get(t, 0.0)))
        valid["Value"] = valid["Shares_Count"] * valid["Price"]
        total = float(valid["Value"].sum())
        if total <= 0:
            st.error("\u274c  Share counts must produce a positive portfolio value.")
            st.stop()
        valid["Weight"] = valid["Value"] / total
        w = valid["Weight"].values

    ind_ret = prices[tickers].pct_change().dropna()

    if rebalance_freq == "Daily (Constant Weights)":
        # Original logic: weights are held constant every day (daily rebalance)
        port_ret = ind_ret.dot(w)

    elif rebalance_freq == "Buy & Hold":
        asset_prices = prices[tickers]
        if input_mode == "Target Weights (%)":
            # Buy-and-hold on target weights uses one-time capital allocation.
            initial_prices = asset_prices.iloc[0].values
            bh_shares = (INITIAL_CAPITAL * w) / initial_prices
        else:
            # Buy-and-hold on shares mode tracks explicit share counts.
            bh_shares = np.array([float(shares_dict.get(t, 0.0)) for t in tickers], dtype=float)
        port_value = asset_prices.multiply(bh_shares, axis=1).sum(axis=1)
        port_ret = port_value.pct_change().dropna()

    else:
        # Periodic rebalancing: Monthly, Quarterly, Annually
        freq_map = {"Monthly": "M", "Quarterly": "Q", "Annually": "Y"}
        freq = freq_map[rebalance_freq]

        asset_prices = prices[tickers]
        dates = asset_prices.index
        price_arr = asset_prices.values          # (T, N)
        n_days = len(dates)

        # Determine segment boundaries (each rebalance period)
        period_codes = dates.to_period(freq).asi8
        changes = np.diff(period_codes) != 0
        boundaries = np.flatnonzero(changes) + 1
        seg_starts = np.concatenate([[0], boundaries]).astype(int)
        seg_ends = np.concatenate([boundaries, [n_days]]).astype(int)

        all_values = np.empty(n_days)
        capital = INITIAL_CAPITAL

        for i in range(len(seg_starts)):
            s, e = int(seg_starts[i]), int(seg_ends[i])
            seg_p = price_arr[s:e]               # (days_in_seg, N)
            p0 = seg_p[0]                        # prices at segment start
            shares = (capital * w) / p0           # fractional shares
            daily_total = (seg_p * shares).sum(axis=1)
            all_values[s:e] = daily_total
            capital = float(daily_total[-1])

            # Apply transaction costs only at actual rebalance points
            # (i.e., before starting the next segment).
            if i < (len(seg_starts) - 1) and transaction_cost > 0:
                end_values = seg_p[-1] * shares
                end_total = float(np.sum(end_values))
                if end_total > 0:
                    drifted_w = end_values / end_total
                    turnover = float(np.abs(drifted_w - w).sum())
                    cost_fraction = turnover * float(transaction_cost)
                    capital *= max(0.0, 1.0 - cost_fraction)

        port_value = pd.Series(all_values, index=dates)
        port_ret = port_value.pct_change().dropna()

    port_ret.name = "Portfolio"
    return port_ret, ind_ret, valid.set_index("Ticker")["Weight"]


@st.cache_data(show_spinner="\U0001F4E1 Downloading Fama-French factors \u2026", ttl=86400)
def download_fama_french() -> pd.DataFrame:
    """
    Download the Fama-French 5-Factor daily dataset directly from
    Kenneth French's website. Returns a DataFrame with columns
    ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF'] in decimal form
    (not percentage). Cached for 24 hours.
    """
    import urllib.request, zipfile, io

    url = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
           "ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")
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
        names=["Date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"],
    )
    ff_df["Date"] = pd.to_datetime(ff_df["Date"], format="%Y%m%d")
    ff_df = ff_df.set_index("Date")
    ff_df = ff_df.apply(pd.to_numeric, errors="coerce").dropna()
    ff_df = ff_df / 100  # percentage ? decimal

    return ff_df


@st.cache_data(show_spinner="\U0001F3E2 Looking up ticker metadata \u2026", ttl=7 * 86400, persist="disk", max_entries=256)
def get_ticker_metadata(tickers: tuple) -> dict:
    """
    For each ticker, query yf.Ticker(t).info to retrieve sector, industry,
    and dividend yield.
    Returns dict:
    {ticker: {"sector": ..., "industry": ..., "dividend_yield": ...}}.
    Cached for 7 days. Accepts a tuple (not list) so Streamlit can hash it.
    """
    default_meta = {"sector": "Unknown", "industry": "Unknown", "dividend_yield": 0.0}

    def _worker(ticker: str):
        # Small jitter to reduce synchronized bursts against Yahoo.
        time.sleep(random.uniform(0.1, 0.5))
        try:
            info = yf.Ticker(ticker).info
            div_y = info.get("trailingAnnualDividendYield", info.get("dividendYield", 0.0))
            if div_y is None:
                div_rate = info.get("trailingAnnualDividendRate", info.get("dividendRate"))
                px = info.get("currentPrice", info.get("regularMarketPrice", info.get("previousClose")))
                try:
                    if div_rate is not None and px not in (None, 0):
                        div_y = float(div_rate) / float(px)
                except Exception:
                    div_y = 0.0
            try:
                div_y = float(div_y) if div_y is not None else 0.0
            except Exception:
                div_y = 0.0
            if (not np.isfinite(div_y)) or (div_y < 0):
                div_y = 0.0
            if div_y > 1.0:
                div_y = div_y / 100.0
            return ticker, {
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "dividend_yield": div_y,
            }
        except Exception:
            return ticker, default_meta.copy()

    result = {}
    max_workers = min(8, max(1, len(tickers)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_worker, t) for t in tickers]
        for fut in as_completed(futures):
            try:
                t, meta = fut.result()
            except Exception:
                continue
            result[t] = meta

    for t in tickers:
        result.setdefault(t, default_meta.copy())
    return result


@st.cache_data(show_spinner=False, ttl=7 * 86400, persist="disk", max_entries=256)
def get_dividend_yields(tickers: tuple) -> dict:
    """Fetch dividend yields with lightweight fallbacks, even when sector metadata is off."""
    def _worker(ticker: str):
        time.sleep(random.uniform(0.1, 0.4))
        div_y = 0.0
        try:
            tk = yf.Ticker(ticker)
            fast = getattr(tk, "fast_info", {}) or {}
            div_y = fast.get("dividendYield")
            if div_y is None:
                info = tk.info
                div_y = info.get("trailingAnnualDividendYield", info.get("dividendYield"))
                if div_y is None:
                    div_rate = info.get("trailingAnnualDividendRate", info.get("dividendRate"))
                    px = info.get("currentPrice", info.get("regularMarketPrice", info.get("previousClose")))
                    if div_rate is not None and px not in (None, 0):
                        div_y = float(div_rate) / float(px)
            div_y = float(div_y) if div_y is not None else 0.0
        except Exception:
            div_y = 0.0

        if (not np.isfinite(div_y)) or (div_y < 0):
            div_y = 0.0
        if div_y > 1.0:
            div_y = div_y / 100.0
        return ticker, div_y

    out = {}
    max_workers = min(8, max(1, len(tickers)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_worker, t) for t in tickers]
        for fut in as_completed(futures):
            try:
                t, y = fut.result()
            except Exception:
                continue
            out[t] = y
    for t in tickers:
        out.setdefault(t, 0.0)
    return out


def build_black_litterman_view_matrices(view_rows, tickers):
    """
    Convert user-entered Black-Litterman views into Picking Matrix (P)
    and Views Vector (Q). Supports:
    - Absolute view: Asset i expected return
    - Relative view: Asset A outperforms Asset B by spread
    """
    idx_map = {t: i for i, t in enumerate(tickers)}
    n = len(tickers)
    P_rows, Q_vals, issues = [], [], []

    for i, row in enumerate(view_rows, start=1):
        view_type = row.get("type")
        if view_type == "Absolute":
            ticker = row.get("ticker")
            q_pct = float(row.get("q_pct", 0.0))
            if ticker not in idx_map:
                issues.append(f"View {i}: invalid ticker for absolute view.")
                continue
            p = np.zeros(n, dtype=float)
            p[idx_map[ticker]] = 1.0
            P_rows.append(p)
            Q_vals.append(q_pct / 100.0)
        elif view_type == "Relative":
            outperform = row.get("outperform")
            underperform = row.get("underperform")
            q_pct = float(row.get("q_pct", 0.0))
            if outperform not in idx_map or underperform not in idx_map:
                issues.append(f"View {i}: invalid tickers for relative view.")
                continue
            if outperform == underperform:
                issues.append(f"View {i}: outperform and underperform tickers must differ.")
                continue
            p = np.zeros(n, dtype=float)
            p[idx_map[outperform]] = 1.0
            p[idx_map[underperform]] = -1.0
            P_rows.append(p)
            Q_vals.append(q_pct / 100.0)
        else:
            issues.append(f"View {i}: unknown view type.")

    if not P_rows:
        return None, None, issues
    return np.vstack(P_rows).astype(float), np.array(Q_vals, dtype=float), issues


def build_html_report(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    rf_daily: pd.Series,
    weights: pd.Series,
    sector_map: dict,
    benchmark_name: str,
    input_mode: str,
    window_start: date,
    window_end: date,
) -> str:
    """Build a lightweight HTML tear sheet for download."""
    ann_return = ((1 + portfolio_returns).prod() ** (252 / max(len(portfolio_returns), 1))) - 1
    ann_vol = portfolio_returns.std() * np.sqrt(252)
    aligned_rf = rf_daily.reindex(portfolio_returns.index).ffill().bfill()
    excess = portfolio_returns - aligned_rf
    sharpe = 0.0 if len(excess) < 2 or np.isclose(excess.std(), 0.0) else float(excess.mean() / excess.std() * np.sqrt(252))
    comp = (1 + portfolio_returns).cumprod()
    max_dd = float(((comp - comp.cummax()) / comp.cummax()).min())
    bench_total = (1 + benchmark_returns).prod() - 1
    port_total = (1 + portfolio_returns).prod() - 1

    metrics_df = pd.DataFrame(
        [
            ("Total Return", f"{port_total:.2%}"),
            ("Total Return vs Benchmark", f"{port_total - bench_total:+.2%}"),
            ("Annualized Return", f"{ann_return:.2%}"),
            ("Annualized Volatility", f"{ann_vol:.2%}"),
            ("Sharpe Ratio (Ann.)", f"{sharpe:.2f}"),
            ("Max Drawdown", f"{max_dd:.2%}"),
            ("Benchmark", benchmark_name),
            ("Input Mode", input_mode),
            ("Analysis Window", f"{window_start.isoformat()} to {window_end.isoformat()}"),
        ],
        columns=["Metric", "Value"],
    )

    sector_df = pd.DataFrame(
        {
            "Ticker": weights.index.tolist(),
            "Weight": weights.values,
            "Sector": [sector_map.get(t, {}).get("sector", "Unknown") for t in weights.index],
        }
    )
    sector_df = sector_df.groupby("Sector", as_index=False)["Weight"].sum().sort_values("Weight", ascending=False)
    sector_df["Weight"] = sector_df["Weight"].map(lambda v: f"{v:.2%}")

    metrics_html = metrics_df.to_html(index=False, border=0, classes="table")
    sectors_html = sector_df.to_html(index=False, border=0, classes="table")
    generated_at = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Portfolio Tear Sheet</title>
  <style>
    body {{ font-family: Inter, Arial, sans-serif; margin: 28px; color: #111827; background: #ffffff; }}
    h1 {{ margin: 0 0 6px 0; font-size: 26px; }}
    .meta {{ color: #4b5563; margin-bottom: 18px; font-size: 13px; }}
    h2 {{ margin: 24px 0 10px 0; font-size: 18px; }}
    .table {{ border-collapse: collapse; width: 100%; max-width: 920px; }}
    .table th, .table td {{ border: 1px solid #d1d5db; padding: 8px 10px; text-align: left; font-size: 13px; }}
    .table th {{ background: #f3f4f6; font-weight: 600; }}
  </style>
</head>
<body>
  <h1>Portfolio Analysis Tear Sheet</h1>
  <div class="meta">Generated: {generated_at}</div>
  <h2>Key Metrics</h2>
  {metrics_html}
  <h2>Sector Exposure</h2>
  {sectors_html}
</body>
</html>
"""


# ??????????????????????????????????????????????
# 5. MAIN ? RUN ON BUTTON CLICK
# ??????????????????????????????????????????????
_title_text = st.session_state.get("dark_mode", True) and "#EFEFEF" or "#1A1A1A"
_title_muted = "#888888" if st.session_state.get("dark_mode", True) else "#6B6B6B"
st.markdown(f"""
<div style="padding:2rem 0 1.5rem 0;">
  <div style="font-family:'Inter',sans-serif;font-size:0.7rem;font-weight:700;
              letter-spacing:0.15em;text-transform:uppercase;color:{_title_muted};
              margin-bottom:0.75rem;">Investment Dashboard</div>
  <h1 style="font-family:'Playfair Display',Georgia,serif;font-size:3rem;
             font-weight:700;letter-spacing:-0.03em;line-height:1.05;
             color:{_title_text};margin:0 0 0.5rem 0;">
    Portfolio <em>Analysis</em>
  </h1>
  <p style="font-family:'Inter',sans-serif;font-size:0.85rem;color:{_title_muted};
            font-weight:400;margin:0;letter-spacing:0.02em;">
    Multi-factor backtesting · Risk decomposition · Monte Carlo simulation
  </p>
</div>
""", unsafe_allow_html=True)

if fetch_btn:
    holdings = validate_holdings(edited_holdings)
    st.session_state["holdings"] = holdings
    bench = benchmark_ticker.strip().upper()

    # Gather all tickers we need (portfolio + benchmark + comparison + ^IRX)
    all_tickers = list(dict.fromkeys(holdings["Ticker"].tolist() + [bench]))
    if use_dynamic_rf and "^IRX" not in all_tickers:
        all_tickers.append("^IRX")

    comp_holdings = None
    comp_name = None
    if compare_choice and compare_choice != "None (benchmark only)":
        comp_data = st.session_state["saved_portfolios"].get(compare_choice)
        if comp_data:
            comp_holdings = pd.DataFrame(comp_data["holdings"])
            comp_holdings = validate_holdings(comp_holdings)
            comp_name = compare_choice
            for t in comp_holdings["Ticker"]:
                if t not in all_tickers:
                    all_tickers.append(t)

    all_tickers = tuple(sorted(all_tickers))
    fetch_end = date.today()
    fetch_start = fetch_end - timedelta(days=15 * 365)

    with st.spinner("Downloading data from Yahoo Finance..."):
        try:
            all_prices, first_valid_dates = download_prices(
                all_tickers,
                fetch_start,
                fetch_end + timedelta(days=1),
            )
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

    missing_tickers = [t for t in all_tickers if t not in all_prices.columns and t != "^IRX"]
    if missing_tickers:
        st.warning(
            f"\u26a0\ufe0f No data found for: **{', '.join(missing_tickers)}**. "
            "They have been excluded from the analysis. Please check the tickers or your date range."
        )

    if all_prices.empty:
        st.error(
            "\u274c No valid price data found for any of the provided tickers. "
            "Please check your spelling or try a different date range."
        )
        st.stop()

    if bench not in all_prices.columns:
        st.error(f"Benchmark **{bench}** has no data in the fetched dataset.")
        st.stop()

    # Fama-French factors - separate source (Ken French website), no Yahoo
    try:
        ff_factors = download_fama_french()
    except Exception as e:
        st.warning(f"\u26a0\ufe0f  Could not download Fama-French factors: {e}")
        ff_factors = pd.DataFrame()

    # Ticker metadata - cached, only portfolio tickers (not benchmark)
    portfolio_tickers = tuple(sorted(holdings["Ticker"].tolist()))
    if fetch_sector_data:
        try:
            sector_map = get_ticker_metadata(portfolio_tickers)
        except Exception:
            sector_map = {
                t: {"sector": "Unknown", "industry": "Unknown", "dividend_yield": 0.0}
                for t in portfolio_tickers
            }
    else:
        try:
            div_map = get_dividend_yields(portfolio_tickers)
        except Exception:
            div_map = {}
        sector_map = {
            t: {
                "sector": "Unknown",
                "industry": "Unknown",
                "dividend_yield": float(div_map.get(t, 0.0)),
            }
            for t in portfolio_tickers
        }

    st.session_state["analysis_ready"] = True
    st.session_state["fetched_all_prices"] = all_prices
    st.session_state["fetched_first_valid_dates"] = first_valid_dates
    st.session_state["fetched_comp_holdings"] = comp_holdings
    st.session_state["fetched_comp_name"] = comp_name
    st.session_state["fetched_bench"] = bench
    st.session_state["fetched_portfolio_tickers"] = tuple(sorted(holdings["Ticker"].tolist()))
    st.session_state["ff_factors"] = ff_factors
    st.session_state["sector_map"] = sector_map
    st.session_state["fetch_window_start"] = fetch_start
    st.session_state["fetch_window_end"] = fetch_end

    available_start = all_prices.index.min().date()
    available_end = all_prices.index.max().date()
    default_start = max(available_start, available_end - timedelta(days=3 * 365))
    st.session_state["analysis_start_date"] = default_start
    st.session_state["analysis_end_date"] = available_end

if not st.session_state.get("analysis_ready", False):
    st.info("\U0001F448 Please define your portfolio and click **Fetch Market Data** to begin.")
    st.stop()

holdings = validate_holdings(edited_holdings)
current_tickers = tuple(sorted(holdings["Ticker"].tolist()))
fetched_tickers = tuple(sorted(st.session_state.get("fetched_portfolio_tickers", ())))
if fetched_tickers and current_tickers != fetched_tickers:
    st.session_state["analysis_ready"] = False
    st.info("Portfolio tickers changed. Click **Fetch Market Data** to refresh prices.")
    st.stop()
st.session_state["holdings"] = holdings

all_prices = st.session_state.get("fetched_all_prices", pd.DataFrame()).copy()
if all_prices.empty:
    st.session_state["analysis_ready"] = False
    st.info("\U0001F448 Please define your portfolio and click **Fetch Market Data** to begin.")
    st.stop()

bench = st.session_state.get("fetched_bench", benchmark_ticker.strip().upper())
comp_holdings = st.session_state.get("fetched_comp_holdings")
comp_name = st.session_state.get("fetched_comp_name")
first_valid_dates = st.session_state.get("fetched_first_valid_dates", {})

data_min = all_prices.index.min().date()
data_max = all_prices.index.max().date()
default_start = max(data_min, data_max - timedelta(days=3 * 365))
if "analysis_start_date" not in st.session_state:
    st.session_state["analysis_start_date"] = default_start
if "analysis_end_date" not in st.session_state:
    st.session_state["analysis_end_date"] = data_max
if st.session_state["analysis_start_date"] < data_min or st.session_state["analysis_start_date"] > data_max:
    st.session_state["analysis_start_date"] = default_start
if st.session_state["analysis_end_date"] < data_min or st.session_state["analysis_end_date"] > data_max:
    st.session_state["analysis_end_date"] = data_max

st.markdown("#### Analysis Date Window")
d1, d2 = st.columns(2)
analysis_start = d1.date_input(
    "Analysis Start Date",
    min_value=data_min,
    max_value=data_max,
    key="analysis_start_date",
)
analysis_end = d2.date_input(
    "Analysis End Date",
    min_value=data_min,
    max_value=data_max,
    key="analysis_end_date",
)
if analysis_end <= analysis_start:
    st.error("Analysis End Date must be after Analysis Start Date.")
    st.stop()

prices = all_prices.loc[
    (all_prices.index >= pd.Timestamp(analysis_start))
    & (all_prices.index <= pd.Timestamp(analysis_end))
].copy()
if prices.empty:
    st.error("No data available in this date range from the cached dataset. Click **Fetch Market Data**.")
    st.stop()

late_start_assets = []
late_start_threshold = pd.Timestamp(analysis_start) + pd.Timedelta(days=30)
for t in holdings["Ticker"].tolist():
    first_dt = first_valid_dates.get(t)
    if first_dt is None:
        continue
    try:
        first_ts = pd.Timestamp(first_dt)
    except Exception:
        continue
    if pd.notna(first_ts) and first_ts > late_start_threshold:
        late_start_assets.append(f"{t} ({first_ts.strftime('%Y-%m-%d')})")

if late_start_assets:
    st.warning(
        "Note: The following assets did not trade for the entire backtest period. "
        "Their earliest available price was assumed constant (0% return) prior to their inception: "
        + ", ".join(late_start_assets)
    )

# --- Dynamic risk-free rate from 13-week T-Bill (^IRX) ---
static_rf_daily = (1 + risk_free_rate) ** (1 / 252) - 1
if use_dynamic_rf and "^IRX" in all_prices.columns:
    irx_full = all_prices["^IRX"].ffill().bfill()
    irx_series = irx_full.loc[prices.index] if not prices.empty else irx_full
    dynamic_rf_daily = (1 + (irx_series / 100)) ** (1 / 252) - 1
    dynamic_rf_daily.name = "RF_daily"
    rf_float = float(irx_series.iloc[-1] / 100)
    all_prices_no_irx = all_prices.drop(columns=["^IRX"])
    prices = prices.drop(columns=["^IRX"], errors="ignore")
else:
    dynamic_rf_daily = None
    rf_float = risk_free_rate
    all_prices_no_irx = all_prices.drop(columns=["^IRX"], errors="ignore")
    prices = prices.drop(columns=["^IRX"], errors="ignore")

if bench not in prices.columns:
    st.error(f"Benchmark **{bench}** has no data.")
    st.stop()

input_mode = st.session_state.get("input_mode", "Shares")
optim_shares = st.session_state.get("optim_shares", {})
optim_weights = st.session_state.get("optim_weights", {})
port_ret, ind_ret, wt = compute_weights_and_returns(
    prices,
    holdings,
    rebalance_freq,
    shares_dict=optim_shares,
    input_mode=input_mode,
    weights_dict=optim_weights,
    transaction_cost=transaction_cost_rate,
)
bench_ret = prices[bench].pct_change().dropna()
bench_ret.name = "Benchmark"

# Comparison portfolio
comp_ret = None
comp_wt = None
if comp_holdings is not None:
    comp_ret, _, comp_wt = compute_weights_and_returns(
        prices,
        comp_holdings,
        rebalance_freq,
        shares_dict=None,
        input_mode="Shares",
        weights_dict=None,
        transaction_cost=transaction_cost_rate,
    )
    comp_ret.name = comp_name

# Align dates
common = port_ret.index.intersection(bench_ret.index)
port_ret = port_ret.loc[common]
bench_ret = bench_ret.loc[common]
ind_ret = ind_ret.loc[ind_ret.index.isin(common)]
if comp_ret is not None:
    comp_ret = comp_ret.reindex(common).dropna()

# Finalize the daily risk-free rate Series aligned to portfolio dates
if dynamic_rf_daily is not None:
    rf_series = dynamic_rf_daily.reindex(common).ffill().bfill()
else:
    rf_series = pd.Series(static_rf_daily, index=common, name="RF_daily")

# Stress test data (2007+) reuses the already downloaded full history
stress_prices = all_prices_no_irx.loc[
    (all_prices_no_irx.index >= pd.Timestamp(STRESS_START_DATE))
    & (all_prices_no_irx.index <= pd.Timestamp(analysis_end))
].copy()
if not stress_prices.empty:
    stress_port, _, _ = compute_weights_and_returns(
        stress_prices,
        holdings,
        rebalance_freq,
        shares_dict=optim_shares,
        input_mode=input_mode,
        weights_dict=optim_weights,
        transaction_cost=transaction_cost_rate,
    )
    stress_bench = (
        stress_prices[bench].pct_change().dropna()
        if bench in stress_prices.columns
        else pd.Series(dtype=float)
    )
else:
    stress_port = pd.Series(dtype=float)
    stress_bench = pd.Series(dtype=float)

ff_factors = st.session_state.get("ff_factors", pd.DataFrame())
sector_map = st.session_state.get("sector_map", {})

# Store everything
for k, v in {
    "port_ret": port_ret, "bench_ret": bench_ret, "ind_ret": ind_ret,
    "weights": wt, "bench": bench, "rf": risk_free_rate,
    "rf_series": rf_series, "rf_float": rf_float,
    "holdings": holdings, "prices": prices,
    "stress_port": stress_port, "stress_bench": stress_bench,
    "comp_ret": comp_ret, "comp_wt": comp_wt, "comp_name": comp_name,
    "ff_factors": ff_factors, "sector_map": sector_map,
    "analysis_start": analysis_start, "analysis_end": analysis_end,
}.items():
    st.session_state[k] = v

# ?? Retrieve ??
port_ret     = st.session_state["port_ret"]
bench_ret    = st.session_state["bench_ret"]
ind_ret      = st.session_state["ind_ret"]
wt           = st.session_state["weights"]
bench        = st.session_state["bench"]
rf           = st.session_state["rf"]
rf_series    = st.session_state.get("rf_series", pd.Series(dtype=float))
rf_float     = st.session_state.get("rf_float", rf)
holdings     = st.session_state["holdings"]
prices       = st.session_state["prices"]
stress_port  = st.session_state["stress_port"]
stress_bench = st.session_state["stress_bench"]
comp_ret     = st.session_state.get("comp_ret")
comp_wt      = st.session_state.get("comp_wt")
comp_name    = st.session_state.get("comp_name")
ff_factors   = st.session_state.get("ff_factors", pd.DataFrame())
sector_map   = st.session_state.get("sector_map", {})
analysis_start = st.session_state.get("analysis_start", data_min)
analysis_end = st.session_state.get("analysis_end", data_max)
input_mode = st.session_state.get("input_mode", "Shares")


# ??????????????????????????????????????????????
# 6. HOLDINGS SUMMARY & KEY METRICS
# ??????????????????????????????????????????????
def annualized_return(r): return ((1 + r).prod() ** (252 / max(len(r), 1))) - 1
def annualized_vol(r):    return r.std() * np.sqrt(252)

def dynamic_sharpe(returns, rf_daily):
    """Annualized Sharpe ratio using a daily risk-free rate Series."""
    aligned_rf = rf_daily.reindex(returns.index).ffill().bfill()
    excess = returns - aligned_rf
    if len(excess) < 2 or excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(252))

def _capture_ratio_fallback(portfolio_returns, benchmark_returns, up_market=True):
    """Compute up/down capture ratio without relying on QuantStats availability."""
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if aligned.empty:
        return np.nan
    p = aligned.iloc[:, 0].astype(float)
    b = aligned.iloc[:, 1].astype(float)
    mask = (b > 0) if up_market else (b < 0)
    if mask.sum() == 0:
        return np.nan
    p_seg = p[mask]
    b_seg = b[mask]
    p_comp = (1 + p_seg).prod() - 1
    b_comp = (1 + b_seg).prod() - 1
    if np.isclose(b_comp, 0.0):
        return np.nan
    return float(p_comp / b_comp)

def safe_up_capture(portfolio_returns, benchmark_returns):
    fn = getattr(qs.stats, "up_capture", None)
    if callable(fn):
        try:
            return float(fn(portfolio_returns, benchmark_returns))
        except Exception:
            pass
    return _capture_ratio_fallback(portfolio_returns, benchmark_returns, up_market=True)

def safe_down_capture(portfolio_returns, benchmark_returns):
    fn = getattr(qs.stats, "down_capture", None)
    if callable(fn):
        try:
            return float(fn(portfolio_returns, benchmark_returns))
        except Exception:
            pass
    return _capture_ratio_fallback(portfolio_returns, benchmark_returns, up_market=False)

def max_drawdown(r):      c = (1 + r).cumprod(); return ((c - c.cummax()) / c.cummax()).min()

st.subheader("Your Holdings")
disp = holdings.copy()
latest = prices.iloc[-1]
disp["Price"] = disp["Ticker"].map(latest)
if input_mode == "Target Weights (%)":
    wt_map = wt.to_dict()
    disp["Weight"] = disp["Ticker"].map(lambda t: float(wt_map.get(t, 0.0)))
    disp["Target Weight (%)"] = disp["Weight"] * 100.0
    # Show a notional dollar split to keep the table readable in weight mode.
    disp["Value"] = disp["Weight"] * 10_000.0
else:
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
if "Target Weight (%)" in disp_fmt.columns:
    disp_fmt["Target Weight (%)"] = disp["Target Weight (%)"].map("{:.2f}%".format)
st.dataframe(disp_fmt, width='stretch', hide_index=True)

dividend_yield_est = 0.0
for t, wgt in wt.items():
    div_y = sector_map.get(t, {}).get("dividend_yield", 0.0)
    try:
        div_y = float(div_y) if div_y is not None else 0.0
    except Exception:
        div_y = 0.0
    if np.isfinite(div_y) and div_y > 0:
        dividend_yield_est += float(wgt) * div_y

m1, m2, m3, m4, m5, m6 = st.columns(6)
ann_ret = annualized_return(port_ret)
b_ann   = annualized_return(bench_ret)
m1.metric("Ann. Return",     f"{ann_ret:.2%}")
m2.metric("Ann. Volatility", f"{annualized_vol(port_ret):.2%}")
m3.metric("Sharpe Ratio",    f"{dynamic_sharpe(port_ret, rf_series):.2f}")
m4.metric("Est. Dividend Yield", f"{dividend_yield_est:.2%}")
m5.metric("Max Drawdown",    f"{max_drawdown(port_ret):.2%}")
m6.metric("vs Benchmark",    f"{ann_ret - b_ann:+.2%}")

report_html = build_html_report(
    portfolio_returns=port_ret,
    benchmark_returns=bench_ret,
    rf_daily=rf_series,
    weights=wt,
    sector_map=sector_map,
    benchmark_name=bench,
    input_mode=input_mode,
    window_start=analysis_start,
    window_end=analysis_end,
)
with st.sidebar:
    st.divider()
    st.download_button(
        "📄 Download Report",
        data=report_html,
        file_name=f"portfolio_tearsheet_{analysis_end.isoformat()}.html",
        mime="text/html",
        use_container_width=True,
    )

st.divider()


# ??????????????????????????????????????????????
# 7. TABS
# ??????????????????????????????????????????????
asset_count_for_ui = len([t for t in wt.index if t in prices.columns])
compact_legends = asset_count_for_ui > 15
tab_perf, tab_risk, tab_factor, tab_dd, tab_stress, tab_opt, tab_mc = st.tabs(
    ["\U0001F4C8 Performance", "\u2696\ufe0f Risk & Concentration", "\U0001F52C Factor Exposure",
     "\U0001F4C9 Drawdowns", "\U0001F9EA Stress Tests", "\U0001F3AF Optimization", "\U0001F3B2 Monte Carlo"]
)


# ==================== TAB 1 ? PERFORMANCE ====================
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
                        showlegend=not compact_legends,
                        legend=dict(orientation="h", y=1.02, x=0),
                        margin=dict(l=50, r=20, t=40, b=40), height=460,
                        hovermode="x unified")
    st.plotly_chart(fig_g, width='stretch')

    # Metric cards
    st.subheader("Return Metrics")
    qs_total  = qs.stats.comp(port_ret)
    qs_cagr   = qs.stats.cagr(port_ret)
    qs_vol    = qs.stats.volatility(port_ret)
    qs_sharpe = dynamic_sharpe(port_ret, rf_series)
    qs_info   = qs.stats.information_ratio(port_ret, bench_ret)
    b_total   = qs.stats.comp(bench_ret)
    b_cagr    = qs.stats.cagr(bench_ret)
    b_vol     = qs.stats.volatility(bench_ret)

    k1, k2, k3 = st.columns(3)
    k1.metric("Total Return (Portfolio)", f"{qs_total:.2%}", delta=f"{qs_total - b_total:+.2%} vs {bench}")
    k2.metric("CAGR (Portfolio)",         f"{qs_cagr:.2%}",  delta=f"{qs_cagr - b_cagr:+.2%} vs {bench}")
    k3.metric("Volatility (Ann.)",        f"{qs_vol:.2%}",    delta=f"{qs_vol - b_vol:+.2%} vs {bench}",
              delta_color="inverse")
    up_cap = safe_up_capture(port_ret, bench_ret)
    down_cap = safe_down_capture(port_ret, bench_ret)
    k4, k5, k6 = st.columns(3)
    k4.metric("Sharpe Ratio (Ann.)",      f"{qs_sharpe:.2f}")
    k5.metric("Information Ratio (vs Bench)", f"{qs_info:.2f}",
              help="> 0.5 is good, > 1.0 is excellent.")
    k6.metric("Sortino Ratio (Ann.)",     f"{qs.stats.sortino(port_ret):.2f}",
              help="Like Sharpe but only penalises downside volatility.")
    k7, k8 = st.columns(2)
    k7.metric(
        "Up Market Capture", 
        f"{up_cap:.2f}", 
        help="> 1.0 is excellent. Shows the portfolio's participation in benchmark gains."
    )
    k8.metric(
        "Down Market Capture", 
        f"{down_cap:.2f}", 
        help="< 1.0 is excellent. Shows the portfolio's participation in benchmark losses.",
        delta_color="inverse"
    )

    # Detailed stats table
    st.subheader("Detailed Statistics")
    def _fmt(fn, *a, fmt="{:.2f}"): return fmt.format(fn(*a))
    stats_rows = [
        ("Total Return",    f"{qs.stats.comp(port_ret):.2%}",    f"{qs.stats.comp(bench_ret):.2%}"),
        ("CAGR",            f"{qs.stats.cagr(port_ret):.2%}",    f"{qs.stats.cagr(bench_ret):.2%}"),
        ("Ann. Volatility", f"{qs.stats.volatility(port_ret):.2%}", f"{qs.stats.volatility(bench_ret):.2%}"),
        ("Sharpe",          f"{dynamic_sharpe(port_ret, rf_series):.2f}", f"{dynamic_sharpe(bench_ret, rf_series):.2f}"),
        ("Sortino",         f"{qs.stats.sortino(port_ret):.2f}",  f"{qs.stats.sortino(bench_ret):.2f}"),
        ("Calmar",          f"{qs.stats.calmar(port_ret):.2f}",   f"{qs.stats.calmar(bench_ret):.2f}"),
        ("Max Drawdown",    f"{qs.stats.max_drawdown(port_ret):.2%}", f"{qs.stats.max_drawdown(bench_ret):.2%}"),
        ("Best Day",        f"{qs.stats.best(port_ret):.2%}",     f"{qs.stats.best(bench_ret):.2%}"),
        ("Worst Day",       f"{qs.stats.worst(port_ret):.2%}",    f"{qs.stats.worst(bench_ret):.2%}"),
        ("Win Rate",        f"{qs.stats.win_rate(port_ret):.1%}", f"{qs.stats.win_rate(bench_ret):.1%}"),
        ("Profit Factor",   f"{qs.stats.profit_factor(port_ret):.2f}", f"{qs.stats.profit_factor(bench_ret):.2f}"),
        ("Skew",            f"{qs.stats.skew(port_ret):.2f}",     f"{qs.stats.skew(bench_ret):.2f}"),
        ("Kurtosis",        f"{qs.stats.kurtosis(port_ret):.2f}", f"{qs.stats.kurtosis(bench_ret):.2f}"),
        ("Up Capture Ratio", f"{up_cap:.2f}", "1.00"),
        ("Down Capture Ratio", f"{down_cap:.2f}", "1.00"),
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
                "Sharpe": lambda: f"{dynamic_sharpe(comp_ret, rf_series):.2f}",
                "Sortino": lambda: f"{qs.stats.sortino(comp_ret):.2f}",
                "Calmar": lambda: f"{qs.stats.calmar(comp_ret):.2f}",
                "Max Drawdown": lambda: f"{qs.stats.max_drawdown(comp_ret):.2%}",
                "Best Day": lambda: f"{qs.stats.best(comp_ret):.2%}",
                "Worst Day": lambda: f"{qs.stats.worst(comp_ret):.2%}",
                "Win Rate": lambda: f"{qs.stats.win_rate(comp_ret):.1%}",
                "Profit Factor": lambda: f"{qs.stats.profit_factor(comp_ret):.2f}",
                "Skew": lambda: f"{qs.stats.skew(comp_ret):.2f}",
                "Kurtosis": lambda: f"{qs.stats.kurtosis(comp_ret):.2f}",
                "Up Capture Ratio": lambda: f"{safe_up_capture(comp_ret, bench_ret):.2f}",
                "Down Capture Ratio": lambda: f"{safe_down_capture(comp_ret, bench_ret):.2f}",
            }
            cols[comp_name].append(fn_map.get(metric, lambda: "?")())
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
        hovertemplate="Year %{y} ? %{x}<br>Return: %{z:.2%}<extra></extra>"))
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


# ==================== TAB 2 ? RISK & CONCENTRATION ====================
with tab_risk:

    valid_tickers = [t for t in wt.index if t in prices.columns]
    w_arr = wt[valid_tickers].values
    many_tickers = len(valid_tickers) > 15

    # ?? 1. HHI (Herfindahl-Hirschman Index) ?????????????
    st.subheader("Portfolio Concentration ? HHI")
    hhi = float(np.sum(w_arr ** 2))
    hhi_norm = (hhi - 1 / len(w_arr)) / (1 - 1 / len(w_arr)) if len(w_arr) > 1 else 1.0

    hh1, hh2 = st.columns([1, 2])
    with hh1:
        if hhi_norm < 0.15:
            label, color = "Highly Diversified", "\U0001F7E2"
        elif hhi_norm < 0.25:
            label, color = "Moderately Concentrated", "\U0001F7E1"
        elif hhi_norm < 0.50:
            label, color = "Concentrated", "\U0001F7E0"
        else:
            label, color = "Highly Concentrated", "\U0001F534"
        st.metric("HHI", f"{hhi:.4f}", help="Range: 1/N (equal weight) to 1.0 (single stock)")
        st.metric("Normalized HHI", f"{hhi_norm:.2%}")
        st.markdown(f"**{color} {label}**")
    with hh2:
        st.caption(
            "The **Herfindahl-Hirschman Index** measures portfolio concentration. "
            "It's the sum of squared weights. A portfolio equally split among *N* "
            "assets has HHI = 1/N (the minimum); a single-stock portfolio has HHI = 1.\n\n"
            "The **Normalized HHI** rescales this to 0-100%, where 0% = perfectly "
            "equal-weighted and 100% = single asset.\n\n"
            "| Norm. HHI | Interpretation |\n"
            "|-----------|----------------|\n"
            "| < 15%     | \U0001F7E2 Highly Diversified |\n"
            "| 15-25%    | \U0001F7E1 Moderately Concentrated |\n"
            "| 25-50%    | \U0001F7E0 Concentrated |\n"
            "| > 50%     | \U0001F534 Highly Concentrated |"
        )

    st.divider()

    # ?? 2. Weight allocation + Rolling vol ??????????????
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Weight Allocation")
        # Chart-only aggregation for readability on large portfolios.
        wt_full = wt[valid_tickers].astype(float).copy()
        wt_chart = wt_full.copy()
        if many_tickers:
            small = wt_chart[wt_chart < 0.015]
            wt_chart = wt_chart[wt_chart >= 0.015]
            if float(small.sum()) > 0:
                wt_chart.loc["Others"] = float(small.sum())
        wt_sorted = wt_chart.sort_values(ascending=True)
        fig_bar_w = go.Figure(go.Bar(
            x=wt_sorted.values, y=wt_sorted.index, orientation="h",
            text=[f"{v:.1%}" for v in wt_sorted.values],
            textposition="auto",
            marker_color=px.colors.qualitative.Plotly[:len(wt_sorted)],
        ))
        fig_bar_w.update_layout(
            xaxis_tickformat=".0%", xaxis_title="Weight",
            margin=dict(l=20, r=20, t=20, b=40),
            height=max(320, len(wt_sorted) * 32),
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

    # ?? 3. Sector Exposure (NEW) ????????????????????????
    st.subheader("Sector Exposure")
    if sector_map:
        # Build a DataFrame: ticker ? weight ? sector
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
                f"\u26a0\ufe0f **{top_sector['Sector']}** accounts for **{top_sector['Weight']:.1%}** "
                f"of your portfolio ? over half your exposure is in a single sector. "
                f"Consider diversifying across more sectors."
            )
        elif n_sectors <= 2:
            st.info(
                f"\U0001F4CA Your portfolio spans only **{n_sectors} sector(s)**. "
                "Broader sector exposure can help reduce concentration risk."
            )
        else:
            st.success(
                f"\u2705 Your portfolio spans **{n_sectors} sectors**, "
                f"with the largest (**{top_sector['Sector']}**) at **{top_sector['Weight']:.1%}**."
            )
    else:
        st.info("Sector data not available. Re-run analysis to fetch sector information.")

    st.divider()

    # ?? 4. Covariance matrix (PyPortfolioOpt) ???????????
    st.subheader("Annualized Covariance Matrix")

    asset_prices = prices[valid_tickers]
    if HAS_RISK_MODELS:
        st.caption("Computed with `pypfopt.risk_models.CovarianceShrinkage` (Ledoit-Wolf) ? "
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
        text=(np.round(cov_matrix.values, 4) if not many_tickers else None),
        texttemplate=("%{text}" if not many_tickers else None),
        hovertemplate="%{y} / %{x}<br>Cov: %{z:.4f}<extra></extra>",
    ))
    fig_cov.update_layout(height=max(400, len(valid_tickers) * 20),
                          margin=dict(l=40, r=20, t=20, b=40))
    st.plotly_chart(fig_cov, width='stretch')

    # Also show correlation
    st.subheader("Asset Correlation Matrix")
    corr = ind_ret[valid_tickers].corr()
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu_r", zmid=0,
        text=(np.round(corr.values, 2) if not many_tickers else None),
        texttemplate=("%{text}" if not many_tickers else None),
        zmin=-1, zmax=1))
    fig_corr.update_layout(height=max(400, len(valid_tickers) * 20),
                           margin=dict(l=40, r=20, t=20, b=40))
    st.plotly_chart(fig_corr, width='stretch')

    st.divider()

    # ?? 5. Marginal Contribution to Risk (MCR) ?????????
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

    # Chart-only aggregation for readability on large portfolios.
    risk_chart_df = risk_df.copy()
    if many_tickers:
        small_mask = (risk_chart_df["Weight"] < 0.015) & (risk_chart_df["% of Risk"] < 0.015)
        if small_mask.any():
            others_row = pd.DataFrame(
                {
                    "Ticker": ["Others"],
                    "Weight": [float(risk_chart_df.loc[small_mask, "Weight"].sum())],
                    "MCR": [np.nan],
                    "Risk Contribution": [float(risk_chart_df.loc[small_mask, "Risk Contribution"].sum())],
                    "% of Risk": [float(risk_chart_df.loc[small_mask, "% of Risk"].sum())],
                }
            )
            risk_chart_df = pd.concat([risk_chart_df.loc[~small_mask], others_row], ignore_index=True)
    risk_chart_df = risk_chart_df.sort_values("% of Risk", ascending=True)

    # Bar chart ? weight vs risk contribution side by side
    fig_mcr = go.Figure()
    fig_mcr.add_trace(go.Bar(
        y=risk_chart_df["Ticker"], x=risk_chart_df["Weight"], name="Weight",
        orientation="h", marker_color="#636EFA",
        text=[f"{v:.1%}" for v in risk_chart_df["Weight"]], textposition="auto",
    ))
    fig_mcr.add_trace(go.Bar(
        y=risk_chart_df["Ticker"], x=risk_chart_df["% of Risk"], name="% of Risk",
        orientation="h", marker_color="#EF553B",
        text=[f"{v:.1%}" for v in risk_chart_df["% of Risk"]], textposition="auto",
    ))
    fig_mcr.update_layout(
        barmode="group", xaxis_tickformat=".0%",
        xaxis_title="Proportion",
        legend=dict(orientation="h", y=1.02, x=0),
        height=max(320, len(risk_chart_df) * 42),
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
        f"\U0001F4A1 **{top_risk['Ticker']}** contributes **{top_risk['% of Risk']:.1%}** "
        f"of total portfolio risk while having a **{top_risk['Weight']:.1%}** weight. "
        + ("This is disproportionately high ? consider reducing exposure."
           if top_risk["% of Risk"] > top_risk["Weight"] * 1.5
           else "This is roughly proportional to its weight.")
    )

    st.metric("Portfolio Volatility (annualized, from cov matrix)", f"{port_vol_ann:.2%}")


# ==================== TAB 3 ? FACTOR EXPOSURE (Fama-French 5-Factor) ====================
with tab_factor:
    # ?? Single-factor (CAPM) ? always available ????????
    st.subheader("Single-Factor Model (CAPM vs Benchmark)")

    from scipy import stats as sp_stats
    # Compute excess returns by subtracting the dynamic daily risk-free rate
    rf_aligned_sf = rf_series.reindex(port_ret.index).ffill().bfill()
    excess_port = port_ret - rf_aligned_sf
    excess_bench = bench_ret - rf_aligned_sf
    aligned_sf = pd.concat([excess_port, excess_bench], axis=1).dropna()
    slope_sf, intercept_sf, r_sf, p_sf, se_sf = sp_stats.linregress(
        aligned_sf.iloc[:, 1], aligned_sf.iloc[:, 0])

    sf1, sf2, sf3, sf4 = st.columns(4)
    sf1.metric("Beta (vs benchmark)", f"{slope_sf:.3f}")
    sf2.metric("Alpha (daily)", f"{intercept_sf:.4%}")
    sf3.metric("R\u00b2", f"{r_sf**2:.2%}")
    sf4.metric("Tracking Error",
               f"{(aligned_sf.iloc[:, 0] - aligned_sf.iloc[:, 1]).std() * np.sqrt(252):.2%}")

    fig_sf = go.Figure(data=go.Scatter(
        x=aligned_sf.iloc[:, 1], y=aligned_sf.iloc[:, 0],
        mode="markers", marker=dict(size=3, opacity=0.35, color="#636EFA")))
    xr = np.linspace(aligned_sf.iloc[:, 1].min(), aligned_sf.iloc[:, 1].max(), 100)
    fig_sf.add_trace(go.Scatter(x=xr, y=intercept_sf + slope_sf * xr, mode="lines",
                                name=f"\u03b2 = {slope_sf:.3f}",
                                line=dict(color="red", width=2)))
    fig_sf.update_layout(xaxis_title=f"Benchmark ({bench}) Excess Return",
                         yaxis_title="Portfolio Excess Return",
                         xaxis_tickformat=".1%", yaxis_tickformat=".1%",
                         height=400, margin=dict(l=50, r=20, t=30, b=40),
                         showlegend=True,
                         legend=dict(orientation="h", y=1.02, x=0))
    st.plotly_chart(fig_sf, width='stretch')

    st.divider()
    roll_cov = excess_port.rolling(252).cov(excess_bench)
    roll_var = excess_bench.rolling(252).var()
    roll_beta = (roll_cov / roll_var).replace([np.inf, -np.inf], np.nan)
    roll_port_excess = excess_port.rolling(252).mean()
    roll_bench_excess = excess_bench.rolling(252).mean()
    roll_alpha_ann = (roll_port_excess - (roll_beta * roll_bench_excess)) * 252

    st.subheader("Rolling 1-Year Beta & Alpha")
    c_beta, c_alpha = st.columns(2)

    with c_beta:
        fig_beta = go.Figure()
        fig_beta.add_trace(go.Scatter(
            x=roll_beta.index,
            y=roll_beta.values,
            name="Rolling Beta",
            line=dict(color="#FFA15A", width=2),
        ))
        fig_beta.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="#888",
            annotation_text="Market Beta (1.0)",
        )
        fig_beta.add_hline(
            y=0.0,
            line_dash="dash",
            line_color="#888",
            annotation_text="Market Neutral (0.0)",
        )
        fig_beta.update_layout(
            xaxis_title="Date",
            yaxis_title="Beta",
            height=350,
            margin=dict(l=50, r=20, t=30, b=40),
        )
        st.plotly_chart(fig_beta, width='stretch')

    with c_alpha:
        fig_alpha = go.Figure()
        fig_alpha.add_trace(go.Scatter(
            x=roll_alpha_ann.index,
            y=roll_alpha_ann.values,
            name="Rolling Alpha (Ann.)",
            line=dict(color="#00CC96", width=2),
        ))
        fig_alpha.add_hline(y=0.0, line_dash="dash", line_color="#888")
        fig_alpha.update_layout(
            xaxis_title="Date",
            yaxis_title="Alpha (Annualized)",
            yaxis_tickformat="+.1%",
            height=350,
            margin=dict(l=50, r=20, t=30, b=40),
        )
        st.plotly_chart(fig_alpha, width='stretch')

    st.caption(
        "Rolling Beta tracks style drift versus the benchmark. Rolling Alpha "
        "shows time-varying excess return generation after accounting for beta."
    )

    st.divider()
    # ?? Fama-French 5-Factor Model ?????????????????????
    st.subheader("Fama-French 5-Factor Model")
    st.caption(
        "Regresses portfolio excess returns against five systematic risk factors "
        "from Kenneth French's data library: **Mkt-RF** (market risk premium), "
        "**SMB** (Small Minus Big - size factor), **HML** (High Minus Low - value factor), "
        "**RMW** (Robust Minus Weak - profitability), and "
        "**CMA** (Conservative Minus Aggressive - investment)."
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
            st.warning(f"Only {len(ff_common)} overlapping dates with FF data - "
                       "need at least 30 for a meaningful regression.")
        else:
            pr_ff = port_ret.loc[ff_common]
            factors = ff_factors.loc[ff_common]

            # Excess return = portfolio return - risk-free rate
            y = pr_ff - factors["RF"]
            X = factors[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]]
            X = sm_api.add_constant(X)

            model = sm_api.OLS(y, X).fit()
            params = model.params
            pvals = model.pvalues
            ci = model.conf_int(alpha=0.05)
            ci.columns = ["Lower 95%", "Upper 95%"]

            alpha_daily = params["const"]
            alpha_annual = (1 + alpha_daily) ** 252 - 1
            mkt_beta = params["Mkt-RF"]
            smb_beta = params["SMB"]
            hml_beta = params["HML"]
            rmw_beta = params["RMW"]
            cma_beta = params["CMA"]

            # ?? Metric cards ??
            st.markdown("##### Factor Loadings")
            ff1, ff2, ff3 = st.columns(3)
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

            ff4, ff5, ff6 = st.columns(3)
            ff4.metric(
                "Value Beta (HML)", f"{hml_beta:.3f}",
                help=f"p-value: {pvals['HML']:.4f} | "
                     f"Positive = value tilt, Negative = growth tilt",
            )
            ff5.metric(
                "Profitability Beta (RMW)", f"{rmw_beta:.3f}",
                help=f"p-value: {pvals['RMW']:.4f} | "
                     f"Positive = quality/profitability tilt",
            )
            ff6.metric(
                "Investment Beta (CMA)", f"{cma_beta:.3f}",
                help=f"p-value: {pvals['CMA']:.4f} | "
                     f"Positive = conservative investment tilt",
            )

            ff7, ff8, ff9, ff10 = st.columns(4)
            ff7.metric("R^2", f"{model.rsquared:.2%}")
            ff8.metric("Adj. R^2", f"{model.rsquared_adj:.2%}")
            ff9.metric("Observations", f"{int(model.nobs):,}")
            ff10.metric(
                "Residual Vol (ann.)",
                f"{model.resid.std() * np.sqrt(252):.2%}",
                help="Idiosyncratic risk - the volatility not explained by the five factors.",
            )

            st.divider()

            # ?? Factor loadings bar chart ??
            st.markdown("##### Factor Loadings - Visual")
            betas = pd.DataFrame({
                "Factor": [
                    "Mkt-RF\n(Market)",
                    "SMB\n(Size)",
                    "HML\n(Value)",
                    "RMW\n(Profitability)",
                    "CMA\n(Investment)",
                ],
                "Beta": [mkt_beta, smb_beta, hml_beta, rmw_beta, cma_beta],
                "Lower": [
                    ci.loc["Mkt-RF", "Lower 95%"],
                    ci.loc["SMB", "Lower 95%"],
                    ci.loc["HML", "Lower 95%"],
                    ci.loc["RMW", "Lower 95%"],
                    ci.loc["CMA", "Lower 95%"],
                ],
                "Upper": [
                    ci.loc["Mkt-RF", "Upper 95%"],
                    ci.loc["SMB", "Upper 95%"],
                    ci.loc["HML", "Upper 95%"],
                    ci.loc["RMW", "Upper 95%"],
                    ci.loc["CMA", "Upper 95%"],
                ],
                "p-value": [
                    pvals["Mkt-RF"],
                    pvals["SMB"],
                    pvals["HML"],
                    pvals["RMW"],
                    pvals["CMA"],
                ],
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
                xaxis_title="Factor Loading (beta)",
                height=320,
                margin=dict(l=120, r=60, t=20, b=40),
            )
            st.plotly_chart(fig_betas, width='stretch')

            st.caption(
                "**Colored bars** are statistically significant at the 5% level. "
                "**Gray bars** are not significant - the loading may be due to noise. "
                "Error bars show the 95% confidence interval."
            )

            st.divider()

            # ?? Interpretation helper ??
            st.markdown("##### What Do These Factors Mean?")
            interp_lines = []
            if pvals["Mkt-RF"] < 0.05:
                if mkt_beta > 1.05:
                    interp_lines.append(
                        f"**Mkt-RF = {mkt_beta:.2f}**: Your portfolio is **aggressive** - "
                        "it amplifies market moves."
                    )
                elif mkt_beta < 0.95:
                    interp_lines.append(
                        f"**Mkt-RF = {mkt_beta:.2f}**: Your portfolio is **defensive** - "
                        "it dampens market swings."
                    )
                else:
                    interp_lines.append(
                        f"**Mkt-RF = {mkt_beta:.2f}**: Your portfolio moves roughly in line "
                        "with the market."
                    )

            if pvals["SMB"] < 0.05:
                tilt = "small-cap" if smb_beta > 0 else "large-cap"
                interp_lines.append(
                    f"**SMB = {smb_beta:.2f}**: Significant **{tilt} tilt**."
                )
            else:
                interp_lines.append(
                    f"**SMB = {smb_beta:.2f}**: No significant size tilt (p = {pvals['SMB']:.2f})."
                )

            if pvals["HML"] < 0.05:
                tilt = "value" if hml_beta > 0 else "growth"
                interp_lines.append(
                    f"**HML = {hml_beta:.2f}**: Significant **{tilt} tilt**."
                )
            else:
                interp_lines.append(
                    f"**HML = {hml_beta:.2f}**: No significant value/growth tilt (p = {pvals['HML']:.2f})."
                )

            if pvals["RMW"] < 0.05:
                interp_lines.append(
                    f"**RMW = {rmw_beta:.2f}**: Significant profitability exposure."
                )
            else:
                interp_lines.append(
                    f"**RMW = {rmw_beta:.2f}**: No significant profitability tilt (p = {pvals['RMW']:.2f})."
                )

            if pvals["CMA"] < 0.05:
                interp_lines.append(
                    f"**CMA = {cma_beta:.2f}**: Significant investment-style exposure."
                )
            else:
                interp_lines.append(
                    f"**CMA = {cma_beta:.2f}**: No significant investment tilt (p = {pvals['CMA']:.2f})."
                )

            if pvals["const"] < 0.05 and alpha_annual > 0:
                interp_lines.append(
                    f"**Alpha = {alpha_annual:.2%}/yr**: Statistically significant positive alpha."
                )
            elif pvals["const"] < 0.05 and alpha_annual < 0:
                interp_lines.append(
                    f"**Alpha = {alpha_annual:.2%}/yr**: Statistically significant negative alpha."
                )

            for line in interp_lines:
                st.markdown(line)

            st.divider()

            # ?? Full regression table ??
            st.markdown("##### Full Regression Output")
            coef_order = ["const", "Mkt-RF", "SMB", "HML", "RMW", "CMA"]
            reg_table = pd.DataFrame({
                "Coefficient": params[coef_order].values,
                "Std Error": model.bse[coef_order].values,
                "t-stat": model.tvalues[coef_order].values,
                "p-value": pvals[coef_order].values,
                "95% CI Lower": ci.loc[coef_order, "Lower 95%"].values,
                "95% CI Upper": ci.loc[coef_order, "Upper 95%"].values,
            }, index=["Alpha (const)", "Mkt-RF", "SMB", "HML", "RMW", "CMA"])

            # Format
            reg_display = reg_table.copy()
            reg_display["Coefficient"] = reg_display["Coefficient"].map("{:.6f}".format)
            reg_display["Std Error"] = reg_display["Std Error"].map("{:.6f}".format)
            reg_display["t-stat"] = reg_display["t-stat"].map("{:.3f}".format)
            reg_display["p-value"] = reg_display["p-value"].map("{:.4f}".format)
            reg_display["95% CI Lower"] = reg_display["95% CI Lower"].map("{:.4f}".format)
            reg_display["95% CI Upper"] = reg_display["95% CI Upper"].map("{:.4f}".format)
            st.dataframe(reg_display, width='stretch')

            st.divider()

            # Factor Return Attribution (JP Morgan-style)
            st.markdown("##### Factor Return Attribution")
            st.caption(
                "Breaks annualized portfolio excess return into contributions from "
                "Alpha, Market (Mkt-RF), Size (SMB), Value (HML), "
                "Profitability (RMW), and Investment (CMA)."
            )

            mkt_ann = annualized_return(factors["Mkt-RF"])
            smb_ann = annualized_return(factors["SMB"])
            hml_ann = annualized_return(factors["HML"])
            rmw_ann = annualized_return(factors["RMW"])
            cma_ann = annualized_return(factors["CMA"])
            port_excess_ann = annualized_return(y)

            market_contrib = mkt_beta * mkt_ann
            size_contrib = smb_beta * smb_ann
            value_contrib = hml_beta * hml_ann
            profit_contrib = rmw_beta * rmw_ann
            invest_contrib = cma_beta * cma_ann
            alpha_contrib = alpha_annual
            total_excess_model = (
                alpha_contrib
                + market_contrib
                + size_contrib
                + value_contrib
                + profit_contrib
                + invest_contrib
            )

            wf_labels = [
                "Alpha",
                "Market Risk",
                "Size Factor",
                "Value Factor",
                "Profitability Factor",
                "Investment Factor",
                "Total Excess Return",
            ]
            wf_measures = ["relative", "relative", "relative", "relative", "relative", "relative", "total"]
            wf_values = [
                alpha_contrib,
                market_contrib,
                size_contrib,
                value_contrib,
                profit_contrib,
                invest_contrib,
                0.0,
            ]
            wf_text = [
                f"{alpha_contrib:+.2%}",
                f"{market_contrib:+.2%}",
                f"{size_contrib:+.2%}",
                f"{value_contrib:+.2%}",
                f"{profit_contrib:+.2%}",
                f"{invest_contrib:+.2%}",
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
                    "Profitability (RMW)",
                    "Investment (CMA)",
                    "Total Excess Return (Model)",
                    "Portfolio Excess Return (Actual)",
                ],
                "Beta (Sensitivity)": [
                    np.nan,
                    mkt_beta,
                    smb_beta,
                    hml_beta,
                    rmw_beta,
                    cma_beta,
                    np.nan,
                    np.nan,
                ],
                "Factor Return (Ann.)": [
                    np.nan,
                    mkt_ann,
                    smb_ann,
                    hml_ann,
                    rmw_ann,
                    cma_ann,
                    np.nan,
                    np.nan,
                ],
                "Contribution to Portfolio (Ann.)": [
                    alpha_contrib,
                    market_contrib,
                    size_contrib,
                    value_contrib,
                    profit_contrib,
                    invest_contrib,
                    total_excess_model,
                    port_excess_ann,
                ],
            })

            attr_display = attr_df.copy()
            attr_display["Beta (Sensitivity)"] = attr_display["Beta (Sensitivity)"].map(
                lambda v: "-" if pd.isna(v) else f"{v:.3f}"
            )
            attr_display["Factor Return (Ann.)"] = attr_display["Factor Return (Ann.)"].map(
                lambda v: "-" if pd.isna(v) else f"{v:+.2%}"
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


# ==================== TAB 4 ? DRAWDOWNS & TAIL RISK ====================
with tab_dd:

    # ?? 1. Key risk metrics row ?????????????????????????
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

    try:
        from scipy import stats as sp_stats
    except Exception:
        sp_stats = None

    if sp_stats is not None:
        z = sp_stats.norm.ppf(0.05)

        # Portfolio Modified VaR (Cornish-Fisher)
        S_p = skew_val
        K_p = kurt_val
        z_cf_p = (
            z
            + (1 / 6) * (z**2 - 1) * S_p
            + (1 / 24) * (z**3 - 3 * z) * K_p
            - (1 / 36) * (2 * z**3 - 5 * z) * (S_p**2)
        )
        mod_var_95_p = float(port_ret.mean() + z_cf_p * port_ret.std())

        # Benchmark Modified VaR (Cornish-Fisher)
        S_b = qs.stats.skew(bench_ret)
        K_b = qs.stats.kurtosis(bench_ret)
        z_cf_b = (
            z
            + (1 / 6) * (z**2 - 1) * S_b
            + (1 / 24) * (z**3 - 3 * z) * K_b
            - (1 / 36) * (2 * z**3 - 5 * z) * (S_b**2)
        )
        mod_var_95_b = float(bench_ret.mean() + z_cf_b * bench_ret.std())
    else:
        mod_var_95_p = np.nan
        mod_var_95_b = np.nan

    dr1, dr2, dr3, dr4, dr5 = st.columns(5)
    dr1.metric("Max Drawdown", f"{mdd_val:.2%}",
               delta=f"{mdd_val - mdd_bench:+.2%} vs {bench}",
               delta_color="inverse",
               help="Largest peak-to-trough decline in portfolio value.")
    dr2.metric("CVaR (95%)", f"{cvar_95:.2%}",
               help="Expected Shortfall ? average loss on the worst 5% of days.")
    dr3.metric("VaR (95%)", f"{var_95:.2%}",
               help="Value at Risk ? threshold below which the worst 5% of days fall.")
    dr4.metric("Calmar Ratio", f"{calmar_val:.2f}",
               help="CAGR ? Max Drawdown. Higher = better risk-adjusted returns.")
    dr5.metric(
        "Modified VaR (95%)",
        f"{mod_var_95_p:.2%}",
        help="Cornish-Fisher VaR. Adjusts standard VaR for Skewness and Kurtosis (fat tails).",
    )

    # CVaR plain-English explanation
    st.info(
        f"\U0001F4C9 **On the worst 5% of trading days, your average loss is "
        f"{abs(cvar_95):.2%}.** This means roughly once a month you can expect "
        f"a daily loss of at least {abs(var_95):.2%}, and when those bad days "
        f"happen, the average hit is {abs(cvar_95):.2%}."
    )

    st.divider()

    # ?? 2. Underwater chart ?????????????????????????????
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

    # ?? 3. Worst drawdown periods (quantstats) ?????????
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

    # ?? 4. Tail risk deep dive ??????????????????????????
    st.subheader("Tail Risk Analysis")

    tc1, tc2, tc3 = st.columns(3)

    # Skewness
    with tc1:
        st.metric("Skewness", f"{skew_val:.3f}")
        if skew_val < -0.5:
            st.caption("\u26a0\ufe0f **Negatively skewed** \u2014 heavy left tail, "
                       "meaning extreme losses are more common than extreme gains.")
        elif skew_val > 0.5:
            st.caption("\u2705 **Positively skewed** \u2014 the right tail is heavier, "
                       "meaning extreme gains are more frequent than extreme losses.")
        else:
            st.caption("\u2194\ufe0f **Roughly symmetric** \u2014 gains and losses are "
                       "similarly distributed.")

    # Kurtosis
    with tc2:
        st.metric("Excess Kurtosis", f"{kurt_val:.3f}")
        if kurt_val > 1:
            st.caption("\u26a0\ufe0f **Leptokurtic** (fat tails) \u2014 extreme moves in "
                       "either direction happen more often than a normal "
                       "distribution would predict.")
        elif kurt_val < -1:
            st.caption("\u2705 **Platykurtic** (thin tails) \u2014 extreme moves "
                       "are less common than normal.")
        else:
            st.caption("\u2194\ufe0f **Near-normal tails** \u2014 tail behavior is close "
                       "to what a Gaussian distribution would predict.")

    # Tail ratio
    with tc3:
        st.metric("Tail Ratio", f"{tail_ratio:.2f}",
                  help="Right tail (gains) ? left tail (losses) at the 95th percentile. "
                       "> 1 = fatter right tail (good).")
        if tail_ratio > 1.0:
            st.caption("\u2705 **Right tail dominant** \u2014 extreme gains tend to "
                       "outsize extreme losses.")
        else:
            st.caption("\u26a0\ufe0f **Left tail dominant** \u2014 extreme losses tend to "
                       "outsize extreme gains.")

    st.divider()

    # ?? 5. Return distribution analysis ???????????????????????
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

    # ?? 6. Extended risk statistics table ???????????????
    st.subheader("Comprehensive Risk Statistics")

    risk_stats = {
        "Metric": [
            "Max Drawdown", "Avg. Drawdown",
            "VaR (95%)", "Modified VaR (95%)", "VaR (99%)",
            "CVaR (95%)", "CVaR (99%)",
            "Skewness", "Excess Kurtosis", "Tail Ratio",
            "Calmar Ratio", "Ulcer Index", "Recovery Factor",
            "Gain/Pain Ratio",
        ],
        "Portfolio": [
            f"{mdd_val:.2%}",
            f"{dd_series[dd_series < 0].mean():.2%}" if (dd_series < 0).any() else "0.00%",
            f"{var_95:.2%}",
            f"{mod_var_95_p:.2%}",
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
            f"{mod_var_95_b:.2%}",
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


# ==================== TAB 5 ? STRESS TESTS ====================
with tab_stress:
    st.subheader("Historical Stress Test Scenarios")
    st.caption("Uses separately downloaded data back to 2007.")

    scenarios = {
        "Global Financial Crisis (Oct 2007 - Mar 2009)":        ("2007-10-09", "2009-03-09"),
        "GFC - Lehman Phase (Sep - Nov 2008)":                  ("2008-09-12", "2008-11-20"),
        "European Debt Crisis (Apr - Oct 2011)":                ("2011-04-29", "2011-10-03"),
        "US Debt Downgrade Shock (Jul - Aug 2011)":             ("2011-07-22", "2011-08-19"),
        "China Devaluation / Oil Crash (Aug 2015 - Feb 2016)":  ("2015-08-10", "2016-02-11"),
        "Vol-mageddon (Jan - Feb 2018)":                        ("2018-01-26", "2018-02-08"),
        "Q4 2018 Selloff (Sep - Dec 2018)":                     ("2018-09-20", "2018-12-24"),
        "COVID Crash (Feb - Mar 2020)":                         ("2020-02-19", "2020-03-23"),
        "COVID Reopening / Bond Rout (Nov 2020 - Mar 2021)":    ("2020-11-09", "2021-03-31"),
        "2022 Rate Shock (Jan - Oct 2022)":                     ("2022-01-03", "2022-10-12"),
        "Russia-Ukraine Invasion Shock (Feb - Mar 2022)":       ("2022-02-17", "2022-03-16"),
        "2023 Banking Crisis - SVB to FRC (Mar - May 2023)":    ("2023-03-08", "2023-05-04"),
        "Yen Carry Unwind Volatility Spike (Jul - Aug 2024)":   ("2024-07-10", "2024-08-08"),
        "Trump Tariffs Shock (Apr 2025)":                       ("2025-04-02", "2025-04-08"),
        "Tariff Escalation (Apr 2 - Apr 21, 2025)":             ("2025-04-02", "2025-04-21"),
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
                        "Benchmark": "?", "Excess": "?"})
    st.dataframe(pd.DataFrame(results), width='stretch', hide_index=True)

    st.subheader("What-If: Uniform Market Shock")
    shock_pct = st.slider("Simulated benchmark drop (%)", -50, 0, -20, step=1)
    estimated_loss = slope_sf * (shock_pct / 100)
    st.metric(f"Estimated portfolio loss (\u03b2 = {slope_sf:.2f})", f"{estimated_loss:.2%}")


# ??????????????????????????????????????????????

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

        bl_view_rows = []
        with st.expander("⚙️ Input Custom Market Views (Black-Litterman)"):
            st.caption(
                "Add absolute views (e.g., AAPL expected return 12%) and/or "
                "relative views (e.g., MSFT outperforms AAPL by 2%)."
            )
            bl_view_count = st.number_input(
                "Number of Custom Views",
                min_value=0,
                max_value=20,
                value=0,
                step=1,
                key="bl_view_count",
            )
            for i in range(int(bl_view_count)):
                st.markdown(f"**View {i + 1}**")
                t_col, a_col, b_col = st.columns([1.0, 1.3, 1.3])
                view_type = t_col.selectbox(
                    "Type",
                    options=["Absolute", "Relative"],
                    key=f"bl_view_type_{i}",
                )
                if view_type == "Absolute":
                    tkr = a_col.selectbox("Ticker", options=valid_tickers, key=f"bl_abs_ticker_{i}")
                    q_pct = b_col.number_input(
                        "Expected Return (%)",
                        value=8.0,
                        step=0.25,
                        key=f"bl_abs_q_{i}",
                    )
                    bl_view_rows.append({"type": "Absolute", "ticker": tkr, "q_pct": float(q_pct)})
                else:
                    outperform = a_col.selectbox("Outperform", options=valid_tickers, key=f"bl_rel_out_{i}")
                    underperform = b_col.selectbox("Underperform", options=valid_tickers, key=f"bl_rel_under_{i}")
                    spread_pct = st.number_input(
                        f"Outperformance Spread (%) - View {i + 1}",
                        value=2.0,
                        step=0.25,
                        key=f"bl_rel_q_{i}",
                    )
                    bl_view_rows.append(
                        {
                            "type": "Relative",
                            "outperform": outperform,
                            "underperform": underperform,
                            "q_pct": float(spread_pct),
                        }
                    )
            if int(bl_view_count) > 0:
                st.caption("Views are converted internally into the Black-Litterman P matrix and Q vector.")

        bl_P, bl_Q, bl_view_issues = build_black_litterman_view_matrices(bl_view_rows, valid_tickers)
        if bl_view_issues:
            for msg in bl_view_issues:
                st.warning(msg)

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
                ef.max_sharpe(risk_free_rate=rf_float)
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

        # 3) Black-Litterman (market priors + optional user views)
        bl_w = _zero_weights()
        if black_litterman is not None and EfficientFrontier is not None:
            try:
                bench_mu_ann = float(bench_ret.mean() * 252)
                bench_var_ann = float(bench_ret.var() * 252)
                delta = 2.5 if bench_var_ann <= 1e-12 else max((bench_mu_ann - rf_float) / bench_var_ann, 1e-6)

                eq_mcaps = pd.Series(1.0, index=valid_tickers)
                implied_rets = black_litterman.market_implied_prior_returns(
                    eq_mcaps,
                    delta,
                    cov_for_opt,
                    risk_free_rate=rf_float,
                )

                posterior_rets = implied_rets
                if bl_P is not None and bl_Q is not None:
                    bl_model = black_litterman.BlackLittermanModel(
                        cov_for_opt,
                        pi=implied_rets,
                        P=bl_P,
                        Q=bl_Q,
                        omega="default",
                        tau=0.05,
                    )
                    posterior_rets = bl_model.bl_returns()

                ef_bl = EfficientFrontier(posterior_rets, cov_for_opt)
                ef_bl.max_sharpe(risk_free_rate=rf_float)
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

        # Visualization-only noise filter: keep assets with any strategy weight > 0.5%.
        plot_opt_df = opt_df.loc[opt_df[chart_cols].max(axis=1) > 0.005].copy()
        if plot_opt_df.empty:
            plot_opt_df = opt_df.head(min(len(opt_df), 20)).copy()

        fig_opt = go.Figure()
        for col in chart_cols:
            fig_opt.add_trace(go.Bar(
                x=plot_opt_df["Ticker"],
                y=plot_opt_df[col],
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
            height=max(520, 320 + len(plot_opt_df) * 28),
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
    st.subheader("Monte Carlo Simulation")

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        sim_method = st.selectbox(
            "Simulation Method",
            options=["GBM (Parametric)", "Historical Bootstrapping (Non-Parametric)"],
            help="GBM assumes normal distribution. Bootstrapping samples actual historical returns, preserving fat tails and real correlations.",
        )
    with mc2:
        sim_count = st.slider("Simulation Count", min_value=500, max_value=5000, value=1000, step=100)
    with mc3:
        horizon_years = st.slider("Forecast Horizon (Years)", min_value=1, max_value=10, value=5, step=1)
    with mc4:
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

            hist_returns = None
            n_hist_days = 0
            chol = None
            if sim_method == "Historical Bootstrapping (Non-Parametric)":
                hist_returns = ind_ret[valid_tickers].values.astype(float)
                hist_returns = np.nan_to_num(hist_returns, nan=0.0, posinf=0.0, neginf=0.0)
                n_hist_days = len(hist_returns)
                if n_hist_days < 2:
                    raise ValueError("Not enough historical return observations for bootstrapping.")
            else:
                eye = np.eye(n_assets)
                jitter = 1e-12
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
                    if sim_method == "GBM (Parametric)":
                        z = rng.standard_normal((n_assets, sim_count))
                        corr_shocks = chol @ z
                        log_returns = drift[:, None] + sqrt_dt * corr_shocks
                        asset_values *= np.exp(log_returns)
                    else:
                        step_idx = rng.integers(0, n_hist_days, size=sim_count)
                        sampled_rets = hist_returns[step_idx, :]  # (sim_count, n_assets)
                        asset_values *= (1 + sampled_rets.T)      # (n_assets, sim_count)
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
                title=f"Monte Carlo Forecast - {sim_method}",
                xaxis_title="Years Ahead",
                yaxis_title="Portfolio Value",
                yaxis_tickprefix="$",
                yaxis_tickformat=",.0f",
                showlegend=not compact_legends,
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
    "of future results. For educational/analytical purposes only ? not financial advice. Built solely on vibes, for the vibes."
)



