# mf_explorer_webapp_streamlit_v2.py

from __future__ import annotations
import io
from dataclasses import dataclass
from functools import reduce
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import requests
import streamlit as st
from pandas.tseries.offsets import DateOffset
import plotly.graph_objects as go

# ────────────────────────────────────────────────────────────────────────────────
# Use vendored mftool from repo root (./mftool)
# ────────────────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))  # ensure local packages take priority
from mftool import Mftool  # <- vendored package in ./mftool

# ────────────────────────────────────────────────────────────────────────────────
# Configuration & App Setup
# ────────────────────────────────────────────────────────────────────────────────
APP_NAME = "Mutual Fund Explorer"
st.set_page_config(page_title=APP_NAME, page_icon="📈", layout="wide")

st.title(APP_NAME)
st.markdown("Compare and analyze the performance of Indian mutual funds. Start by selecting funds from the sidebar.")

# ────────────────────────────────────────────────────────────────────────────────
# Data classes & Session State
# ────────────────────────────────────────────────────────────────────────────────
@dataclass
class SchemeData:
    code: str
    name: str
    daily_df: pd.DataFrame
    month_end_df: pd.DataFrame

if "watchlist" not in st.session_state:
    st.session_state.watchlist: dict[str, SchemeData] = {}

# ────────────────────────────────────────────────────────────────────────────────
# Data Fetching & Processing (Cached)
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Fetching latest fund list...", ttl=6 * 3600)
def fetch_navall_text() -> str:
    """Fetches the complete list of schemes from AMFI."""
    url = "https://www.amfiindia.com/spages/NAVAll.txt"
    resp = requests.get(url, timeout=40)
    resp.raise_for_status()
    return resp.text

@st.cache_data(show_spinner=False)
def parse_schemes_from_navall(navall_text: str) -> dict[str, list[tuple[str, str]]]:
    """Parses the NAVAll text into a dictionary of {amc: [(code, name), ...]}."""
    schemes_by_amc = {}
    current_amc = None
    lines = navall_text.strip().splitlines()
    for line in lines:
        if "mutual fund" in line.lower() and ";" not in line:
            current_amc = line.strip()
            if current_amc not in schemes_by_amc:
                schemes_by_amc[current_amc] = []
        elif current_amc and ";" in line:
            parts = line.strip().split(";")
            if len(parts) > 5 and parts[0].isdigit():
                schemes_by_amc[current_amc].append((parts[0], parts[3]))
    return schemes_by_amc

@st.cache_data(show_spinner="Fetching NAV data for '{scheme_code}'...", ttl=6 * 3600)
def fetch_scheme_nav_dataframe(scheme_code: str) -> pd.DataFrame:
    """Fetches and processes historical NAV for a single scheme."""
    mf = Mftool()
    data = mf.get_scheme_historical_nav(scheme_code, as_Dataframe=True)
    if data is None or data.empty:
        return pd.DataFrame(columns=["date", "nav"])
    
    df = data.copy().rename_axis("date").reset_index()
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna().sort_values("date", ascending=True).reset_index(drop=True)
    return df[["date", "nav"]]

def to_month_end(df: pd.DataFrame) -> pd.DataFrame:
    """Converts a daily NAV dataframe to a month-end dataframe."""
    if df.empty:
        return pd.DataFrame(columns=["date", "nav"])
    s = df.copy()
    s['date'] = pd.to_datetime(s['date'])
    month_end_df = s.groupby(s['date'].dt.to_period('M')).last().reset_index(drop=True)
    if not month_end_df.empty:
        month_end_df['date'] = month_end_df['date'] + pd.offsets.MonthEnd(0)
    return month_end_df.sort_values('date').reset_index(drop=True)

def load_scheme_data(code: str, name: str) -> SchemeData | None:
    """Loads daily and month-end data for a scheme."""
    daily = fetch_scheme_nav_dataframe(code)
    if daily.empty:
        st.sidebar.error(f"No NAV data found for {name} ({code}).")
        return None
    month_end = to_month_end(daily)
    return SchemeData(code=code, name=name, daily_df=daily, month_end_df=month_end)

# ────────────────────────────────────────────────────────────────────────────────
# UI: Sidebar for Fund Selection
# ────────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 Fund Selection")
    try:
        navall_text = fetch_navall_text()
        schemes_by_amc = parse_schemes_from_navall(navall_text)
        amcs = sorted(schemes_by_amc.keys())
    except Exception as e:
        st.error(f"Failed to load fund list: {e}")
        schemes_by_amc = {}
        amcs = []

    if amcs:
        selected_amc = st.selectbox("1. Select a Fund House (AMC)", amcs)
        
        if selected_amc:
            amc_schemes = schemes_by_amc.get(selected_amc, [])
            scheme_options = {f"{name} ({code})": code for code, name in amc_schemes}
            
            selected_schemes = st.multiselect(
                "2. Find and add schemes to your watchlist",
                options=scheme_options.keys(),
                help="Type to search for funds. You can add multiple funds."
            )

            # Add selected schemes to watchlist
            for selection in selected_schemes:
                code = scheme_options[selection]
                if code not in st.session_state.watchlist:
                    name = selection.rsplit(' (', 1)[0]
                    data = load_scheme_data(code, name)
                    if data:
                        st.session_state.watchlist[code] = data
                        st.rerun()

    st.header("📋 Your Watchlist")
    if not st.session_state.watchlist:
        st.info("Your watchlist is empty. Add funds above to begin analysis.")
    else:
        # Create a more intuitive removal list
        for code, data in list(st.session_state.watchlist.items()):
            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                st.caption(data.name)
            with col2:
                if st.button("🗑️", key=f"del_{code}", help="Remove this fund"):
                    del st.session_state.watchlist[code]
                    st.rerun()
        
        if st.button("Clear All", use_container_width=True):
            st.session_state.watchlist.clear()
            st.rerun()

# ────────────────────────────────────────────────────────────────────────────────
# Main Content Area
# ────────────────────────────────────────────────────────────────────────────────
if not st.session_state.watchlist:
    st.info("⬅️ **Welcome!** Select one or more mutual funds from the sidebar to see charts and performance data.")
    st.stop()

tab_chart, tab_data, tab_performance = st.tabs(
    ["📊 NAV Chart", "📅 Data Table", "📈 Performance Analysis"]
)

# ─── Tab 1: Interactive NAV Chart ───────────────────────────────────────────────
with tab_chart:
    st.subheader("NAV History Comparison")
    
    col1, col2 = st.columns(2)
    with col1:
        show_month_end = st.toggle("Show Month-End NAV Only", value=False)
    with col2:
        rebase_to_100 = st.toggle("Rebase NAV to 100", value=True, disabled=(len(st.session_state.watchlist) < 2))

    # Determine the common date range
    inception_dates = [s.daily_df['date'].min() for s in st.session_state.watchlist.values() if not s.daily_df.empty]
    if inception_dates:
        start_date = max(inception_dates)
        end_date = datetime.now()
        
        # Date range slider for focused analysis
        chart_start, chart_end = st.slider(
            "Select Date Range",
            min_value=start_date.date(),
            max_value=end_date.date(),
            value=(start_date.date(), end_date.date()),
            format="DD MMM YYYY"
        )
        chart_start, chart_end = pd.to_datetime(chart_start), pd.to_datetime(chart_end)
    else:
        st.warning("No data available to plot.")
        st.stop()

    # Create Plotly figure
    fig = go.Figure()
    for s in st.session_state.watchlist.values():
        df = s.month_end_df if show_month_end else s.daily_df
        df_filtered = df[(df['date'] >= chart_start) & (df['date'] <= chart_end)].copy()

        if not df_filtered.empty:
            y_values = df_filtered['nav']
            if rebase_to_100:
                base_nav = y_values.iloc[0]
                if pd.notna(base_nav) and base_nav != 0:
                    y_values = (y_values / base_nav) * 100

            fig.add_trace(go.Scatter(
                x=df_filtered['date'],
                y=y_values,
                mode='lines',
                name=s.name,
                hovertemplate='%{y:.2f}'
            ))

    # Customize layout
    yaxis_title = "Rebased to 100" if rebase_to_100 else "NAV (₹)"
    fig.update_layout(
        title=f"NAV Performance since {chart_start.strftime('%d-%b-%Y')}",
        yaxis_title=yaxis_title,
        legend_title="Funds",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

# ─── Tab 2: Merged Data Table ───────────────────────────────────────────────────
with tab_data:
    st.subheader("Merged NAV Data")

    export_freq = st.radio("Select Data Frequency", ["Daily", "Month-End"], horizontal=True, index=1)
    
    frames = []
    for s in st.session_state.watchlist.values():
        df_to_use = s.month_end_df if export_freq == "Month-End" else s.daily_df
        if not df_to_use.empty:
            frames.append(df_to_use.rename(columns={"nav": s.name}))
            
    if frames:
        merged_df = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer'), frames)
        merged_df = merged_df.sort_values('date', ascending=False).reset_index(drop=True)
        
        # Display a preview
        st.dataframe(merged_df.head(100), use_container_width=True)

        # Excel Export
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            merged_df.to_excel(writer, sheet_name="NAVs", index=False)
        
        st.download_button(
            label="💾 Download as Excel",
            data=out.getvalue(),
            file_name=f"nav_comparison_{export_freq.lower()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.info("No data available for the selected funds.")

# ─── Tab 3: Performance Analysis ──────────────────────────────────────────────
# (Performance helper functions remain the same as in the original code)
def _pick_start_date_for_period(
    df: pd.DataFrame, end_date: pd.Timestamp, months: int, tolerance_days: int = 15
) -> pd.Timestamp | None:
    target = end_date - DateOffset(months=months)
    mask = df['date'] >= target
    if not mask.any():
        return None
    candidate = df.loc[mask, 'date'].iloc[0]
    if (candidate - target) > pd.Timedelta(days=tolerance_days):
        return None
    return candidate

def calc_point_to_point_returns(df: pd.DataFrame) -> list[tuple]:
    if df.empty or len(df) < 2: return []
    end_date, end_nav = df['date'].iloc[-1], float(df['nav'].iloc[-1])
    inception_date, inception_nav = df['date'].iloc[0], float(df['nav'].iloc[0])
    
    periods = {"1M": 1, "3M": 3, "6M": 6, "1Y": 12, "3Y": 36, "5Y": 60, "10Y": 120}
    out = []
    for label, months in periods.items():
        start_date = _pick_start_date_for_period(df, end_date, months)
        if start_date is None:
            out.append((label, "N/A", "N/A"))
            continue
        start_nav = float(df.loc[df['date'] == start_date, 'nav'].iloc[0])
        years = (end_date - start_date).days / 365.25
        if years >= 1.0:
            cagr = ((end_nav / start_nav) ** (1/years) - 1) * 100
            out.append((label, "N/A", f"{cagr:.2f}"))
        else:
            ret = ((end_nav / start_nav) - 1) * 100
            out.append((label, f"{ret:.2f}", "N/A"))
    
    si_years = (end_date - inception_date).days / 365.25
    if si_years >= 1.0:
        si_cagr = ((end_nav / inception_nav) ** (1/si_years) - 1) * 100
        out.append(("Since Inception", "N/A", f"{si_cagr:.2f}"))
    else:
        si_ret = ((end_nav / inception_nav) - 1) * 100
        out.append(("Since Inception", f"{si_ret:.2f}", "N/A"))
    return out

def calc_calendar_year_returns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame(columns=["Year", "Return (%)"])
    tmp = df.copy()
    tmp['year'] = tmp['date'].dt.year
    rows = []
    for yr, grp in tmp.groupby('year'):
        if len(grp) < 2: continue
        ret = ((grp['nav'].iloc[-1] / grp['nav'].iloc[0]) - 1) * 100
        rows.append((str(yr), f"{ret:.2f}"))
    return pd.DataFrame(sorted(rows, key=lambda x: x[0], reverse=True), columns=["Year", "Return (%)"])

def calc_fin_year_returns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame(columns=["Financial Year", "Return (%)"])
    tmp = df.copy()
    tmp['fy'] = tmp['date'].dt.to_period('Q-MAR').apply(lambda p: p.qyear)
    rows = []
    for fy, grp in tmp.groupby('fy'):
        if len(grp) < 2: continue
        ret = ((grp['nav'].iloc[-1] / grp['nav'].iloc[0]) - 1) * 100
        label = f"FY {fy-1}-{str(fy)[-2:]}"
        rows.append((label, f"{ret:.2f}"))
    return pd.DataFrame(sorted(rows, key=lambda x: x[0], reverse=True), columns=["Financial Year", "Return (%)"])


with tab_performance:
    st.subheader("Performance Analysis")

    # Use selectbox for single fund analysis
    fund_names = {s.name: s.code for s in st.session_state.watchlist.values()}
    selected_name = st.selectbox("Select a fund to analyze", options=fund_names.keys())

    if selected_name:
        code = fund_names[selected_name]
        data = st.session_state.watchlist[code]
        df = data.month_end_df.copy() # Always use month-end for consistency

        if df.empty:
            st.error("No month-end NAV data available for analysis.")
        else:
            # Point-to-Point Returns
            st.markdown("##### Period Returns")
            pr_data = calc_point_to_point_returns(df)
            pr_df = pd.DataFrame(pr_data, columns=["Period", "Absolute Return (%)", "CAGR (%)"])
            st.dataframe(pr_df, use_container_width=True, hide_index=True)

            # Calendar and Financial Year Returns in columns
            col_cy, col_fy = st.columns(2)
            with col_cy:
                st.markdown("##### Calendar Year Returns")
                cy_df = calc_calendar_year_returns(df)
                st.dataframe(cy_df, use_container_width=True, hide_index=True)
            with col_fy:
                st.markdown("##### Financial Year Returns")
                fy_df = calc_fin_year_returns(df)
                st.dataframe(fy_df, use_container_width=True, hide_index=True)

            # Combined Excel Export
            out_perf = io.BytesIO()
            with pd.ExcelWriter(out_perf, engine="openpyxl") as writer:
                pr_df.to_excel(writer, sheet_name="Period_Returns", index=False)
                cy_df.to_excel(writer, sheet_name="Calendar_Year", index=False)
                fy_df.to_excel(writer, sheet_name="Financial_Year", index=False)
            
            st.download_button(
                label="💾 Download Performance Data",
                data=out_perf.getvalue(),
                file_name=f"{selected_name.replace(' ', '_')}_performance.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
