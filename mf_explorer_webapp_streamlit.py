"""
Mutual Fund Explorer – Web App (Streamlit)
-----------------------------------------
FINAL VERSION – AMFI via mftool ONLY (no MFAPI)
• Uses a vendored/bootstrap mftool so it works on Streamlit Cloud even if pip install fails.
• Features:
  - AMC & Scheme discovery via AMFI NAVAll.txt
  - Watchlist & comparison
  - NAV chart (daily / month-end), optional rebasing to 100, custom start date when rebasing
  - Month-end merged table
  - Performance Analysis: Period (1M–10Y & SI), Calendar Year, Financial Year
  - Excel exports

Run locally:
  pip install -r requirements.txt
  streamlit run mf_explorer_webapp_streamlit.py

Deploy (Streamlit Community Cloud):
  Push this file + requirements.txt + .streamlit/config.toml to GitHub, then New app → main file path = mf_explorer_webapp_streamlit.py
"""

from __future__ import annotations
import io
import sys
from dataclasses import dataclass
from functools import reduce
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from pandas.tseries.offsets import DateOffset
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ────────────────────────────────────────────────────────────────────────────────
# Ensure local vendored mftool is available (AMFI-only, no MFAPI)
# ────────────────────────────────────────────────────────────────────────────────

def _ensure_local_mftool():
    """Try to import mftool; if it fails, download repo files to a writable folder
    (./mftool or /tmp/mftool), add that folder to sys.path, then import again.
    This keeps AMFI (via mftool) as the sole data source.
    """
    try:
        from mftool import Mftool  # noqa: F401
        return
    except Exception:
        pass

    import zipfile

    # Choose a writable base folder
    candidates = [Path.cwd(), Path("/tmp")]
    REF = "master"  # you may pin a tag/commit here, e.g., "v2.5" or a SHA
    ZIP_URL = f"https://codeload.github.com/NayakwadiS/mftool/zip/refs/heads/{REF}"

    for base in candidates:
        try:
            pkg_dir = base / "mftool"
            # Clean if exists
            if pkg_dir.exists():
                for p in sorted(pkg_dir.rglob("*"), reverse=True):
                    if p.is_file():
                        p.unlink()
                for p in sorted(pkg_dir.rglob("*"), reverse=True):
                    if p.is_dir():
                        p.rmdir()
            pkg_dir.mkdir(parents=True, exist_ok=True)

            # Download zip
            r = requests.get(ZIP_URL, timeout=60)
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                # Top-level dir name in archive (e.g., "mftool-master/")
                tops = sorted({n.split("/")[0] for n in z.namelist()})
                top = [t for t in tops if t.startswith("mftool-")][0]
                needed = ["__init__.py", "mftool.py", "utils.py", "const.json", "scheme_codes.json"]
                for fn in needed:
                    with z.open(f"{top}/{fn}") as src, open(pkg_dir / fn, "wb") as dst:
                        dst.write(src.read())

            # Ensure path and import
            if str(base) not in sys.path:
                sys.path.insert(0, str(base))
            from mftool import Mftool  # noqa: F401
            return
        except Exception:
            continue

    raise RuntimeError("Unable to prepare local mftool package for AMFI access.")


_ensure_local_mftool()
from mftool import Mftool  # local vendored copy

# ────────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────────
APP_NAME = "Mutual Fund Explorer (Performance Analysis V7.2) – Web"
CHART_COLORS = ['#0078D7', '#E81123', '#00B294', '#F7630C', '#5C2D91', '#00CC6A', '#DA3B01']
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

st.set_page_config(page_title=APP_NAME, page_icon="📈", layout="wide")
st.title(APP_NAME)

# ────────────────────────────────────────────────────────────────────────────────
# Data classes
# ────────────────────────────────────────────────────────────────────────────────
@dataclass
class SchemeData:
    code: str
    name: str
    daily_df: pd.DataFrame  # columns: [date, nav]
    month_end_df: pd.DataFrame  # columns: [date, nav]

# ────────────────────────────────────────────────────────────────────────────────
# Utility & Data Layer
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_navall_text() -> str:
    url = "https://www.amfiindia.com/spages/NAVAll.txt"
    resp = requests.get(url, timeout=40)
    resp.raise_for_status()
    return resp.text


def parse_amc_names(navall_text: str) -> list[str]:
    lines = [ln.strip() for ln in navall_text.splitlines()]
    amcs = sorted({ln for ln in lines if ("mutual fund" in ln.lower() and ";" not in ln)}, key=str.casefold)
    return amcs


def parse_schemes_for_amc(navall_text: str, amc_name: str) -> list[tuple[str, str]]:
    """Return list of (scheme_code, scheme_name) for the given AMC."""
    schemes: list[tuple[str, str]] = []
    is_target = False
    for raw in navall_text.splitlines():
        line = raw.strip()
        if "mutual fund" in line.lower() and ";" not in line:
            is_target = (line == amc_name)
        elif is_target and ";" in line:
            parts = line.split(";")
            # Format varies a bit, but parts[0] is code, parts[3] is scheme name in current layout
            if len(parts) > 5 and parts[0].isdigit():
                schemes.append((parts[0], parts[3]))
    return schemes


@st.cache_data(show_spinner=True, ttl=60 * 60)
def fetch_scheme_nav_dataframe(scheme_code: str) -> pd.DataFrame:
    """Fetch historical NAV via AMFI using vendored mftool only.
    Returns DataFrame with columns ['date','nav'] sorted ascending.
    """
    mf = Mftool()
    data = mf.get_scheme_historical_nav(scheme_code, as_Dataframe=True)
    if data is None or data.empty:
        return pd.DataFrame(columns=["date", "nav"])  # graceful N/A; UI will handle

    df = data.copy().rename_axis("date").reset_index()
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    df["nav"]  = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna().sort_values("date", ascending=True).reset_index(drop=True)
    return df[["date", "nav"]]


def to_month_end(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "nav"])    
    s = df.copy()
    s['date'] = pd.to_datetime(s['date'])
    s = s.sort_values('date')
    # choose last available row per month, then set date to month-end
    month_end_df = s.groupby(s['date'].dt.to_period('M')).last().reset_index(drop=True)
    if not month_end_df.empty:
        month_end_df['date'] = month_end_df['date'] + pd.offsets.MonthEnd(0)
    return month_end_df.sort_values('date').reset_index(drop=True)


def load_scheme_data(code: str, name: str) -> SchemeData:
    daily = fetch_scheme_nav_dataframe(code)
    month_end = to_month_end(daily) if not daily.empty else pd.DataFrame(columns=["date", "nav"])
    return SchemeData(code=code, name=name, daily_df=daily, month_end_df=month_end)


def merge_frames_on_date(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    merged = reduce(lambda L, R: pd.merge(L, R, on='date', how='outer'), frames)
    return merged.sort_values('date', ascending=False).reset_index(drop=True)

# ────────────────────────────────────────────────────────────────────────────────
# Session State
# ────────────────────────────────────────────────────────────────────────────────
if "watchlist" not in st.session_state:
    st.session_state.watchlist: dict[str, SchemeData] = {}

# ────────────────────────────────────────────────────────────────────────────────
# Sidebar – AMC → Scheme → Watchlist
# ────────────────────────────────────────────────────────────────────────────────
st.sidebar.header("1) Select Fund House (AMC)")
try:
    navall_text = fetch_navall_text()
    amcs = parse_amc_names(navall_text)
except Exception as e:
    st.sidebar.error(f"Failed to load AMCs: {e}")
    amcs = []
    navall_text = ""

selected_amc = st.sidebar.selectbox("AMC", amcs, index=0 if amcs else None)

st.sidebar.header("2) Find Scheme")
if selected_amc:
    schemes = parse_schemes_for_amc(navall_text, selected_amc)
    q = st.sidebar.text_input("Search scheme")
    filtered = [s for s in schemes if q.lower() in s[1].lower()] if q else schemes
    show_df = pd.DataFrame(filtered, columns=["Code", "Scheme Name"]) if filtered else pd.DataFrame(columns=["Code", "Scheme Name"]) 
    st.sidebar.dataframe(show_df, height=250, use_container_width=True)

    add_choice = st.sidebar.selectbox(
        "Add to comparison", [f"{c} – {n}" for c, n in filtered] if filtered else ["No schemes"], index=0 if filtered else None
    )
    if filtered and st.sidebar.button("➕ Add"):
        code = add_choice.split(" – ", 1)[0]
        name_map = dict(filtered)
        name = name_map.get(code)
        if code in st.session_state.watchlist:
            st.sidebar.info("Already in watchlist.")
        else:
            with st.spinner(f"Fetching NAV for {name} ({code})…"):
                data = load_scheme_data(code, name)
            if data.daily_df.empty:
                st.sidebar.error("No NAV data found for the selected scheme.")
            else:
                st.session_state.watchlist[code] = data
                st.sidebar.success(f"Added: {name}")

st.sidebar.header("4) Watchlist")
if st.session_state.watchlist:
    wl_df = pd.DataFrame([(s.code, s.name) for s in st.session_state.watchlist.values()], columns=["Code", "Name"]).sort_values("Name")
    st.sidebar.dataframe(wl_df, use_container_width=True, height=220)
    remove_code = st.sidebar.selectbox("Remove scheme", [s.code for s in st.session_state.watchlist.values()])
    if st.sidebar.button("➖ Remove"):
        st.session_state.watchlist.pop(remove_code, None)
else:
    st.sidebar.info("Add schemes to see charts & tables.")

# ────────────────────────────────────────────────────────────────────────────────
# Main Tabs
# ────────────────────────────────────────────────────────────────────────────────
tab_overview, tab_month_end, tab_performance = st.tabs(["📊 Chart Overview", "📅 Month-End Data", "📈 Performance Analysis"])

# ────────────────────────────────────────────────────────────────────────────────
# Tab 1 – Chart Overview
# ────────────────────────────────────────────────────────────────────────────────
with tab_overview:
    st.subheader("NAV History Comparison")
    if not st.session_state.watchlist:
        st.info("Add funds to the watchlist from the sidebar.")
    else:
        col1, col2, col3 = st.columns([1,1,2])
        with col1:
            show_month_end = st.checkbox("Show Month-End Only", value=False)
        with col2:
            scale_to_100 = st.checkbox("Scale NAV to 100 (requires ≥2 funds)", value=False, disabled=(len(st.session_state.watchlist) < 2))
        with col3:
            st.caption("When not rebasing, time-range defaults to 'Since inception of latest fund'.")

        # Time range controls
        if scale_to_100:
            mode = st.radio(
                "Chart Time Range",
                ["Since Inception of Latest Fund", "Since Inception of Oldest Fund", "From Custom Date"],
                horizontal=True,
            )
            custom_month = st.selectbox("Start Month", MONTHS, index=datetime.now().month - 1, key="custom_month")
            custom_year = st.selectbox(
                "Start Year",
                options=list(range(datetime.now().year, 1990, -1)),
                index=0,
                key="custom_year",
            )
        else:
            mode = "Since Inception of Latest Fund"
            custom_month = MONTHS[datetime.now().month - 1]
            custom_year = datetime.now().year

        # Build chart data
        series = []
        inception_dates = []
        for s in st.session_state.watchlist.values():
            df = s.month_end_df if show_month_end else s.daily_df
            if not df.empty:
                series.append((s.name, df))
                inception_dates.append(df['date'].min())

        if not series:
            st.warning("No data to plot.")
        else:
            if mode == "Since Inception of Latest Fund":
                chart_start = max(inception_dates)
            elif mode == "Since Inception of Oldest Fund":
                chart_start = min(inception_dates)
            else:
                month_num = MONTHS.index(custom_month) + 1
                chart_start = pd.to_datetime(f"{custom_year}-{month_num:02d}-01")

            fig, ax = plt.subplots(figsize=(10, 5))
            for i, (name, df) in enumerate(series):
                dfp = df[df['date'] >= chart_start].copy()
                if dfp.empty:
                    continue
                ycol = 'nav'
                if scale_to_100:
                    base = dfp['nav'].iloc[0]
                    if pd.isna(base) or base == 0:
                        continue
                    dfp['scaled'] = (dfp['nav'] / base) * 100.0
                    ycol = 'scaled'
                ax.plot(dfp['date'], dfp[ycol], label=name, linewidth=1.6, color=CHART_COLORS[i % len(CHART_COLORS)])

            title_date = chart_start.strftime('%d-%b-%Y')
            if scale_to_100:
                ax.set_ylabel("Rebased Performance (Starts at 100)")
                ax.set_title(f"Rebased NAV Since {title_date}")
            else:
                ax.set_ylabel("NAV (₹)")
                ax.set_title(f"NAV History Since {title_date}")

            ax.legend(fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            fig.autofmt_xdate()
            st.pyplot(fig, use_container_width=True)

        # Export merged NAVs
        st.markdown("---")
        st.subheader("Export Comparison to Excel")
        if st.session_state.watchlist:
            use_month = st.radio("Export Frequency", ["Daily", "Month-End"], horizontal=True)
            frames = []
            for s in st.session_state.watchlist.values():
                base = s.month_end_df if use_month == "Month-End" else s.daily_df
                if not base.empty:
                    frames.append(base[["date", "nav"]].rename(columns={"nav": s.name}))
            merged = merge_frames_on_date(frames)
            if not merged.empty:
                out = io.BytesIO()
                with pd.ExcelWriter(out, engine="openpyxl") as writer:
                    merged.to_excel(writer, sheet_name="NAVs", index=False)
                st.download_button(
                    label="💾 Download Excel",
                    data=out.getvalue(),
                    file_name=f"nav_comparison_{use_month.lower()}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            else:
                st.info("No data to export yet.")

# ────────────────────────────────────────────────────────────────────────────────
# Tab 2 – Month-End Data
# ────────────────────────────────────────────────────────────────────────────────
with tab_month_end:
    st.subheader("Merged Month-End NAVs")
    if not st.session_state.watchlist:
        st.info("Add funds to the watchlist.")
    else:
        frames = []
        for s in st.session_state.watchlist.values():
            if not s.month_end_df.empty:
                frames.append(s.month_end_df[["date", "nav"]].rename(columns={"nav": s.name}))
        merged = merge_frames_on_date(frames)
        if merged.empty:
            st.warning("No month-end data available.")
        else:
            merged_display = merged.copy()
            merged_display['date'] = merged_display['date'].dt.strftime('%Y-%m-%d')
            st.dataframe(merged_display, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────────
# Performance helpers (with strict start-date tolerance)
# ────────────────────────────────────────────────────────────────────────────────

def _pick_start_date_for_period(df: pd.DataFrame, end_date: pd.Timestamp, months: int, tolerance_days: int = 15) -> pd.Timestamp | None:
    """Pick the first available date >= target (end_date - months), only if
    it falls within `tolerance_days` of the target; else return None.
    Prevents showing 5Y when only ~3Y data exists.
    """
    target = end_date - DateOffset(months=months)
    mask = df['date'] >= target
    if not mask.any():
        return None
    candidate = df.loc[mask, 'date'].iloc[0]
    if (candidate - target) > pd.Timedelta(days=tolerance_days):
        return None
    return candidate


def calc_point_to_point_returns(df: pd.DataFrame) -> list[tuple]:
    """List of tuples: (Period, Return %, CAGR %, Start Date, End Date)
    - < 1 year actual data  → simple return
    - ≥ 1 year actual data  → CAGR
    - Insufficient history within tolerance → N/A
    """
    if df.empty:
        return []

    end_date = pd.to_datetime(df['date'].iloc[-1])
    end_nav = float(df['nav'].iloc[-1])
    inception_date = pd.to_datetime(df['date'].iloc[0])
    inception_nav = float(df['nav'].iloc[0])

    periods = {"1M": 1, "3M": 3, "6M": 6, "1Y": 12, "3Y": 36, "5Y": 60, "10Y": 120}
    out: list[tuple] = []

    for label, months in periods.items():
        start_date = _pick_start_date_for_period(df, end_date, months, tolerance_days=15)
        if start_date is None:
            out.append((label, "N/A", "N/A", "N/A", "N/A"))
            continue
        start_nav = float(df.loc[df['date'] == start_date, 'nav'].iloc[0])
        years = (end_date - start_date).days / 365.25
        if years >= 1.0:
            cagr = ((end_nav / start_nav) ** (1/years) - 1) * 100
            out.append((label, "N/A", f"{cagr:.2f}", start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
        else:
            ret = ((end_nav / start_nav) - 1) * 100
            out.append((label, f"{ret:.2f}", "N/A", start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))

    # Since Inception
    si_years = (end_date - inception_date).days / 365.25
    if si_years >= 1.0:
        si_cagr = ((end_nav / inception_nav) ** (1/si_years) - 1) * 100
        out.append(("Since Inception", "N/A", f"{si_cagr:.2f}", inception_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
    else:
        si_ret = ((end_nav / inception_nav) - 1) * 100
        out.append(("Since Inception", f"{si_ret:.2f}", "N/A", inception_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))

    return out


def calc_calendar_year_returns(df: pd.DataFrame) -> list[tuple]:
    if df.empty:
        return []
    tmp = df.copy()
    tmp['year'] = tmp['date'].dt.year
    rows: list[tuple] = []
    for yr, grp in tmp.groupby('year'):
        if len(grp) < 2:
            continue
        start_nav = float(grp['nav'].iloc[0])
        end_nav = float(grp['nav'].iloc[-1])
        ret = ((end_nav / start_nav) - 1) * 100
        rows.append((str(yr), f"{ret:.2f}"))
    return sorted(rows, key=lambda x: x[0], reverse=True)


def calc_fin_year_returns(df: pd.DataFrame) -> list[tuple]:
    if df.empty:
        return []
    tmp = df.copy()
    # FY label FY 2023-24 means Apr 2023 to Mar 2024 → qyear = 2024
    tmp['fy'] = tmp['date'].dt.to_period('Q-MAR').apply(lambda p: p.qyear)
    rows: list[tuple] = []
    for fy, grp in tmp.groupby('fy'):
        if len(grp) < 2:
            continue
        start_nav = float(grp['nav'].iloc[0])
        end_nav = float(grp['nav'].iloc[-1])
        ret = ((end_nav / start_nav) - 1) * 100
        label = f"FY {fy-1}-{str(fy)[-2:]}"
        rows.append((label, f"{ret:.2f}"))
    return sorted(rows, key=lambda x: x[0], reverse=True)

# ────────────────────────────────────────────────────────────────────────────────
# Tab 3 – Performance Analysis
# ────────────────────────────────────────────────────────────────────────────────
with tab_performance:
    st.subheader("Performance Analysis")
    if not st.session_state.watchlist:
        st.info("Add at least one scheme to analyze.")
    else:
        scheme_names = sorted([s.name for s in st.session_state.watchlist.values()])
        name_to_code = {s.name: s.code for s in st.session_state.watchlist.values()}
        pick = st.selectbox("Select Scheme", scheme_names)

        mode = st.radio("Date Mode", ["Since Inception", "Custom Range"], horizontal=True)
        colA, colB = st.columns(2)
        if mode == "Custom Range":
            start_str = colA.text_input("Start (YYYY-MM-DD)", value="2015-01-01")
            end_str = colB.text_input("End (YYYY-MM-DD)", value=datetime.today().strftime('%Y-%m-%d'))
        else:
            start_str = None
            end_str = None

        if st.button("Calculate Performance", type="primary"):
            code = name_to_code[pick]
            data = st.session_state.watchlist[code]
            df = data.month_end_df.copy()
            if df.empty:
                st.error("No month-end NAV data available for the selected scheme.")
            else:
                if mode == "Since Inception":
                    start_date = df['date'].iloc[0]
                    end_date = df['date'].iloc[-1]
                else:
                    try:
                        start_date = pd.to_datetime(start_str)
                        end_date = pd.to_datetime(end_str)
                    except Exception as e:
                        st.error(f"Invalid date format. Use YYYY-MM-DD. Error: {e}")
                        st.stop()

                filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
                if filtered.empty:
                    st.error("No NAV data available in the specified range.")
                    st.stop()

                # Period returns
                pr = calc_point_to_point_returns(filtered)
                pr_df = pd.DataFrame(pr, columns=["Period", "Return (%)", "CAGR (%)", "Start Date", "End Date"])
                st.markdown("### Period Returns")
                st.dataframe(pr_df, use_container_width=True)

                # Calendar year
                cy = calc_calendar_year_returns(filtered)
                cy_df = pd.DataFrame(cy, columns=["Year", "Return (%)"]) if cy else pd.DataFrame(columns=["Year", "Return (%)"]) 
                st.markdown("### Calendar Year Returns")
                st.dataframe(cy_df, use_container_width=True)

                # Financial year
                fy = calc_fin_year_returns(filtered)
                fy_df = pd.DataFrame(fy, columns=["Financial Year", "Return (%)"]) if fy else pd.DataFrame(columns=["Financial Year", "Return (%)"]) 
                st.markdown("### Financial Year Returns")
                st.dataframe(fy_df, use_container_width=True)

                # Optional export
                out = io.BytesIO()
                with pd.ExcelWriter(out, engine="openpyxl") as writer:
                    pr_df.to_excel(writer, sheet_name="Period_Returns", index=False)
                    cy_df.to_excel(writer, sheet_name="Calendar_Year", index=False)
                    fy_df.to_excel(writer, sheet_name="Financial_Year", index=False)
                st.download_button(
                    label="💾 Download Performance (Excel)",
                    data=out.getvalue(),
                    file_name=f"{pick.replace(' ', '_')}_performance.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
