import html
from typing import Dict, List, Optional

import altair as alt
import folium
import numpy as np
import pandas as pd
import streamlit as st
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# =========================================================
# PAGE
# =========================================================
st.set_page_config(
    page_title="Sediment Zone Monitoring Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# STYLE
# =========================================================
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    .mini-card {
        border: 1px solid #e8edf4;
        border-radius: 14px;
        padding: 12px 14px;
        background: #ffffff;
    }

    .mini-label {
        font-size: 0.76rem;
        color: #6b7280;
        margin-bottom: 6px;
    }

    .mini-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #111827;
    }

    .top-chip {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: #f3f4f6;
        border: 1px solid #e5e7eb;
        margin: 0 8px 8px 0;
        font-size: 0.84rem;
        color: #374151;
    }

    .note-box {
        border-left: 4px solid #f59e0b;
        background: #fffaf0;
        padding: 12px 14px;
        border-radius: 8px;
    }

    /* Hide uploaded file name row */
    [data-testid="stFileUploaderFile"] {
        display: none;
    }

    /* Hide sidebar completely */
    [data-testid="stSidebar"] {
        display: none;
    }

    /* Cleaner uploader spacing */
    [data-testid="stFileUploader"] section {
        padding-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# CONFIG
# =========================================================
PARAM_LIMITS: Dict[str, float] = {
    "Lead mg/kg": 10.0,
    "Zinc mg/kg": 50.0,
    "Iron mg/kg": 2000.0,
    "Manganese mg/kg": 50.0,
    "Arsenic mg/kg": 3.0,
    "Copper mg/kg": 2.0,
    "Nickel mg/kg": 2.0,
    "Chromium mg/kg": 10.0,
    "Cadmium mg/kg": 1.0,
    "Mercury mg/kg": 0.5,
}

ZONE_COLORS = [
    "#2563eb", "#16a34a", "#dc2626", "#7c3aed", "#ea580c",
    "#0891b2", "#ca8a04", "#db2777", "#4f46e5", "#059669",
    "#9333ea", "#0f766e", "#b91c1c", "#0284c7", "#65a30d",
]

STATUS_COLORS = {
    "CLEAN": "#dcfce7",
    "HOTSPOT": "#fef3c7",
    "IMPACTED": "#fee2e2",
}

STATUS_TEXT_COLORS = {
    "CLEAN": "#166534",
    "HOTSPOT": "#92400e",
    "IMPACTED": "#991b1b",
}

# =========================================================
# HELPERS
# =========================================================
def safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def guess_date_column(df: pd.DataFrame) -> Optional[str]:
    for c in ["Sampling Date", "Date", "Received Date", "sampling_date", "date"]:
        if c in df.columns:
            return c
    return None


def detect_zone_column(df: pd.DataFrame) -> Optional[str]:
    for c in ["zone", "Zone", "ZONE"]:
        if c in df.columns:
            return c
    return None


def zone_color(zone_id: int) -> str:
    return ZONE_COLORS[int(zone_id) % len(ZONE_COLORS)]


def color_status_cell(val):
    bg = STATUS_COLORS.get(val, "#ffffff")
    fg = STATUS_TEXT_COLORS.get(val, "#111827")
    return f"background-color: {bg}; color: {fg}; font-weight: 700;"


def simple_status(percent_affected: float, affected_count: int) -> str:
    if affected_count == 0:
        return "CLEAN"
    elif percent_affected >= 30:
        return "IMPACTED"
    else:
        return "HOTSPOT"


def simple_message(status: str) -> str:
    if status == "CLEAN":
        return "No exceedance detected"
    elif status == "HOTSPOT":
        return "Exceedance is limited to a few samples"
    else:
        return "Exceedance is more widespread in this zone"


def get_active_parameters(
    analysis_mode: str,
    selected_param: str,
    all_param_cols: List[str]
) -> List[str]:
    if analysis_mode == "Selected parameter only":
        return [selected_param] if selected_param in all_param_cols else []
    return all_param_cols


def affected_params_for_sample(row: pd.Series, active_limits: Dict[str, float]) -> List[str]:
    exceeded = []
    for param, limit in active_limits.items():
        if param in row.index and pd.notna(row[param]) and row[param] > limit:
            exceeded.append(param)
    return exceeded


def build_hotspot_table(df: pd.DataFrame, active_limits: Dict[str, float]) -> pd.DataFrame:
    work = df.copy()
    work["affected_parameters"] = work.apply(
        lambda r: affected_params_for_sample(r, active_limits),
        axis=1
    )
    work["affected_count"] = work["affected_parameters"].apply(len)
    work = work[work["affected_count"] > 0].copy()
    work["affected_parameters_text"] = work["affected_parameters"].apply(lambda x: ", ".join(x))
    return work


def build_zone_summary(
    df: pd.DataFrame,
    zone_col: str,
    hotspots: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    hotspot_counts = hotspots.groupby(zone_col).size().to_dict() if not hotspots.empty else {}

    for zone, g in df.groupby(zone_col):
        total_samples = len(g)
        affected_count = int(hotspot_counts.get(zone, 0))
        percent_affected = (affected_count / total_samples * 100) if total_samples > 0 else 0.0
        status = simple_status(percent_affected, affected_count)

        projects = []
        if "Project Name" in g.columns:
            projects = sorted(g["Project Name"].dropna().astype(str).unique().tolist())

        zone_hotspots = hotspots[hotspots[zone_col] == zone].copy()
        zone_affected_params = []
        if not zone_hotspots.empty:
            all_params = set()
            for vals in zone_hotspots["affected_parameters"]:
                all_params.update(vals)
            zone_affected_params = sorted(all_params)

        rows.append({
            "Zone": int(zone),
            "Projects": ", ".join(projects) if projects else "Not available",
            "Status": status,
            "% Affected": round(percent_affected, 1),
            "Alert": "YES" if affected_count > 0 else "NO",
            "What it means": simple_message(status),
            "Affected samples": affected_count,
            "Total samples": total_samples,
            "Affected parameters": ", ".join(zone_affected_params) if zone_affected_params else "None",
            "center_lat": float(g["Latitude"].mean()) if "Latitude" in g.columns else np.nan,
            "center_lon": float(g["Longitude"].mean()) if "Longitude" in g.columns else np.nan,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values(
        ["% Affected", "Affected samples", "Zone"],
        ascending=[False, False, True]
    ).reset_index(drop=True)


def build_param_exceedance_table(df: pd.DataFrame, active_limits: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for param, limit in active_limits.items():
        if param in df.columns:
            vals = pd.to_numeric(df[param], errors="coerce").dropna()
            total = len(vals)
            affected = int((vals > limit).sum())
            rows.append({
                "Parameter": param,
                "Limit": limit,
                "Affected samples": affected,
                "Total samples": total,
                "Affected %": round((affected / total * 100), 1) if total else 0.0,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values(
        ["Affected samples", "Affected %", "Parameter"],
        ascending=[False, False, True]
    ).reset_index(drop=True)


def build_trend(df: pd.DataFrame, zone_col: str, date_col: str, trend_param: str) -> pd.DataFrame:
    if trend_param not in df.columns:
        return pd.DataFrame()

    tmp = df.dropna(subset=[date_col]).copy()
    if tmp.empty:
        return pd.DataFrame()

    return (
        tmp.groupby([date_col, zone_col])[trend_param]
        .median()
        .reset_index(name="median_value")
        .sort_values([date_col, zone_col])
    )


def build_zone_popup(zone_row: pd.Series, hotspots: pd.DataFrame, zone_col: str) -> str:
    z = int(zone_row["Zone"])
    zhot = hotspots[hotspots[zone_col] == z].copy()

    parts = [
        f"<b>Zone {z}</b>",
        f"Projects: {html.escape(str(zone_row['Projects']))}",
        f"Status: {html.escape(str(zone_row['Status']))}",
        f"% affected: {zone_row['% Affected']}%",
        f"Alert: {html.escape(str(zone_row['Alert']))}",
        f"Meaning: {html.escape(str(zone_row['What it means']))}",
        f"Affected parameters: {html.escape(str(zone_row['Affected parameters']))}",
    ]

    if not zhot.empty:
        parts.append("<br><b>Affected samples</b>")
        for _, r in zhot.head(12).iterrows():
            sample_name = html.escape(str(r.get("Sample name", "Sample")))
            proj = html.escape(str(r.get("Project Name", "")))
            affected = html.escape(str(r.get("affected_parameters_text", "")))
            parts.append(f"• {sample_name} | {proj} | {affected}")
        if len(zhot) > 12:
            parts.append(f"... and {len(zhot) - 12} more")
    else:
        parts.append("<br>No affected samples in this zone")

    return "<br>".join(parts)


def make_display_chips(items: List[str]):
    st.markdown(
        "".join([f'<span class="top-chip">{html.escape(str(c))}</span>' for c in items]),
        unsafe_allow_html=True
    )


# =========================================================
# LOAD
# =========================================================
st.title("Sediment Zone Monitoring Dashboard")

uploaded_file = st.file_uploader("", type=["xlsx"])

if uploaded_file is None:
    st.info("Upload zoned Excel file to start.")
    st.stop()

try:
    raw_df = pd.read_excel(uploaded_file, engine="openpyxl")
except Exception as e:
    st.error(f"Could not read the uploaded Excel file: {e}")
    st.stop()

raw_df.columns = (
    raw_df.columns.astype(str)
    .str.strip()
    .str.replace("\n", " ", regex=False)
    .str.replace("\r", " ", regex=False)
    .str.replace(r"\s+", " ", regex=True)
)

all_columns = raw_df.columns.tolist()

# =========================================================
# AUTO-DETECT REQUIRED COLUMNS
# =========================================================
zone_col = detect_zone_column(raw_df)
date_col = guess_date_column(raw_df)

if zone_col is None:
    st.error("Zone column was not found in the uploaded file.")
    st.stop()

available_params = [c for c in PARAM_LIMITS.keys() if c in all_columns]
if not available_params:
    st.error("No regulated parameter columns were found in the file.")
    st.stop()

# =========================================================
# MAIN FILTERS
# =========================================================
st.markdown("## 1. Filters")

col1, col2, col3 = st.columns([1.2, 1.3, 1.2])

with col1:
    selected_param = st.selectbox(
        "Parameter",
        available_params,
        index=available_params.index("Lead mg/kg") if "Lead mg/kg" in available_params else 0
    )

with col2:
    zone_options = sorted(pd.Series(raw_df[zone_col].dropna().unique()).tolist())
    selected_zones = st.multiselect(
        "Zones",
        zone_options,
        default=zone_options
    )

with col3:
    if date_col and date_col in raw_df.columns:
        raw_df[date_col] = pd.to_datetime(raw_df[date_col], errors="coerce")
        date_values = raw_df[date_col].dropna()

        if not date_values.empty:
            min_date = date_values.min().date()
            max_date = date_values.max().date()
            selected_date_range = st.date_input(
                "Date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
        else:
            selected_date_range = None
    else:
        selected_date_range = None

st.caption("Analysis scope was removed. The dashboard always analyzes the selected parameter only, while zone and date filters still control the displayed data.")

# =========================================================
# PREP DATA
# =========================================================
numeric_cols = [zone_col, "Latitude", "Longitude"] + available_params
df = safe_numeric(raw_df, numeric_cols)
df = df.dropna(subset=[zone_col]).copy()
df[zone_col] = df[zone_col].astype(int)

if {"Latitude", "Longitude"}.issubset(df.columns):
    df = df.dropna(subset=["Latitude", "Longitude"]).copy()

if date_col and date_col in df.columns:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

# =========================================================
# APPLY FILTERS
# =========================================================
if selected_zones:
    df = df[df[zone_col].isin(selected_zones)].copy()

if (
    selected_date_range
    and isinstance(selected_date_range, tuple)
    and len(selected_date_range) == 2
    and date_col
    and date_col in df.columns
):
    start_date, end_date = selected_date_range
    df = df[
        (df[date_col].dt.date >= start_date) &
        (df[date_col].dt.date <= end_date)
    ].copy()

if df.empty:
    st.warning("No data after filters.")
    st.stop()

# =========================================================
# ACTIVE ANALYSIS LOGIC
# =========================================================
active_params = [selected_param]
active_limits = {selected_param: PARAM_LIMITS[selected_param]}

hotspots = build_hotspot_table(df, active_limits)
zone_summary = build_zone_summary(df, zone_col, hotspots)
param_exceedance = build_param_exceedance_table(df, active_limits)
trend_param = selected_param
trend_df = build_trend(df, zone_col, date_col, trend_param) if date_col else pd.DataFrame()

# =========================================================
# DISPLAY ALL DATA
# =========================================================
st.markdown("## 2. All Data")
st.caption("This table shows the data after the selected zone/date filters. The selected parameter is used for summaries, map alerts, graph, and affected samples.")
st.dataframe(df, use_container_width=True)

# =========================================================
# KPI
# =========================================================
mc1, mc2, mc3, mc4 = st.columns(4)

with mc1:
    st.markdown(
        f'<div class="mini-card"><div class="mini-label">Zones</div><div class="mini-value">{zone_summary["Zone"].nunique()}</div></div>',
        unsafe_allow_html=True,
    )

with mc2:
    st.markdown(
        f'<div class="mini-card"><div class="mini-label">Samples</div><div class="mini-value">{len(df)}</div></div>',
        unsafe_allow_html=True,
    )

with mc3:
    st.markdown(
        f'<div class="mini-card"><div class="mini-label">Affected samples</div><div class="mini-value">{len(hotspots)}</div></div>',
        unsafe_allow_html=True,
    )

with mc4:
    affected_param_count = (
        param_exceedance[param_exceedance["Affected samples"] > 0].shape[0]
        if not param_exceedance.empty else 0
    )
    st.markdown(
        f'<div class="mini-card"><div class="mini-label">Affected parameters</div><div class="mini-value">{affected_param_count}</div></div>',
        unsafe_allow_html=True,
    )

# =========================================================
# ZONE DECISION SUMMARY
# =========================================================
st.markdown("## 3. Zones Summary")

simple_cols = [
    "Zone", "Projects", "Status", "% Affected", "Alert",
    "Affected samples", "Total samples", "Affected parameters", "What it means"
]

styled_zone_summary = zone_summary[simple_cols].style.map(
    color_status_cell,
    subset=["Status"]
)
st.dataframe(styled_zone_summary, use_container_width=True)

# =========================================================
# MAP
# =========================================================
st.markdown("## 4. Map")

map_mode = st.radio(
    "Map view",
    ["Zone map", "Sample map", "Affected samples only"],
    horizontal=True
)

center_lat = float(df["Latitude"].mean())
center_lon = float(df["Longitude"].mean())
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=9,
    tiles="CartoDB Positron"
)

if map_mode == "Zone map":
    cluster = MarkerCluster(name="Samples").add_to(m)

    for _, row in df.iterrows():
        z = int(row[zone_col])
        color = zone_color(z)
        affected_list = affected_params_for_sample(row, active_limits)

        popup_txt = (
            f"Zone {z}<br>"
            f"{html.escape(str(row.get('Sample name', 'Sample')))}<br>"
            f"{html.escape(str(row.get('Project Name', '')))}"
        )

        if affected_list:
            popup_txt += f"<br><b>Affected:</b> {html.escape(', '.join(affected_list))}"
        else:
            popup_txt += "<br><b>Affected:</b> None"

        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=7 if affected_list else 5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            popup=folium.Popup(popup_txt, max_width=320),
        ).add_to(cluster)

    for _, zr in zone_summary.iterrows():
        z = int(zr["Zone"])
        border = "#dc2626" if zr["Alert"] == "YES" else "#111827"
        popup_html = build_zone_popup(zr, hotspots, zone_col)

        folium.Marker(
            location=[zr["center_lat"], zr["center_lon"]],
            icon=folium.DivIcon(
                html=(
                    f"<div style='background:{zone_color(z)}; color:white; font-weight:700; "
                    f"font-size:12px; padding:4px 8px; border-radius:999px; "
                    f"border:2px solid {border}; box-shadow:0 1px 4px rgba(0,0,0,.25);'>Z{z}</div>"
                )
            ),
            popup=folium.Popup(popup_html, max_width=420),
            tooltip=f"Zone {z} | {zr['Status']}",
        ).add_to(m)

elif map_mode == "Sample map":
    for _, row in df.iterrows():
        z = int(row[zone_col])
        color = zone_color(z)
        affected_list = affected_params_for_sample(row, active_limits)
        border = "#dc2626" if affected_list else color

        popup_txt = (
            f"Zone {z}<br>"
            f"{html.escape(str(row.get('Sample name', 'Sample')))}<br>"
            f"{html.escape(str(row.get('Project Name', '')))}"
        )

        if affected_list:
            popup_txt += f"<br><b>Affected:</b> {html.escape(', '.join(affected_list))}"
        else:
            popup_txt += "<br><b>Affected:</b> None"

        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=7 if affected_list else 5,
            color=border,
            weight=3 if affected_list else 1,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=folium.Popup(popup_txt, max_width=320),
            tooltip=f"Zone {z}",
        ).add_to(m)

else:
    if hotspots.empty:
        st.info("No affected samples found for the current selected parameter.")

    for _, row in hotspots.iterrows():
        z = int(row[zone_col])
        color = zone_color(z)

        popup_txt = (
            f"Zone {z}<br>"
            f"{html.escape(str(row.get('Sample name', 'Sample')))}<br>"
            f"{html.escape(str(row.get('Project Name', '')))}<br>"
            f"<b>Affected:</b> {html.escape(str(row.get('affected_parameters_text', '')))}"
        )

        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=8,
            color="#dc2626",
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(popup_txt, max_width=320),
            tooltip=f"Zone {z}",
        ).add_to(m)

st_folium(m, width=None, height=620)

# =========================================================
# LINE GRAPH
# =========================================================
st.markdown("## 5. Line Graph")
st.caption(f"One line per zone showing median {selected_param} over time.")

if date_col and date_col in df.columns and selected_param in df.columns:
    trend_df = df[[date_col, zone_col, selected_param]].copy()
    trend_df[date_col] = pd.to_datetime(trend_df[date_col], errors="coerce")
    trend_df[selected_param] = pd.to_numeric(trend_df[selected_param], errors="coerce")
    trend_df = trend_df.dropna(subset=[date_col, zone_col, selected_param])

    if trend_df.empty:
        st.info(f"No valid time data available for {selected_param}.")
    else:
        # Monthly grouping makes sparse sediment data easier to read.
        # Change freq="M" to freq="D" if you want daily values instead.
        trend_df = (
            trend_df
            .groupby([pd.Grouper(key=date_col, freq="ME"), zone_col])[selected_param]
            .median()
            .reset_index(name="Median Value")
            .sort_values([date_col, zone_col])
        )

        trend_df["Zone"] = "Zone " + trend_df[zone_col].astype(str)

        line_chart = (
            alt.Chart(trend_df)
            .mark_line(point=True)
            .encode(
                x=alt.X(f"{date_col}:T", title="Sampling Date"),
                y=alt.Y("Median Value:Q", title=f"Median {selected_param}"),
                color=alt.Color("Zone:N", title="Zone"),
                tooltip=[
                    alt.Tooltip(f"{date_col}:T", title="Date", format="%Y-%m-%d"),
                    alt.Tooltip("Zone:N", title="Zone"),
                    alt.Tooltip("Median Value:Q", title=f"Median {selected_param}", format=".2f"),
                ],
            )
            .properties(height=460)
            .interactive()
        )

        st.altair_chart(line_chart, use_container_width=True)

        limit_value = PARAM_LIMITS.get(selected_param)
        if limit_value is not None:
            st.caption(f"Regulatory/reference limit used in this dashboard: {limit_value}")
else:
    st.info("No date column was found, so the line graph cannot be displayed.")

# =========================================================
# AFFECTED SAMPLE TABLE
# =========================================================
st.markdown("## 6. Affected Samples")

if hotspots.empty:
    st.success(f"No affected samples found for {selected_param}.")
else:
    st.caption(f"Showing samples where {selected_param} exceeds the limit of {PARAM_LIMITS[selected_param]}.")

    table_cols = [
        c for c in [
            zone_col, date_col, "Latitude", "Longitude",
            "Project Name", "Sample name", "affected_parameters_text"
        ]
        if c is not None and c in hotspots.columns
    ]

    st.dataframe(
        hotspots[table_cols + [selected_param]].rename(
            columns={"affected_parameters_text": "Affected parameters"}
        ),
        use_container_width=True
    )
