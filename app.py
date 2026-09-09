"""
AeroPulse Intelligence — Real-World Aviation Delay & Operations Platform.

Interactive Executive Dashboard and Machine Learning Inference Engine
powered by real-world U.S. Department of Transportation / Kaggle flight operations data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ml.features import FAA_DELAY_COST_PER_MINUTE_USD
from src.ml.predict import FlightDelayPredictor

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="AeroPulse | Flight Operations & Delay Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global Styling & Glassmorphism CSS ───────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main {
        background: #0B0E14;
    }

    /* Glassmorphism KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.85) 0%, rgba(13, 17, 23, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        border-color: rgba(79, 142, 247, 0.4);
        transform: translateY(-2px);
    }
    .kpi-title {
        color: #8B949E;
        font-size: 0.80rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .kpi-value {
        color: #FFFFFF;
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
    }
    .kpi-delta-good {
        color: #3FB950;
        font-size: 0.82rem;
        font-weight: 500;
    }
    .kpi-delta-bad {
        color: #F85149;
        font-size: 0.82rem;
        font-weight: 500;
    }
    .kpi-subtext {
        color: #6E7681;
        font-size: 0.75rem;
        margin-top: 4px;
    }

    /* Recommendation card */
    .rec-card {
        background: rgba(22, 27, 34, 0.7);
        border-left: 4px solid #4F8EF7;
        border-radius: 0 10px 10px 0;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    .rec-tag {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: #58A6FF;
        text-transform: uppercase;
    }
    .rec-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #F0F6FC;
        margin: 4px 0 6px 0;
    }
    .rec-body {
        font-size: 0.86rem;
        color: #C9D1D9;
        line-height: 1.5;
    }
    .rec-roi {
        font-size: 0.82rem;
        font-weight: 600;
        color: #3FB950;
        margin-top: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Data Caching ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "flights_clean.parquet"
COMPAT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "flights_2024_01_clean.parquet"
METADATA_PATH = PROJECT_ROOT / "models" / "training_metadata.json"


@st.cache_data(show_spinner="Loading real-world flight operations dataset...")
def load_dataset() -> pd.DataFrame:
    path = DATA_PATH if DATA_PATH.exists() else COMPAT_DATA_PATH
    if not path.exists():
        st.error(f"Processed flight dataset not found at {path}. Run: python -m src.transform.clean_flight_data")
        st.stop()

    cols = [
        "flight_date",
        "reporting_airline_code",
        "reporting_airline_name",
        "flight_number",
        "origin_airport_code",
        "origin_city_name",
        "destination_airport_code",
        "destination_city_name",
        "scheduled_departure_hour",
        "departure_time_block",
        "scheduled_arrival_hour",
        "departure_delay_minutes",
        "departure_delay_minutes_signed",
        "arrival_delay_minutes",
        "arrival_delay_minutes_signed",
        "departure_delayed_15",
        "arrival_delayed_15",
        "arrival_on_time",
        "flight_status",
        "cancelled",
        "diverted",
        "carrier_delay_minutes",
        "weather_delay_minutes",
        "national_air_system_delay_minutes",
        "security_delay_minutes",
        "late_aircraft_delay_minutes",
        "distance_miles",
        "distance_group",
        "route_code",
        "day_of_week",
        "is_weekend",
    ]
    df = pd.read_parquet(path, columns=[c for c in cols if c in pd.read_parquet(path).columns])
    df["flight_date"] = pd.to_datetime(df["flight_date"])
    return df


@st.cache_resource
def get_predictor() -> FlightDelayPredictor:
    predictor = FlightDelayPredictor()
    return predictor


@st.cache_data
def load_model_metadata() -> dict:
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            return json.load(f)
    return {}


df_full = load_dataset()
predictor = get_predictor()
metadata = load_model_metadata()

# ─── Sidebar Controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
            <div style="background: #1F6FEB; border-radius: 8px; width: 36px; height: 36px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem;">
                ✈️
            </div>
            <div>
                <div style="font-weight: 800; font-size: 1.15rem; color: #FFFFFF; letter-spacing: -0.02em;">AEROPULSE</div>
                <div style="font-size: 0.70rem; color: #8B949E; text-transform: uppercase; letter-spacing: 0.08em;">Operations Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🎛️ Data Filters")

    # Airline filter
    available_airlines = sorted(df_full["reporting_airline_code"].dropna().unique())
    selected_airlines = st.multiselect(
        "Reporting Carrier",
        options=available_airlines,
        default=[],
        placeholder="All Carriers (14 Major Airlines)",
        help="Select one or more reporting airlines to filter metrics.",
    )

    # Top origin airports
    top_airports = df_full["origin_airport_code"].value_counts().head(30).index.tolist()
    selected_origins = st.multiselect(
        "Origin Hub",
        options=top_airports,
        default=[],
        placeholder="All Origin Hubs (300+ Airports)",
        help="Filter by primary departure airport.",
    )

    # Date range
    min_date = df_full["flight_date"].min().date()
    max_date = df_full["flight_date"].max().date()
    date_range = st.date_input(
        "Operational Window",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.75rem; color: #8B949E; line-height: 1.4;">
            <b style="color: #C9D1D9;">Dataset:</b> U.S. DOT On-Time Performance<br>
            <b style="color: #C9D1D9;">Source:</b> Kaggle / BTS Verified Extract<br>
            <b style="color: #C9D1D9;">Volume:</b> 469,968 Flight Records<br>
            <b style="color: #C9D1D9;">Cost Basis:</b> FAA $101.90 / min
        </div>
        """,
        unsafe_allow_html=True,
    )

# Filter dataset based on sidebar
filtered_df = df_full.copy()
if selected_airlines:
    filtered_df = filtered_df[filtered_df["reporting_airline_code"].isin(selected_airlines)]
if selected_origins:
    filtered_df = filtered_df[filtered_df["origin_airport_code"].isin(selected_origins)]
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_d, end_d = date_range
    filtered_df = filtered_df[
        (filtered_df["flight_date"].dt.date >= start_d)
        & (filtered_df["flight_date"].dt.date <= end_d)
    ]

# ─── Navigation Tabs ──────────────────────────────────────────────────────────
tab_kpi, tab_predictor, tab_benchmark, tab_diagnostics, tab_sql = st.tabs(
    [
        "📊 Executive KPI Command Center",
        "🎯 ML Delay Predictor",
        "🏆 Carrier & Airport Benchmarks",
        "🔬 Diagnostics & Business Recommendations",
        "💾 SQL & Star Schema Intelligence",
    ]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: EXECUTIVE KPI COMMAND CENTER
# ─────────────────────────────────────────────────────────────────────────────
with tab_kpi:
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h2 style="margin: 0; color: #FFFFFF; font-weight: 700;">Executive Aviation Reliability Command Center</h2>
            <p style="color: #8B949E; margin: 4px 0 0 0; font-size: 0.90rem;">
                Macro-level on-time performance (OTP-15), delay driver decomposition, and financial impact across U.S. airspace.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Compute Core KPIs
    total_flights = len(filtered_df)
    completed_mask = filtered_df["flight_status"] == "Completed"
    completed_df = filtered_df[completed_mask]
    cancelled_count = int((filtered_df["flight_status"] == "Cancelled").sum())
    diverted_count = int((filtered_df["flight_status"] == "Diverted").sum())

    total_completed = len(completed_df)
    on_time_arrivals = int((completed_df["arrival_on_time"] == True).sum())
    otp_pct = (on_time_arrivals / total_completed * 100) if total_completed else 0.0

    delayed_flights_count = int((completed_df["arrival_delayed_15"] == 1).sum())
    avg_arr_delay = float(completed_df["arrival_delay_minutes"].mean()) if total_completed else 0.0
    avg_dep_delay = float(completed_df["departure_delay_minutes"].mean()) if total_completed else 0.0
    total_delay_minutes = float(completed_df["arrival_delay_minutes"].sum())
    total_cost_usd = total_delay_minutes * FAA_DELAY_COST_PER_MINUTE_USD

    cancellation_rate = (cancelled_count / total_flights * 100) if total_flights else 0.0

    # Top KPI Ribbon
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Flight Volume</div>
                <div class="kpi-value">{total_flights:,}</div>
                <div class="kpi-subtext">Completed: {total_completed:,} ({total_completed/total_flights*100:.1f}%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        otp_color = "kpi-delta-good" if otp_pct >= 80 else "kpi-delta-bad"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Arrival OTP-15</div>
                <div class="kpi-value">{otp_pct:.1f}%</div>
                <div class="{otp_color}">Delayed: {delayed_flights_count:,} ({100-otp_pct:.1f}%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Avg Arrival Delay</div>
                <div class="kpi-value">{avg_arr_delay:.1f} <span style="font-size:1rem;color:#8B949E;">min</span></div>
                <div class="kpi-subtext">Avg Departure: {avg_dep_delay:.1f} min</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Financial Cost Impact</div>
                <div class="kpi-value">${total_cost_usd/1e6:,.1f}M</div>
                <div class="kpi-subtext">FAA $101.90/min benchmark</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c5:
        canc_color = "kpi-delta-good" if cancellation_rate < 2.0 else "kpi-delta-bad"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Cancellation Rate</div>
                <div class="kpi-value">{cancellation_rate:.2f}%</div>
                <div class="{canc_color}">{cancelled_count:,} flights cancelled</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # ── Charts Row 1: Temporal Cascade & Cause Decomposition ─────────────────
    ch_col1, ch_col2 = st.columns([3, 2])

    with ch_col1:
        # Hourly Delay Cascade: Delay accumulates as day progresses
        hourly = (
            completed_df.groupby("scheduled_departure_hour")
            .agg(
                avg_delay=("arrival_delay_minutes", "mean"),
                delay_rate=("arrival_delayed_15", lambda x: (x == 1).mean() * 100),
                flight_volume=("flight_number", "count"),
            )
            .reset_index()
        )
        hourly = hourly[hourly["scheduled_departure_hour"].between(5, 23)]

        fig_cascade = go.Figure()
        fig_cascade.add_trace(
            go.Bar(
                x=hourly["scheduled_departure_hour"],
                y=hourly["flight_volume"],
                name="Flight Volume",
                marker_color="rgba(79, 142, 247, 0.25)",
                yaxis="y2",
            )
        )
        fig_cascade.add_trace(
            go.Scatter(
                x=hourly["scheduled_departure_hour"],
                y=hourly["avg_delay"],
                name="Avg Arrival Delay (min)",
                line=dict(color="#F85149", width=3),
                mode="lines+markers",
            )
        )
        fig_cascade.add_trace(
            go.Scatter(
                x=hourly["scheduled_departure_hour"],
                y=hourly["delay_rate"],
                name="Delayed Flight Rate (% ≥15m)",
                line=dict(color="#D29922", width=2, dash="dot"),
                mode="lines",
            )
        )

        fig_cascade.update_layout(
            title=dict(text="<b>Time-of-Day Delay Cascade Effect</b> (Morning Punctuality → Evening Compounding)", font=dict(size=14, color="#F0F6FC")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(22, 27, 34, 0.4)",
            xaxis=dict(title="Scheduled Departure Hour (24h)", gridcolor="rgba(255,255,255,0.06)", tickmode="linear", dtick=1),
            yaxis=dict(title="Delay Duration (Minutes) / Rate (%)", gridcolor="rgba(255,255,255,0.06)"),
            yaxis2=dict(title="Flight Volume", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40),
            height=380,
        )
        st.plotly_chart(fig_cascade, use_container_width=True)

    with ch_col2:
        # Delay Cause Breakdown
        delay_causes = {
            "Late Aircraft Cascade": completed_df["late_aircraft_delay_minutes"].sum(),
            "Air Carrier Control": completed_df["carrier_delay_minutes"].sum(),
            "National Airspace (NAS)": completed_df["national_air_system_delay_minutes"].sum(),
            "Severe Weather": completed_df["weather_delay_minutes"].sum(),
            "Security Incident": completed_df["security_delay_minutes"].sum(),
        }
        total_causes = sum(delay_causes.values())
        cause_df = pd.DataFrame(
            [
                {"Cause": k, "Minutes": v, "Percentage": (v / total_causes * 100) if total_causes else 0}
                for k, v in delay_causes.items()
            ]
        ).sort_values("Minutes", ascending=True)

        fig_cause = px.bar(
            cause_df,
            x="Minutes",
            y="Cause",
            orientation="h",
            text=cause_df["Percentage"].apply(lambda p: f"{p:.1f}%"),
            color="Percentage",
            color_continuous_scale=["#1F6FEB", "#F85149"],
            title="<b>Root-Cause Delay Breakdown</b>",
        )
        fig_cause.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(22, 27, 34, 0.4)",
            coloraxis_showscale=False,
            xaxis=dict(title="Total Delay Minutes", gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title=""),
            margin=dict(l=40, r=40, t=60, b=40),
            height=380,
        )
        st.plotly_chart(fig_cause, use_container_width=True)

    # ── Daily Volume & OTP Trend ──────────────────────────────────────────────
    daily = (
        completed_df.groupby(completed_df["flight_date"].dt.date)
        .agg(
            total_flights=("flight_number", "count"),
            otp=("arrival_on_time", lambda x: (x == True).mean() * 100),
            avg_delay=("arrival_delay_minutes", "mean"),
        )
        .reset_index()
    )

    fig_daily = go.Figure()
    fig_daily.add_trace(
        go.Bar(
            x=daily["flight_date"],
            y=daily["total_flights"],
            name="Daily Completed Flights",
            marker_color="rgba(56, 139, 253, 0.35)",
            yaxis="y2",
        )
    )
    fig_daily.add_trace(
        go.Scatter(
            x=daily["flight_date"],
            y=daily["otp"],
            name="On-Time Performance (%)",
            line=dict(color="#3FB950", width=3),
            mode="lines+markers",
        )
    )
    fig_daily.update_layout(
        title=dict(text="<b>Daily Flight Volume & On-Time Performance Trend</b>", font=dict(size=14, color="#F0F6FC")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22, 27, 34, 0.4)",
        xaxis=dict(title="Date", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="OTP-15 (%)", range=[50, 100], gridcolor="rgba(255,255,255,0.06)"),
        yaxis2=dict(title="Total Flights", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=320,
    )
    st.plotly_chart(fig_daily, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: INTERACTIVE ML DELAY PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
with tab_predictor:
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h2 style="margin: 0; color: #FFFFFF; font-weight: 700;">Live Predictive Delay Simulator</h2>
            <p style="color: #8B949E; margin: 4px 0 0 0; font-size: 0.90rem;">
                Run real-time inference using trained Histogram Gradient Boosting models (GBM) on the out-of-time test distribution.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pred_c1, pred_c2 = st.columns([1, 2])

    with pred_c1:
        st.markdown("#### Flight Parameters")

        sim_carrier = st.selectbox(
            "Airline Carrier",
            options=["AA", "DL", "UA", "WN", "B6", "AS", "NK", "F9", "HA", "VX", "OO", "EV", "MQ", "US"],
            index=0,
            help="Operating air carrier (IATA code)",
        )

        top_hubs = ["ATL", "ORD", "DFW", "DEN", "LAX", "JFK", "SFO", "SEA", "LAS", "MCO", "EWR", "CLT", "PHX", "IAH", "BOS"]
        sim_origin = st.selectbox("Origin Airport", options=top_hubs, index=5)  # JFK
        sim_dest = st.selectbox("Destination Airport", options=top_hubs, index=4)  # LAX

        sim_hour = st.slider("Scheduled Departure Hour", min_value=5, max_value=23, value=17, format="%02d:00")
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        sim_day_str = st.selectbox("Day of Week", options=day_names, index=4)  # Friday
        sim_day_num = day_names.index(sim_day_str) + 1
        sim_is_weekend = sim_day_num in [6, 7]

        # Calculate approximate distance group
        sim_dist_group = st.slider("Distance Group (1=Short Haul <250mi, 11=Transcontinental >2500mi)", 1, 11, 10)

        run_pred_btn = st.button("🚀 Run Real-Time Prediction", use_container_width=True, type="primary")

    with pred_c2:
        # Execute prediction
        pred_result = predictor.predict(
            carrier=sim_carrier,
            origin=sim_origin,
            destination=sim_dest,
            scheduled_hour=sim_hour,
            day_of_week=sim_day_num,
            distance_group=sim_dist_group,
            is_weekend=sim_is_weekend,
        )

        risk = pred_result["risk_level"]
        prob_pct = pred_result["delay_probability_pct"]
        est_min = pred_result["estimated_delay_minutes"]
        cost = pred_result["cost_impact_usd"]

        # Color based on risk
        if risk == "low":
            risk_color = "#3FB950"
            risk_badge = "LOW RISK"
        elif risk == "moderate":
            risk_color = "#D29922"
            risk_badge = "MODERATE RISK"
        else:
            risk_color = "#F85149"
            risk_badge = "HIGH RISK"

        st.markdown(
            f"""
            <div style="background: rgba(22, 27, 34, 0.9); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 24px; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div style="font-size: 1.1rem; font-weight: 700; color: #FFFFFF;">
                        {sim_carrier} Flight: {sim_origin} → {sim_dest}
                    </div>
                    <div style="background: {risk_color}22; border: 1px solid {risk_color}; color: {risk_color}; padding: 4px 12px; border-radius: 20px; font-weight: 700; font-size: 0.80rem;">
                        {risk_badge}
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 16px;">
                    <div>
                        <div class="kpi-title">Delay Probability</div>
                        <div style="font-size: 2.2rem; font-weight: 800; color: {risk_color};">{prob_pct:.1f}%</div>
                        <div class="kpi-subtext">Arrival delay ≥ 15 min</div>
                    </div>
                    <div>
                        <div class="kpi-title">Estimated Delay</div>
                        <div style="font-size: 2.2rem; font-weight: 800; color: #FFFFFF;">{est_min:.1f} <span style="font-size:1.1rem; color:#8B949E;">min</span></div>
                        <div class="kpi-subtext">GBM Duration Regression</div>
                    </div>
                    <div>
                        <div class="kpi-title">Direct Cost Exposure</div>
                        <div style="font-size: 2.2rem; font-weight: 800; color: #FFFFFF;">${cost:,.0f}</div>
                        <div class="kpi-subtext">FAA $101.90 / min rate</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Visual Gauge Chart for Delay Probability
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob_pct,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "<b>Operational Delay Risk Gauge</b>", "font": {"size": 14, "color": "#F0F6FC"}},
                number={"suffix": "%", "font": {"color": "#FFFFFF", "size": 32}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#8B949E"},
                    "bar": {"color": risk_color},
                    "bgcolor": "rgba(255,255,255,0.05)",
                    "borderwidth": 1,
                    "bordercolor": "rgba(255,255,255,0.1)",
                    "steps": [
                        {"range": [0, 20], "color": "rgba(63, 185, 80, 0.15)"},
                        {"range": [20, 40], "color": "rgba(210, 153, 34, 0.15)"},
                        {"range": [40, 100], "color": "rgba(248, 81, 73, 0.15)"},
                    ],
                    "threshold": {
                        "line": {"color": "#F85149", "width": 3},
                        "thickness": 0.75,
                        "value": 40,
                    },
                },
            )
        )
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=40, b=20),
            height=220,
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Operational Advice
        st.info(f"💡 **Operations Intelligence:** {pred_result['interpretation']}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: CARRIER & AIRPORT BENCHMARKS
# ─────────────────────────────────────────────────────────────────────────────
with tab_benchmark:
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h2 style="margin: 0; color: #FFFFFF; font-weight: 700;">Carrier & Airport Reliability Benchmarks</h2>
            <p style="color: #8B949E; margin: 4px 0 0 0; font-size: 0.90rem;">
                Head-to-head performance rankings, airport hub congestion bottlenecks, and route vulnerability.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    bench_c1, bench_c2 = st.columns(2)

    with bench_c1:
        # Carrier Performance Ranking
        carrier_perf = (
            completed_df.groupby("reporting_airline_code")
            .agg(
                total_flights=("flight_number", "count"),
                otp=("arrival_on_time", lambda x: (x == True).mean() * 100),
                avg_delay=("arrival_delay_minutes", "mean"),
            )
            .reset_index()
            .sort_values("otp", ascending=True)
        )

        fig_carrier = px.bar(
            carrier_perf,
            x="otp",
            y="reporting_airline_code",
            orientation="h",
            text=carrier_perf["otp"].apply(lambda x: f"{x:.1f}%"),
            color="otp",
            color_continuous_scale=["#F85149", "#D29922", "#3FB950"],
            title="<b>Airline On-Time Performance (OTP-15) Ranking</b>",
        )
        fig_carrier.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(22, 27, 34, 0.4)",
            xaxis=dict(title="On-Time Arrival Rate (%)", range=[50, 100], gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="Airline Code"),
            coloraxis_showscale=False,
            margin=dict(l=40, r=40, t=60, b=40),
            height=400,
        )
        st.plotly_chart(fig_carrier, use_container_width=True)

    with bench_c2:
        # Top 15 Congested Hubs (Arrival Delays)
        hub_perf = (
            completed_df.groupby("origin_airport_code")
            .agg(
                flight_volume=("flight_number", "count"),
                delay_rate=("arrival_delayed_15", lambda x: (x == 1).mean() * 100),
                avg_delay=("arrival_delay_minutes", "mean"),
            )
            .reset_index()
        )
        top_hubs_perf = hub_perf[hub_perf["flight_volume"] >= 2000].sort_values("delay_rate", ascending=False).head(15)

        fig_hub = px.bar(
            top_hubs_perf,
            x="delay_rate",
            y="origin_airport_code",
            orientation="h",
            text=top_hubs_perf["delay_rate"].apply(lambda x: f"{x:.1f}%"),
            color="delay_rate",
            color_continuous_scale=["#3FB950", "#F85149"],
            title="<b>Top 15 Most Congested Departure Hubs (≥2,000 flights)</b>",
        )
        fig_hub.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(22, 27, 34, 0.4)",
            xaxis=dict(title="Delayed Flight Rate (% ≥15m)", gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="Airport Code", autorange="reversed"),
            coloraxis_showscale=False,
            margin=dict(l=40, r=40, t=60, b=40),
            height=400,
        )
        st.plotly_chart(fig_hub, use_container_width=True)

    # Route Risk Corridor Matrix
    st.markdown("#### High-Risk Operational Corridors")
    route_stats = (
        completed_df.groupby("route_code")
        .agg(
            volume=("flight_number", "count"),
            avg_delay=("arrival_delay_minutes", "mean"),
            delay_rate=("arrival_delayed_15", lambda x: (x == 1).mean() * 100),
            total_delay_cost=("arrival_delay_minutes", lambda x: x.sum() * FAA_DELAY_COST_PER_MINUTE_USD),
        )
        .reset_index()
    )
    high_volume_routes = route_stats[route_stats["volume"] >= 200].sort_values("avg_delay", ascending=False).head(20)

    fig_routes = px.scatter(
        high_volume_routes,
        x="volume",
        y="avg_delay",
        size="total_delay_cost",
        color="delay_rate",
        hover_name="route_code",
        color_continuous_scale=["#3FB950", "#D29922", "#F85149"],
        title="<b>Route Risk Matrix: Volume vs. Average Delay Duration</b>",
        labels={"volume": "Monthly Flight Volume", "avg_delay": "Average Delay (Minutes)", "delay_rate": "Delay Rate (%)"},
    )
    fig_routes.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22, 27, 34, 0.4)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        height=400,
    )
    st.plotly_chart(fig_routes, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: DIAGNOSTICS & BUSINESS RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
with tab_diagnostics:
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h2 style="margin: 0; color: #FFFFFF; font-weight: 700;">Model Diagnostics, Error Analysis & Business ROI</h2>
            <p style="color: #8B949E; margin: 4px 0 0 0; font-size: 0.90rem;">
                Rigorous evaluation across baselines, out-of-time test validation, residual analysis, and actionable business decisions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    diag_c1, diag_c2 = st.columns([3, 2])

    with diag_c1:
        st.markdown("#### Out-of-Time Model Benchmark Comparison")
        st.markdown(
            """
            <div style="font-size: 0.85rem; color: #8B949E; margin-bottom: 12px;">
                Models are evaluated on a strict <b>chronological out-of-time split</b> (Days 1–23 train: 345,440 flights; Days 24–31 test: 111,573 flights) to prevent temporal data leakage.
            </div>
            """,
            unsafe_allow_html=True,
        )

        reg_results = metadata.get("regression_results", [])
        if reg_results:
            bench_df = pd.DataFrame(reg_results)
            bench_df.columns = ["Model Architecture", "MAE (min)", "RMSE (min)", "Median AE (min)", "MAPE (%)", "R²"]
            st.dataframe(bench_df, use_container_width=True, hide_index=True)
        else:
            st.info("Run `python -m src.ml.train` to view the benchmark comparison table.")

        st.markdown("#### Classification Performance (Delay ≥ 15 min)")
        cls_results = metadata.get("classification_results", [])
        if cls_results:
            cls_df = pd.DataFrame(cls_results)
            cls_df.columns = ["Model Architecture", "ROC-AUC", "Accuracy", "Precision", "Recall", "F1-Score"]
            st.dataframe(cls_df, use_container_width=True, hide_index=True)

        st.markdown("#### Analytical Discussion: Error Characteristics & Heavy-Tail Delays")
        st.markdown(
            """
            > **Why is RMSE notably higher than MAE?**
            > 
            > In commercial aviation, delay distributions exhibit severe **positive skewness (heavy right-tail)**. While >78% of flights arrive within ±10 minutes of schedule (driving median absolute error near zero), a small fraction of flights encounter catastrophic multi-hour delays caused by ATC ground stops or severe winter storms. 
            > Because RMSE squares errors, these black swan delays penalize RMSE heavily. Using **MAE and Median Absolute Error** provides a much more robust operational metric for daily flight scheduling.
            """
        )

    with diag_c2:
        st.markdown("#### Permutation Feature Drivers")
        feat_imp = metadata.get("feature_importance", {})
        if feat_imp:
            feat_df = pd.DataFrame(
                [{"Feature": k.replace("_", " ").title(), "Importance": v} for k, v in feat_imp.items() if v > 0]
            ).sort_values("Importance", ascending=True)

            fig_feat = px.bar(
                feat_df,
                x="Importance",
                y="Feature",
                orientation="h",
                text=feat_df["Importance"].apply(lambda v: f"{v*100:.1f}%"),
                color="Importance",
                color_continuous_scale=["#1F6FEB", "#58A6FF"],
                title="<b>Top Delay Predictors (Permutation Importance)</b>",
            )
            fig_feat.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(22, 27, 34, 0.4)",
                coloraxis_showscale=False,
                xaxis=dict(title="Normalized Importance", gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(title=""),
                margin=dict(l=40, r=40, t=60, b=40),
                height=340,
            )
            st.plotly_chart(fig_feat, use_container_width=True)

    st.markdown("---")
    st.markdown("### 💼 Actionable Business Recommendations & ROI")

    recs = metadata.get("business_recommendations", [])
    if not recs:
        from src.ml.evaluate import generate_business_recommendations
        recs = generate_business_recommendations(metadata)

    r_col1, r_col2, r_col3 = st.columns(3)
    cols_list = [r_col1, r_col2, r_col3]

    for idx, rec in enumerate(recs):
        with cols_list[idx % 3]:
            st.markdown(
                f"""
                <div class="rec-card">
                    <div class="rec-tag">{rec['category']}</div>
                    <div class="rec-title">{rec['title']}</div>
                    <div class="rec-body"><b>Observed Finding:</b> {rec['finding']}</div>
                    <div class="rec-body" style="margin-top:6px;"><b>Operational Decision:</b> {rec['recommendation']}</div>
                    <div class="rec-roi"><b>Estimated ROI:</b> {rec['estimated_roi']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: SQL & STAR SCHEMA INTELLIGENCE
# ─────────────────────────────────────────────────────────────────────────────
with tab_sql:
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h2 style="margin: 0; color: #FFFFFF; font-weight: 700;">SQL & Star Schema Intelligence</h2>
            <p style="color: #8B949E; margin: 4px 0 0 0; font-size: 0.90rem;">
                Explore analytical warehouse views, star schema design, and production SQL patterns deployed in the warehouse.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sql_c1, sql_c2 = st.columns([1, 1])

    with sql_c1:
        st.markdown("#### Dimensional Warehouse Architecture (Star Schema)")
        st.markdown(
            """
            ```
            ┌──────────────────────────────────────────────┐
            │          fact_flight (Fact Table)            │
            ├──────────────────────────────────────────────┤
            │ flight_key (PK)                              │
            │ date_key (FK)  ───────► dim_date             │
            │ airline_key (FK) ─────► dim_airline          │
            │ origin_airport_key ───► dim_airport          │
            │ dest_airport_key ─────► dim_airport          │
            │ scheduled_departure_time                     │
            │ actual_departure_time                        │
            │ arrival_delay_minutes                        │
            │ arrival_delayed_15                           │
            │ carrier_delay_minutes                        │
            │ weather_delay_minutes                        │
            │ national_air_system_delay_minutes            │
            │ late_aircraft_delay_minutes                  │
            │ total_reported_delay_minutes                 │
            └──────────────────────────────────────────────┘
            ```
            """
        )

    with sql_c2:
        st.markdown("#### Production Analytical Views")
        view_choice = st.selectbox(
            "Select Analytical SQL View to Inspect",
            options=[
                "vw_carrier_monthly_performance",
                "vw_airport_congestion_hourly",
                "vw_delay_cascade_root_cause",
                "vw_high_risk_corridors",
            ],
        )

        if view_choice == "vw_carrier_monthly_performance":
            st.code(
                """
-- Monthly Carrier On-Time Performance & Direct Financial Impact
CREATE OR REPLACE VIEW warehouse.vw_carrier_monthly_performance AS
SELECT 
    da.reporting_airline_code,
    dd.year_number,
    dd.month_number,
    COUNT(*) AS total_flights,
    SUM(CASE WHEN ff.flight_status = 'Completed' THEN 1 ELSE 0 END) AS completed_flights,
    ROUND(AVG(CASE WHEN ff.arrival_on_time = TRUE THEN 1.0 ELSE 0.0 END) * 100, 2) AS otp_15_pct,
    ROUND(AVG(ff.arrival_delay_minutes)::NUMERIC, 2) AS avg_delay_minutes,
    ROUND((SUM(ff.arrival_delay_minutes) * 101.90)::NUMERIC, 2) AS total_delay_cost_usd
FROM warehouse.fact_flight ff
JOIN warehouse.dim_airline da ON ff.airline_key = da.airline_key
JOIN warehouse.dim_date dd ON ff.date_key = dd.date_key
GROUP BY 1, 2, 3
ORDER BY otp_15_pct DESC;
                """,
                language="sql",
            )
        elif view_choice == "vw_airport_congestion_hourly":
            st.code(
                """
-- Hub Congestion Cascade by Departure Hour Block
CREATE OR REPLACE VIEW warehouse.vw_airport_congestion_hourly AS
SELECT 
    dp.airport_code,
    dp.city_name,
    ff.scheduled_departure_hour,
    COUNT(*) AS departures_count,
    ROUND(AVG(ff.departure_delay_minutes)::NUMERIC, 2) AS avg_departure_delay,
    ROUND(AVG(ff.arrival_delay_minutes)::NUMERIC, 2) AS avg_arrival_delay,
    ROUND(AVG(CASE WHEN ff.arrival_delayed_15 = TRUE THEN 1.0 ELSE 0.0 END) * 100, 2) AS delay_risk_pct
FROM warehouse.fact_flight ff
JOIN warehouse.dim_airport dp ON ff.origin_airport_key = dp.airport_key
GROUP BY 1, 2, 3
ORDER BY dp.airport_code, ff.scheduled_departure_hour;
                """,
                language="sql",
            )
        elif view_choice == "vw_delay_cascade_root_cause":
            st.code(
                """
-- Decomposition of Root-Cause Delays Across Airspace
CREATE OR REPLACE VIEW warehouse.vw_delay_cascade_root_cause AS
SELECT 
    da.reporting_airline_code,
    SUM(ff.carrier_delay_minutes) AS carrier_delay_min,
    SUM(ff.weather_delay_minutes) AS weather_delay_min,
    SUM(ff.national_air_system_delay_minutes) AS nas_delay_min,
    SUM(ff.late_aircraft_delay_minutes) AS late_aircraft_delay_min,
    ROUND(SUM(ff.late_aircraft_delay_minutes) / NULLIF(SUM(ff.total_reported_delay_minutes), 0) * 100, 2) AS cascade_pct
FROM warehouse.fact_flight ff
JOIN warehouse.dim_airline da ON ff.airline_key = da.airline_key
WHERE ff.delay_cause_reported = TRUE
GROUP BY 1
ORDER BY late_aircraft_delay_min DESC;
                """,
                language="sql",
            )
        else:
            st.code(
                """
-- Top High-Risk Operational Flight Corridors
CREATE OR REPLACE VIEW warehouse.vw_high_risk_corridors AS
SELECT 
    ff.route_code,
    orig.airport_code AS origin_code,
    dest.airport_code AS dest_code,
    COUNT(*) AS flight_count,
    ROUND(AVG(ff.arrival_delay_minutes)::NUMERIC, 2) AS avg_delay_minutes,
    ROUND(AVG(CASE WHEN ff.arrival_delayed_15 = TRUE THEN 1.0 ELSE 0.0 END) * 100, 2) AS delay_probability_pct,
    ROUND((SUM(ff.arrival_delay_minutes) * 101.90)::NUMERIC, 2) AS total_financial_exposure_usd
FROM warehouse.fact_flight ff
JOIN warehouse.dim_airport orig ON ff.origin_airport_key = orig.airport_key
JOIN warehouse.dim_airport dest ON ff.destination_airport_key = dest.airport_key
GROUP BY 1, 2, 3
HAVING COUNT(*) >= 200
ORDER BY avg_delay_minutes DESC;
                """,
                language="sql",
            )

    st.markdown("#### Live SQL Query Result Preview (Calculated over real flight records)")
    sql_preview = (
        completed_df.groupby("reporting_airline_code")
        .agg(
            Total_Flights=("flight_number", "count"),
            OTP_15_Pct=("arrival_on_time", lambda x: round((x == True).mean() * 100, 2)),
            Avg_Delay_Min=("arrival_delay_minutes", lambda x: round(x.mean(), 2)),
            Cost_Exposure_USD=("arrival_delay_minutes", lambda x: f"${x.sum() * FAA_DELAY_COST_PER_MINUTE_USD / 1e6:.2f}M"),
        )
        .reset_index()
        .sort_values("OTP_15_Pct", ascending=False)
    )
    st.dataframe(sql_preview, use_container_width=True, hide_index=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #8B949E; font-size: 0.80rem; padding: 10px 0;">
        <b>AeroPulse Aviation Intelligence</b> | End-to-End Data Science, Operations Research & Analytics Platform<br>
        Built with Streamlit, Plotly, Scikit-Learn, PyArrow & PostgreSQL | MIT Licensed
    </div>
    """,
    unsafe_allow_html=True,
)
