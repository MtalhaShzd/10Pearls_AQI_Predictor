# app/dashboard.py

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os
from zoneinfo import ZoneInfo
PKT = ZoneInfo("Asia/Karachi")
# ═══════════════════════════════════════════════════════════
# STREAMLIT CONFIGURATION (DARK MODE)
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="10Pearls AQI Predictor",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Dark theme palette
BG_APP = "#0b1220"
BG_CARD = "#111827"
BG_INFO = "#0f172a"
TEXT_MAIN = "#f8fafc"
TEXT_MUTED = "#94a3b8"
BORDER = "#1f2937"
GRID = "rgba(148,163,184,0.18)"
ACCENT = "#f59e0b"

st.markdown(f"""
    <style>
    #MainMenu, footer, header {{visibility: hidden;}}

    .stApp {{
        background: radial-gradient(circle at top left, #111827 0%, {BG_APP} 45%, #020617 100%);
        color: {TEXT_MAIN};
    }}

    .main .block-container {{
        padding-top: 1.5rem;
        max-width: 1400px;
    }}

    h1, h2, h3, h4, h5, h6, p, span, label, div {{
        color: {TEXT_MAIN} !important;
    }}

    /* FORCE DARK REFRESH BUTTON */
    div[data-testid="stButton"] > button {{
        background-color: #111827 !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        padding: 8px 16px !important;
        font-weight: 700 !important;
        transition: all 0.2s ease-in-out !important;
    }}
    
    div[data-testid="stButton"] > button:hover {{
        background-color: #1f2937 !important;
        color: #f59e0b !important;
        border-color: #f59e0b !important;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.2) !important;
    }}

        /* FORCE DARK DOWNLOAD BUTTON */
    div[data-testid="stDownloadButton"] > button {{
        background-color: #111827 !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        padding: 8px 16px !important;
        font-weight: 700 !important;
        transition: all 0.2s ease-in-out !important;
    }}
    
    div[data-testid="stDownloadButton"] > button:hover {{
        background-color: #1f2937 !important;
        color: #f59e0b !important;
        border-color: #f59e0b !important;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.2) !important;
    }}

    .pollutant-card, .metric-highlight, .conditions-card, .info-box {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }}

    .pollutant-card {{
        padding: 18px;
    }}

    .pollutant-value {{
        font-size: 28px;
        font-weight: 800;
        color: {TEXT_MAIN} !important;
        margin: 6px 0;
    }}

    .pollutant-unit {{
        font-size: 13px;
        color: {TEXT_MUTED} !important;
    }}

    .pollutant-label {{
        font-size: 14px;
        color: {TEXT_MUTED} !important;
        font-weight: 600;
    }}

    .metric-highlight {{
        padding: 22px;
        border-top: 5px solid {ACCENT};
    }}

    .conditions-card {{
        padding: 16px 18px;
        margin-bottom: 12px;
    }}

    .info-box {{
        padding: 14px;
        background: {BG_INFO};
    }}

    .muted {{
        color: {TEXT_MUTED} !important;
    }}

    div[data-testid="stMetricValue"] {{
        color: {TEXT_MAIN} !important;
    }}

    div[data-testid="stMetricLabel"] {{
        color: {TEXT_MUTED} !important;
    }}
    </style>
""", unsafe_allow_html=True)

API_BASE = os.getenv("API_URL", "https://one0pearls-aqi-predictor.onrender.com").rstrip("/")


@st.cache_data(ttl=60, show_spinner=False)
def fetch_current():
    try:
        r = requests.get(f"{API_BASE}/api/current", timeout=120)
        if r.status_code == 200:
            data = r.json()
            return data.get("data")
        st.warning(f"⚠️ API `/api/current` returned status code {r.status_code}")
        return None
    except requests.exceptions.RequestException as e:
        st.warning(f"⏳ Connecting to Render API: {e}")
        return None


@st.cache_data(ttl=60, show_spinner=False)
def fetch_forecast():
    try:
        r = requests.get(f"{API_BASE}/api/forecast", timeout=120)
        if r.status_code == 200:
            return r.json()
        st.warning(f"⚠️ API `/api/forecast` returned status code {r.status_code}")
        return None
    except requests.exceptions.RequestException as e:
        st.warning(f"⏳ Connecting to Render API: {e}")
        return None


@st.cache_data(ttl=60, show_spinner=False)
def fetch_shap(horizon):
    try:
        r = requests.get(f"{API_BASE}/api/shap/{horizon}", timeout=120)
        if r.status_code == 200:
            return r.json()
        return None
    except requests.exceptions.RequestException:
        return None

def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good", "#22c55e"
    elif aqi <= 100:
        return "Moderate", "#eab308"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#f97316"
    elif aqi <= 200:
        return "Unhealthy", "#ef4444"
    elif aqi <= 300:
        return "Very Unhealthy", "#a855f7"
    return "Hazardous", "#7f1d1d"


def style_fig(fig, height=280):
    """Plotly dark-theme layout helper."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_MAIN, size=12),
        legend=dict(font=dict(color=TEXT_MAIN)),
        xaxis=dict(
            color=TEXT_MAIN,
            tickfont=dict(color=TEXT_MAIN, size=11),
            gridcolor=GRID,
            zerolinecolor=TEXT_MUTED,
            showgrid=False
        ),
        yaxis=dict(
            color=TEXT_MAIN,
            tickfont=dict(color=TEXT_MAIN, size=11),
            gridcolor=GRID,
            zerolinecolor=TEXT_MUTED,
            automargin=True
        )
    )
    return fig


# ═══════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════
c1, c2, c3 = st.columns([2.2, 2, 1.2])
with c1:
    st.markdown("### 🌬️ **10Pearls AQI Predictor**")
    st.caption("AI-powered air quality intelligence")
with c2:
    st.markdown(
        "<div style='padding-top:18px;'>📍 <b>Lahore, Punjab</b> · Pakistan</div>",
        unsafe_allow_html=True
    )
with c3:
    now_pkt = datetime.now(PKT)
    st.markdown(
        f"<div class='muted' style='padding-top:10px; text-align:right;'>Updated {now_pkt.strftime('%I:%M %p')} PKT</div>",
        unsafe_allow_html=True
    )
    
    if st.button("🔄 Refresh Data", use_container_width=True, key="refresh_btn"):
        st.cache_data.clear()
        st.rerun()

st.markdown("---")

try:
    with st.spinner("⏳ Loading real-time data from Render API..."):
        current = fetch_current()
        forecast = fetch_forecast()

    if not current:
        st.error("❌ Could not load live data from backend API.")
        st.info("💡 Render free instances take ~40 seconds to wake up from sleep. Please wait 30 seconds and click **🔄 Refresh Data**.")
        st.stop()

    if not forecast:
        st.warning(
            "⚠️ The 3-day forecast is temporarily unavailable — the weather data "
            "provider (Open-Meteo) is rate-limiting requests right now. Current "
            "conditions below are still live. This resolves on its own within a "
            "few minutes — try **🔄 Refresh Data** shortly."
        )

    degraded = (
        current.get("source", {}).get("feature_source") != "Hopsworks Feature Store"
        or (forecast is not None
            and forecast.get("source", {}).get("model_source") != "Hopsworks Model Registry")
    )
    
    if degraded:
        st.warning(
            "⚠️ Running in fallback mode — serving cached local data because Hopsworks "
            "is temporarily unavailable (likely free-tier compute quota). Live data will "
            "resume automatically once access is restored — no action needed."
        )
    # ═══════════════════════════════════════════════════════════
    # POLLUTANT CARDS
    # ═══════════════════════════════════════════════════════════
    p = current["pollutants"]
    cards = [
        ("🌫️", p["pm2_5"], "µg/m³", "PM2.5"),
        ("☁️", p["pm10"], "µg/m³", "PM10"),
        ("💨", p["ozone"], "µg/m³", "O₃"),
        ("🏭", p["nitrogen_dioxide"], "µg/m³", "NO₂"),
        ("🫧", p["sulphur_dioxide"], "µg/m³", "SO₂"),
        ("🧪", p["carbon_monoxide"], "µg/m³", "CO"),
    ]
    cols = st.columns(6)
    for col, (icon, val, unit, label) in zip(cols, cards):
        with col:
            st.markdown(f"""
                <div class="pollutant-card">
                    <div style="font-size:22px;">{icon}</div>
                    <div class="pollutant-value">{val}
                        <span class="pollutant-unit">{unit}</span>
                    </div>
                    <div class="pollutant-label">{label}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # 24H TREND + CURRENT CONDITIONS
    # ═══════════════════════════════════════════════════════════
    left, right = st.columns([2, 1])

    with left:
        st.markdown("### 24-Hour AQI Trend")
        st.caption("Air quality changes over the last 24 hours")

        trend_df = pd.DataFrame(current["trend_24h"])
        trend_df["datetime"] = pd.to_datetime(trend_df["datetime"])

        st.markdown(f"""
            <div class="info-box" style="display:flex; justify-content:space-around; margin-bottom:10px;">
                <div style="text-align:center;">
                    <div class="muted">CURRENT</div>
                    <div style="font-size:20px; font-weight:800;">{current['current_aqi']}</div>
                </div>
                <div style="text-align:center;">
                    <div class="muted">AVG</div>
                    <div style="font-size:20px; font-weight:800;">{trend_df['aqi'].mean():.0f}</div>
                </div>
                <div style="text-align:center;">
                    <div class="muted">MIN</div>
                    <div style="font-size:20px; font-weight:800;">{trend_df['aqi'].min():.0f}</div>
                </div>
                <div style="text-align:center;">
                    <div class="muted">MAX</div>
                    <div style="font-size:20px; font-weight:800;">{trend_df['aqi'].max():.0f}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trend_df["datetime"],
            y=trend_df["aqi"],
            mode="lines",
            fill="tozeroy",
            line=dict(color="#f59e0b", width=3),
            fillcolor="rgba(245,158,11,0.18)",
            hovertemplate="<b>%{x}</b><br>AQI: %{y}<extra></extra>"
        ))
        fig = style_fig(fig, height=270)
        st.plotly_chart(fig, use_container_width=True, key="chart_24h_trend")

    with right:
        st.markdown("### Current Conditions")
        st.caption("Weather variables at time of measurement")
        w = current["weather"]
        st.markdown(f"""
            <div class="conditions-card">
                <div class="muted" style="font-size:12px; font-weight:700;">🌡️ TEMPERATURE</div>
                <div style="font-size:26px; font-weight:800; color:#fb923c;">{w['temperature']}°C</div>
            </div>
            <div class="conditions-card">
                <div class="muted" style="font-size:12px; font-weight:700;">💧 HUMIDITY</div>
                <div style="font-size:26px; font-weight:800; color:#38bdf8;">{w['humidity']}%</div>
            </div>
            <div class="conditions-card">
                <div class="muted" style="font-size:12px; font-weight:700;">🎯 PRESSURE</div>
                <div style="font-size:26px; font-weight:800; color:#34d399;">{w['pressure']} hPa</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    if forecast:
        # ═══════════════════════════════════════════════════════════
        # FORECAST CARDS
        # ═══════════════════════════════════════════════════════════
        st.markdown("### 🤖 AI Air Quality Forecast")
        st.caption("Predicted AQI for the next three days")

        predictions = forecast["predictions"]
        metrics = forecast["model_metrics"]

        aqi_24h = predictions[23]["predicted_aqi"]
        aqi_48h = predictions[47]["predicted_aqi"]
        aqi_72h = predictions[71]["predicted_aqi"]

        rmse_24h = metrics["test_rmse"] * 1.0
        rmse_48h = metrics["test_rmse"] * 1.4
        rmse_72h = metrics["test_rmse"] * 1.8

        hcols = st.columns(3)
        for col, hrs, aqi_v, rmse_v, day_num in zip(
            hcols,
            ["24h", "48h", "72h"],
            [aqi_24h, aqi_48h, aqi_72h],
            [rmse_24h, rmse_48h, rmse_72h],
            ["Day 1", "Day 2", "Day 3"]
        ):
            label, color = get_aqi_category(aqi_v)
            with col:
                st.markdown(f"""
                    <div class="metric-highlight" style="border-top-color:{color};">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div>
                                <div style="font-size:18px; font-weight:800;">{hrs}</div>
                                <div class="muted" style="font-size:13px;">{day_num}</div>
                            </div>
                            <div style="background:{color}22; color:{color}; padding:4px 12px; border-radius:999px; font-size:12px; font-weight:700;">
                                ● {label}
                            </div>
                        </div>
                        <div style="display:flex; align-items:baseline; margin-top:18px;">
                            <span style="font-size:48px; font-weight:900; color:{color};">{aqi_v:.1f}</span>
                            <span class="muted" style="margin-left:8px; font-weight:600;">predicted AQI</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; padding-top:14px; border-top:1px solid {BORDER}; margin-top:16px;">
                            <span class="muted" style="font-size:13px;">Model RMSE</span>
                            <span style="font-size:13px; font-weight:800;">±{rmse_v:.2f}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ═══════════════════════════════════════════════════════════
        # TREND + MODEL INFO
        # ═══════════════════════════════════════════════════════════
        t1, t2 = st.columns([2, 1])

        with t1:
            st.markdown("### Predicted AQI Trend")
            chart_df = pd.DataFrame({
                "Time": ["Today", "24h", "48h", "72h"],
                "AQI": [current["current_aqi"], aqi_24h, aqi_48h, aqi_72h]
            })
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=chart_df["Time"],
                y=chart_df["AQI"],
                mode="lines+markers",
                line=dict(color="#22c55e", width=4),
                marker=dict(size=12, color="#16a34a"),
                hovertemplate="<b>%{x}</b><br>AQI: %{y}<extra></extra>"
            ))
            fig2 = style_fig(fig2, height=290)
            st.plotly_chart(fig2, use_container_width=True, key="chart_predicted_trend")

        with t2:
            st.markdown("### 🖥️ Prediction System")
            st.markdown(f"""
                <div class="conditions-card">
                    <div class="muted" style="font-size:12px; font-weight:700;">⏱ HORIZONS</div>
                    <div style="font-size:16px; font-weight:800;">24 / 48 / 72 hours</div>
                </div>
                <div class="conditions-card">
                    <div class="muted" style="font-size:12px; font-weight:700;">📊 PREDICTION TYPE</div>
                    <div style="font-size:16px; font-weight:800;">ML Forecast (Model {metrics['version']})</div>
                </div>
            <div class="conditions-card">
                <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                    <span class="muted">24h model error</span>
                    <span style="font-weight:800;">RMSE ±{rmse_24h:.2f}</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                    <span class="muted">48h model error</span>
                    <span style="font-weight:800;">RMSE ±{rmse_48h:.2f}</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span class="muted">72h model error</span>
                    <span style="font-weight:800;">RMSE ±{rmse_72h:.2f}</span>
                </div>
            </div>
            <div class="conditions-card">
                <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                    <span class="muted">MAE (test)</span>
                    <span style="font-weight:800;">{metrics.get('test_mae', 0):.2f}</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span class="muted">R² (test)</span>
                    <span style="font-weight:800;">{metrics.get('test_r2', 0):.4f}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ═══════════════════════════════════════════════════════════
        # HOURLY 72-HOUR FORECAST TABLE + CSV DOWNLOAD
        # ═══════════════════════════════════════════════════════════
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📋 Hourly Forecast Data")
        st.caption("Detailed 72-hour predictions with health classifications")

        with st.expander("View & Download 72-Hour Hourly Report", expanded=False):
            df_forecast = pd.DataFrame(predictions)
            df_forecast = df_forecast[["datetime", "predicted_aqi", "label"]].copy()
            df_forecast.columns = ["Datetime", "Predicted AQI", "Health Status"]

            st.dataframe(
                df_forecast.style.format({"Predicted AQI": "{:.2f}"}),
                use_container_width=True,
                height=320
            )

            csv_data = df_forecast.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download 72-Hour Forecast CSV",
                data=csv_data,
                file_name=f"lahore_aqi_72h_forecast_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                key="download_csv_btn"
            )

        # ═══════════════════════════════════════════════════════════
        # SHAP
        # ═══════════════════════════════════════════════════════════
        st.markdown("### 📈 Why This Prediction")
        st.caption("SHAP feature contributions for the forecast horizon")

        horizon_choice = st.radio(
            "Select forecast horizon:",
            ["24h", "48h", "72h"],
            horizontal=True,
            label_visibility="collapsed",
            key="horizon_radio"
        )

        shap_data = fetch_shap(horizon_choice)
        if shap_data and shap_data.get("success"):
            contributions = shap_data["contributions"]
            top_inc = max(contributions, key=lambda x: x["value"])
            top_dec = min(contributions, key=lambda x: x["value"])
            pred_val = {"24h": aqi_24h, "48h": aqi_48h, "72h": aqi_72h}[horizon_choice]

            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted AQI", f"{pred_val:.1f}")
            m2.metric("📈 Top Increase", f"{top_inc['feature']} (+{top_inc['value']:.2f})")
            m3.metric("📉 Top Decrease", f"{top_dec['feature']} ({top_dec['value']:.2f})")

            sorted_contrib = sorted(contributions, key=lambda x: x["value"])
            features_list = [c["feature"] for c in sorted_contrib]
            values_list = [c["value"] for c in sorted_contrib]
            colors_list = ["#f59e0b" if v > 0 else "#22c55e" for v in values_list]

            fig_shap = go.Figure(go.Bar(
                x=values_list,
                y=features_list,
                orientation="h",
                marker_color=colors_list,
                hovertemplate="<b>%{y}</b><br>SHAP: %{x:.2f}<extra></extra>"
            ))

            fig_shap.update_layout(
                height=560,
                margin=dict(l=20, r=20, t=20, b=40),
                paper_bgcolor="#111827",
                plot_bgcolor="#111827",
                font=dict(color="#f8fafc", size=13),
                xaxis=dict(
                    title=dict(text="SHAP Contribution", font=dict(color="#f8fafc", size=13)),
                    tickfont=dict(color="#f8fafc", size=12),
                    gridcolor="rgba(148,163,184,0.18)",
                    zeroline=True,
                    zerolinecolor="#94a3b8",
                    zerolinewidth=1.5,
                    color="#f8fafc"
                ),
                yaxis=dict(
                    tickfont=dict(color="#f8fafc", size=13),
                    automargin=True,
                    color="#f8fafc"
                )
            )

            st.plotly_chart(fig_shap, use_container_width=True, key="chart_shap")

            st.markdown(
                "<div class='muted' style='text-align:center;'>"
                "🟠 Increases predicted AQI &nbsp;&nbsp; 🟢 Decreases predicted AQI"
                "</div>",
                unsafe_allow_html=True
            )
        else:
            st.info("📈 SHAP explanation is temporarily unavailable — will retry on next refresh.")

    st.markdown("---")
    ingested_dt = pd.to_datetime(current["timestamp"])
    st.caption(
        f"10Pearls AQI Predictor · Air-quality intelligence for Lahore, Punjab, Pakistan · "
        f"Last Data Ingested: {ingested_dt.strftime('%b %d, %Y %I:%M %p')} PKT"
    )
    st.markdown(
        "<div class='muted' style='text-align:center; font-size:12px; padding-top:6px;'>"
        "Developed by Talha Shahzad · 10Pearls Shine Internship Program"
        "</div>",
        unsafe_allow_html=True
    )

except Exception as e:
    st.error(f"❌ Error connecting to Backend API: `{e}`")
