# ============================================================
# NEXA | STOCK INTELLIGENCE DASHBOARD
# Clean Professional Streamlit UI
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.eda import load_data
from src.live_data import download_live_stock_data

from src.feature_engineering import (
    create_moving_averages,
    create_daily_returns,
    create_lag_features,
    create_rsi,
    create_macd,
    create_bollinger_bands,
    create_date_features,
)

from src.predict import (
    load_saved_model,
    load_saved_scaler,
    load_feature_columns,
    predict_next_close,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NEXA | Stock Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL LIGHT UI
# ============================================================

st.markdown(
    """
<style>
/* ---------- Global ---------- */

.stApp {
    background: #f5f7fb;
    color: #172033;
}

.main .block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

[data-testid="stHeader"] {
    background: rgba(245, 247, 251, 0.92);
}

/* ---------- Sidebar ---------- */

section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e6eaf0;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.2rem;
}

.sidebar-brand {
    background: linear-gradient(135deg, #0f2747, #173f70);
    border-radius: 16px;
    padding: 18px 18px 16px 18px;
    margin-bottom: 22px;
    box-shadow: 0 8px 24px rgba(15, 39, 71, 0.12);
}

.sidebar-brand-title {
    color: white;
    font-size: 25px;
    font-weight: 800;
    letter-spacing: 1px;
}

.sidebar-brand-subtitle {
    color: #b9cce3;
    font-size: 11px;
    margin-top: 4px;
    letter-spacing: 0.8px;
}

/* ---------- Header ---------- */

.top-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    margin-bottom: 18px;
}

.top-title {
    font-size: 34px;
    line-height: 1.1;
    font-weight: 800;
    color: #14213d;
    letter-spacing: -1px;
}

.top-title span {
    color: #1769aa;
}

.top-subtitle {
    margin-top: 6px;
    color: #68758a;
    font-size: 14px;
}

.header-badge {
    background: #ffffff;
    border: 1px solid #dce4ee;
    border-radius: 999px;
    padding: 8px 13px;
    color: #31506f;
    font-size: 12px;
    font-weight: 700;
    white-space: nowrap;
}

/* ---------- Status ---------- */

.data-status {
    background: #ffffff;
    border: 1px solid #e1e7ef;
    border-left: 4px solid #2f80ed;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 18px;
    color: #42526b;
    font-size: 13px;
}

/* ---------- Hero ---------- */

.hero {
    background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
    border: 1px solid #dfe7f0;
    border-radius: 20px;
    padding: 24px 28px;
    box-shadow: 0 10px 30px rgba(31, 52, 73, 0.07);
    margin-bottom: 18px;
}

.hero-label {
    color: #738198;
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 1.1px;
    font-weight: 800;
}

.hero-grid {
    display: grid;
    grid-template-columns: 1.2fr 1fr 1fr 0.9fr;
    gap: 18px;
    align-items: center;
    margin-top: 8px;
}

.hero-main-value {
    color: #132238;
    font-size: 39px;
    font-weight: 800;
    line-height: 1.1;
}

.hero-caption {
    color: #7a879a;
    font-size: 12px;
    margin-top: 5px;
}

.hero-number {
    color: #172033;
    font-size: 22px;
    font-weight: 800;
}

.hero-small {
    color: #7a879a;
    font-size: 11px;
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.7px;
}

.signal-up {
    color: #15803d;
    background: #ecfdf3;
    border: 1px solid #bbf7d0;
    border-radius: 10px;
    padding: 9px 13px;
    display: inline-block;
    font-weight: 800;
}

.signal-down {
    color: #b42318;
    background: #fff1f2;
    border: 1px solid #fecdd3;
    border-radius: 10px;
    padding: 9px 13px;
    display: inline-block;
    font-weight: 800;
}

.signal-neutral {
    color: #475569;
    background: #f1f5f9;
    border: 1px solid #dbe3ec;
    border-radius: 10px;
    padding: 9px 13px;
    display: inline-block;
    font-weight: 800;
}

/* ---------- Metric cards ---------- */

div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 16px 18px;
    box-shadow: 0 5px 18px rgba(31, 52, 73, 0.045);
}

div[data-testid="stMetricLabel"] {
    color: #748196 !important;
    font-size: 12px !important;
}

div[data-testid="stMetricValue"] {
    color: #172033 !important;
    font-weight: 800 !important;
}

/* ---------- Sections ---------- */

.section-title {
    color: #172033;
    font-size: 21px;
    font-weight: 800;
    margin-top: 24px;
    margin-bottom: 3px;
}

.section-subtitle {
    color: #7b8799;
    font-size: 12px;
    margin-bottom: 12px;
}

/* ---------- Cards ---------- */

.info-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 17px;
    min-height: 125px;
    box-shadow: 0 5px 18px rgba(31, 52, 73, 0.04);
}

.info-label {
    color: #7a8799;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 800;
}

.info-value {
    color: #172033;
    font-size: 21px;
    font-weight: 800;
    margin-top: 7px;
}

.info-text {
    color: #7a8799;
    font-size: 11px;
    margin-top: 6px;
    line-height: 1.4;
}

/* ---------- Tabs ---------- */

button[data-baseweb="tab"] {
    color: #65748a;
    font-weight: 700;
    font-size: 13px;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #1769aa;
}

div[data-baseweb="tab-highlight"] {
    background: #1769aa;
}

/* ---------- Dataframe ---------- */

div[data-testid="stDataFrame"] {
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    overflow: hidden;
}

/* ---------- Buttons ---------- */

.stButton > button {
    border: 1px solid #d8e1eb;
    background: #ffffff;
    color: #31506f;
    border-radius: 9px;
    font-weight: 700;
}

.stButton > button:hover {
    border-color: #1769aa;
    color: #1769aa;
}

/* ---------- Footer ---------- */

.footer {
    border-top: 1px solid #e3e8ef;
    margin-top: 40px;
    padding-top: 20px;
    text-align: center;
    color: #8a95a6;
    font-size: 11px;
    line-height: 1.7;
}

/* ---------- Responsive ---------- */

@media (max-width: 900px) {
    .hero-grid {
        grid-template-columns: 1fr 1fr;
    }

    .top-header {
        align-items: flex-start;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
<div class="sidebar-brand">
<div class="sidebar-brand-title">NEXA</div>
<div class="sidebar-brand-subtitle">STOCK INTELLIGENCE</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("### Dashboard")

    data_source = st.radio(
        "Data source",
        ["Historical CSV", "Live Yahoo Finance"],
    )

    selected_stock = st.selectbox(
        "Stock",
        ["AAPL"],
    )

    if data_source == "Live Yahoo Finance":

        period = st.selectbox(
            "Market history",
            ["1y", "2y", "5y"],
            index=1,
        )

        if st.button(
            "↻ Refresh data",
            use_container_width=True,
        ):
            st.cache_data.clear()
            st.rerun()

    else:
        period = "2y"

    st.divider()

    st.markdown("### Chart")

    chart_days = st.slider(
        "Visible trading days",
        30,
        500,
        250,
        10,
    )

    st.markdown("### Technical indicators")

    show_ma = st.checkbox(
        "Moving averages",
        True,
    )

    show_bollinger = st.checkbox(
        "Bollinger Bands",
        True,
    )

    show_volume = st.checkbox(
        "Volume",
        True,
    )

    show_rsi = st.checkbox(
        "RSI",
        True,
    )

    show_macd = st.checkbox(
        "MACD",
        True,
    )

    st.markdown("### ML analysis")

    show_actual_predicted = st.checkbox(
        "Historical model prediction",
        True,
    )

    show_feature_impact = st.checkbox(
        "Feature coefficients",
        True,
    )

    st.divider()

    st.caption(
        "Educational stock analytics dashboard"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    f"""
<div class="top-header">
<div>
<div class="top-title"><span>NEXA</span> Market Intelligence</div>
<div class="top-subtitle">
Machine-learning powered stock analysis and next-day price estimation
</div>
</div>
<div class="header-badge">● {selected_stock} ANALYSIS</div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def get_csv_data():
    return load_data("data/raw/AAPL.csv")


@st.cache_data
def get_live_data(ticker, selected_period):
    return download_live_stock_data(
        ticker=ticker,
        period=selected_period,
        interval="1d",
    )


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_data(df, is_live=False):

    df = df.copy()

    if not is_live and "Price" in df.columns:
        df["Date"] = pd.to_datetime(
            df["Price"],
            errors="coerce",
        )

    elif is_live and "Date" in df.columns:
        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce",
        )

    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    for column in [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df.dropna(
        subset=required_columns
    )

    df = df.sort_values("Date")
    df = df.reset_index(drop=True)

    df = create_moving_averages(df)
    df = create_daily_returns(df)
    df = create_lag_features(df)
    df = create_rsi(df)
    df = create_macd(df)
    df = create_bollinger_bands(df)
    df = create_date_features(df)

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return df


# ============================================================
# LOAD DATA
# ============================================================

try:

    if data_source == "Historical CSV":

        raw_df = get_csv_data()

        df = prepare_data(
            raw_df,
            is_live=False,
        )

    else:

        raw_df = get_live_data(
            selected_stock,
            period,
        )

        df = prepare_data(
            raw_df,
            is_live=True,
        )

except Exception as error:

    st.error("Unable to load market data.")
    st.exception(error)
    st.stop()


latest_date = df["Date"].max()

if data_source == "Live Yahoo Finance":

    st.markdown(
        f"""
<div class="data-status">
🟢 Latest available Yahoo Finance market data loaded for
<strong>{selected_stock}</strong>
</div>
""",
        unsafe_allow_html=True,
    )

else:

    st.markdown(
        """
<div class="data-status">
📁 Historical CSV dataset loaded successfully.
</div>
""",
        unsafe_allow_html=True,
    )

st.caption(
    f"Latest available trading date: "
    f"{latest_date.strftime('%d %b %Y')}"
)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def get_model():

    model = load_saved_model()
    scaler = load_saved_scaler()
    feature_columns = load_feature_columns()

    return model, scaler, feature_columns


try:

    model, scaler, feature_columns = get_model()

except Exception as error:

    st.error("Unable to load the saved ML model.")
    st.exception(error)
    st.stop()


# ============================================================
# VALIDATE FEATURES
# ============================================================

missing_features = [
    feature
    for feature in feature_columns
    if feature not in df.columns
]

if missing_features:

    st.error(
        "The current dataset is missing features required by the saved model."
    )

    st.write(missing_features)
    st.stop()


# ============================================================
# PREDICTION
# ============================================================

prediction_df = df.dropna(
    subset=feature_columns
).copy()

if prediction_df.empty:

    st.error(
        "There is not enough valid data for prediction."
    )

    st.stop()


current_price = float(
    prediction_df["Close"].iloc[-1]
)

try:

    predicted_price = float(
        predict_next_close(
            prediction_df,
            model,
            scaler,
            feature_columns,
        )
    )

except Exception as error:

    st.error("Prediction failed.")
    st.exception(error)
    st.stop()


price_change = predicted_price - current_price

percentage_change = (
    price_change / current_price
) * 100


if price_change > 0:

    direction = "BULLISH"
    signal_class = "signal-up"
    signal_icon = "▲"

elif price_change < 0:

    direction = "BEARISH"
    signal_class = "signal-down"
    signal_icon = "▼"

else:

    direction = "NEUTRAL"
    signal_class = "signal-neutral"
    signal_icon = "—"


# ============================================================
# HERO
# ============================================================

st.markdown(
    f"""
<div class="hero">
<div class="hero-label">NEXT-DAY MODEL OUTLOOK • {selected_stock}</div>
<div class="hero-grid">

<div>
<div class="hero-main-value">${predicted_price:,.2f}</div>
<div class="hero-caption">Estimated next closing price</div>
</div>

<div>
<div class="hero-small">Current close</div>
<div class="hero-number">${current_price:,.2f}</div>
</div>

<div>
<div class="hero-small">Expected movement</div>
<div class="hero-number">{price_change:+.2f} ({percentage_change:+.2f}%)</div>
</div>

<div>
<div class="hero-small">Model signal</div>
<div class="{signal_class}">{signal_icon} {direction}</div>
</div>

</div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# KPI ROW
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Current Close",
        f"${current_price:,.2f}",
    )

with c2:
    st.metric(
        "Predicted Next Close",
        f"${predicted_price:,.2f}",
    )

with c3:
    st.metric(
        "Expected Change",
        f"{percentage_change:+.2f}%",
        delta=f"${price_change:+.2f}",
    )

with c4:
    st.metric(
        "Model",
        "Linear Regression",
    )


# ============================================================
# TABS
# ============================================================

tab_overview, tab_technical, tab_ml, tab_market = st.tabs(
    [
        "Overview",
        "Technical Analysis",
        "ML Intelligence",
        "Market Data",
    ]
)


# ============================================================
# PLOTLY THEME
# ============================================================

def apply_chart_layout(fig, height=500):

    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        margin=dict(
            l=15,
            r=15,
            t=45,
            b=20,
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linecolor="#e2e8f0",
    )

    fig.update_yaxes(
        gridcolor="#edf1f5",
        zeroline=False,
    )

    return fig


# ============================================================
# OVERVIEW
# ============================================================

with tab_overview:

    st.markdown(
        '<div class="section-title">Price Overview</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Candlestick price action with selected trend indicators'
        '</div>',
        unsafe_allow_html=True,
    )

    chart_data = (
        df.tail(chart_days)
        .dropna(
            subset=[
                "Open",
                "High",
                "Low",
                "Close",
            ]
        )
        .copy()
    )

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=chart_data["Date"],
            open=chart_data["Open"],
            high=chart_data["High"],
            low=chart_data["Low"],
            close=chart_data["Close"],
            name="AAPL",
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626",
        )
    )

    if show_ma:

        if "MA_20" in chart_data.columns:

            fig.add_trace(
                go.Scatter(
                    x=chart_data["Date"],
                    y=chart_data["MA_20"],
                    mode="lines",
                    name="MA 20",
                    line=dict(
                        color="#2563eb",
                        width=2,
                    ),
                )
            )

        if "MA_50" in chart_data.columns:

            fig.add_trace(
                go.Scatter(
                    x=chart_data["Date"],
                    y=chart_data["MA_50"],
                    mode="lines",
                    name="MA 50",
                    line=dict(
                        color="#f59e0b",
                        width=2,
                    ),
                )
            )

    if show_bollinger:

        upper_candidates = [
            "Upper_Band",
            "BB_Upper",
            "Upper_Bollinger_Band",
        ]

        lower_candidates = [
            "Lower_Band",
            "BB_Lower",
            "Lower_Bollinger_Band",
        ]

        upper_column = next(
            (
                column
                for column in upper_candidates
                if column in chart_data.columns
            ),
            None,
        )

        lower_column = next(
            (
                column
                for column in lower_candidates
                if column in chart_data.columns
            ),
            None,
        )

        if upper_column:

            fig.add_trace(
                go.Scatter(
                    x=chart_data["Date"],
                    y=chart_data[upper_column],
                    mode="lines",
                    name="Upper Band",
                    line=dict(
                        color="#94a3b8",
                        width=1,
                        dash="dot",
                    ),
                )
            )

        if lower_column:

            fig.add_trace(
                go.Scatter(
                    x=chart_data["Date"],
                    y=chart_data[lower_column],
                    mode="lines",
                    name="Lower Band",
                    line=dict(
                        color="#94a3b8",
                        width=1,
                        dash="dot",
                    ),
                )
            )

    fig.update_layout(
        xaxis_rangeslider_visible=False,
    )

    fig = apply_chart_layout(
        fig,
        570,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    if show_volume:

        st.markdown(
            '<div class="section-title">Trading Volume</div>',
            unsafe_allow_html=True,
        )

        volume_data = df.tail(chart_days)

        volume_fig = go.Figure()

        volume_fig.add_trace(
            go.Bar(
                x=volume_data["Date"],
                y=volume_data["Volume"],
                name="Volume",
                marker_color="#94a3b8",
            )
        )

        volume_fig = apply_chart_layout(
            volume_fig,
            280,
        )

        st.plotly_chart(
            volume_fig,
            use_container_width=True,
        )


# ============================================================
# TECHNICAL ANALYSIS
# ============================================================

with tab_technical:

    st.markdown(
        '<div class="section-title">Technical Analysis</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Momentum and trend indicators calculated from the available market history'
        '</div>',
        unsafe_allow_html=True,
    )

    latest_row = df.iloc[-1]

    rsi_value = np.nan

    if "RSI" in df.columns:

        series = df["RSI"].dropna()

        if not series.empty:
            rsi_value = float(series.iloc[-1])

    macd_value = np.nan

    if "MACD" in df.columns:

        series = df["MACD"].dropna()

        if not series.empty:
            macd_value = float(series.iloc[-1])

    ma20_value = np.nan

    if "MA_20" in df.columns:

        series = df["MA_20"].dropna()

        if not series.empty:
            ma20_value = float(series.iloc[-1])

    ma50_value = np.nan

    if "MA_50" in df.columns:

        series = df["MA_50"].dropna()

        if not series.empty:
            ma50_value = float(series.iloc[-1])

    tech_cols = st.columns(4)

    with tech_cols[0]:

        value = (
            f"{rsi_value:.2f}"
            if not np.isnan(rsi_value)
            else "N/A"
        )

        st.markdown(
            f"""
<div class="info-card">
<div class="info-label">RSI</div>
<div class="info-value">{value}</div>
<div class="info-text">Momentum indicator on a 0–100 scale.</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with tech_cols[1]:

        value = (
            f"{macd_value:.4f}"
            if not np.isnan(macd_value)
            else "N/A"
        )

        st.markdown(
            f"""
<div class="info-card">
<div class="info-label">MACD</div>
<div class="info-value">{value}</div>
<div class="info-text">Trend and momentum measurement.</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with tech_cols[2]:

        value = (
            f"${ma20_value:,.2f}"
            if not np.isnan(ma20_value)
            else "N/A"
        )

        st.markdown(
            f"""
<div class="info-card">
<div class="info-label">MA 20</div>
<div class="info-value">{value}</div>
<div class="info-text">Short-term trend reference.</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with tech_cols[3]:

        value = (
            f"${ma50_value:,.2f}"
            if not np.isnan(ma50_value)
            else "N/A"
        )

        st.markdown(
            f"""
<div class="info-card">
<div class="info-label">MA 50</div>
<div class="info-value">{value}</div>
<div class="info-text">Medium-term trend reference.</div>
</div>
""",
            unsafe_allow_html=True,
        )

    if show_rsi and "RSI" in df.columns:

        st.markdown(
            '<div class="section-title">RSI Momentum</div>',
            unsafe_allow_html=True,
        )

        rsi_data = (
            df.tail(chart_days)
            .dropna(subset=["RSI"])
        )

        rsi_fig = go.Figure()

        rsi_fig.add_trace(
            go.Scatter(
                x=rsi_data["Date"],
                y=rsi_data["RSI"],
                mode="lines",
                name="RSI",
                line=dict(
                    color="#2563eb",
                    width=2,
                ),
            )
        )

        rsi_fig.add_hline(
            y=70,
            line_dash="dash",
            line_color="#dc2626",
        )

        rsi_fig.add_hline(
            y=30,
            line_dash="dash",
            line_color="#16a34a",
        )

        rsi_fig.update_yaxes(
            range=[0, 100]
        )

        rsi_fig = apply_chart_layout(
            rsi_fig,
            330,
        )

        st.plotly_chart(
            rsi_fig,
            use_container_width=True,
        )

    if show_macd:

        macd_columns = [
            column
            for column in [
                "MACD",
                "MACD_Line",
                "MACD_Signal",
                "Signal_Line",
            ]
            if column in df.columns
        ]

        if macd_columns:

            st.markdown(
                '<div class="section-title">MACD Trend</div>',
                unsafe_allow_html=True,
            )

            macd_data = (
                df.tail(chart_days)
                .dropna(
                    subset=macd_columns,
                    how="all",
                )
            )

            macd_fig = go.Figure()

            for column in macd_columns:

                macd_fig.add_trace(
                    go.Scatter(
                        x=macd_data["Date"],
                        y=macd_data[column],
                        mode="lines",
                        name=column,
                    )
                )

            macd_fig = apply_chart_layout(
                macd_fig,
                330,
            )

            st.plotly_chart(
                macd_fig,
                use_container_width=True,
            )


# ============================================================
# ML INTELLIGENCE
# ============================================================

with tab_ml:

    st.markdown(
        '<div class="section-title">ML Intelligence</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Model architecture, historical predictions and feature coefficients'
        '</div>',
        unsafe_allow_html=True,
    )

    model_cols = st.columns(3)

    with model_cols[0]:

        st.markdown(
            """
<div class="info-card">
<div class="info-label">Final model</div>
<div class="info-value">Linear Regression</div>
<div class="info-text">
Saved model used for next-day closing-price estimation.
</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with model_cols[1]:

        st.markdown(
            """
<div class="info-card">
<div class="info-label">Problem type</div>
<div class="info-value">Regression</div>
<div class="info-text">
Predicts a continuous numerical closing-price value.
</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with model_cols[2]:

        st.markdown(
            """
<div class="info-card">
<div class="info-label">Target</div>
<div class="info-value">Next-Day Close</div>
<div class="info-text">
Estimated closing price for the next trading session.
</div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">Model Pipeline</div>',
        unsafe_allow_html=True,
    )

    st.code(
        """
Market Data
    ↓
Data Cleaning
    ↓
Feature Engineering
    ↓
Technical Indicators
    ↓
Feature Scaling
    ↓
Linear Regression
    ↓
Next-Day Prediction
    ↓
Dashboard
""",
        language="text",
    )

    if show_actual_predicted:

        st.markdown(
            '<div class="section-title">'
            'Historical Actual vs Model Prediction'
            '</div>',
            unsafe_allow_html=True,
        )

        st.caption(
            "This view compares model outputs with historical next-close values. "
            "It is an analytical comparison, not a separate held-out test."
        )

        backtest = df.dropna(
            subset=feature_columns
        ).copy()

        if not backtest.empty:

            try:

                X = backtest[feature_columns]

                X_scaled = scaler.transform(X)

                predictions = model.predict(
                    X_scaled
                )

                backtest["Model_Prediction"] = predictions

                backtest["Actual_Next_Close"] = (
                    backtest["Close"].shift(-1)
                )

                backtest = backtest.dropna(
                    subset=[
                        "Actual_Next_Close",
                        "Model_Prediction",
                    ]
                )

                backtest = backtest.tail(
                    chart_days
                )

                comparison_fig = go.Figure()

                comparison_fig.add_trace(
                    go.Scatter(
                        x=backtest["Date"],
                        y=backtest["Actual_Next_Close"],
                        mode="lines",
                        name="Actual",
                        line=dict(
                            color="#1769aa",
                            width=2,
                        ),
                    )
                )

                comparison_fig.add_trace(
                    go.Scatter(
                        x=backtest["Date"],
                        y=backtest["Model_Prediction"],
                        mode="lines",
                        name="Model",
                        line=dict(
                            color="#f59e0b",
                            width=2,
                            dash="dot",
                        ),
                    )
                )

                comparison_fig = apply_chart_layout(
                    comparison_fig,
                    430,
                )

                st.plotly_chart(
                    comparison_fig,
                    use_container_width=True,
                )

            except Exception as error:

                st.warning(
                    "Historical model comparison could not be generated."
                )

                st.caption(str(error))

    if show_feature_impact:

        st.markdown(
            '<div class="section-title">'
            'Feature Coefficients'
            '</div>',
            unsafe_allow_html=True,
        )

        if hasattr(model, "coef_"):

            coefficients = np.asarray(
                model.coef_
            ).flatten()

            if len(coefficients) == len(
                feature_columns
            ):

                coefficient_df = pd.DataFrame(
                    {
                        "Feature": feature_columns,
                        "Coefficient": coefficients,
                    }
                )

                coefficient_df[
                    "Absolute Impact"
                ] = coefficient_df[
                    "Coefficient"
                ].abs()

                coefficient_df = (
                    coefficient_df
                    .sort_values(
                        "Absolute Impact",
                        ascending=False,
                    )
                    .head(15)
                    .sort_values("Coefficient")
                )

                coefficient_fig = go.Figure()

                coefficient_fig.add_trace(
                    go.Bar(
                        x=coefficient_df["Coefficient"],
                        y=coefficient_df["Feature"],
                        orientation="h",
                        marker_color="#1769aa",
                    )
                )

                coefficient_fig = apply_chart_layout(
                    coefficient_fig,
                    500,
                )

                st.plotly_chart(
                    coefficient_fig,
                    use_container_width=True,
                )

                st.caption(
                    "These are coefficients of the scaled model features. "
                    "They describe direction and relative magnitude in the "
                    "linear model; they are not causal effects."
                )

            else:

                st.info(
                    "The coefficient count does not match the saved feature list."
                )

        else:

            st.info(
                "The saved model does not expose linear coefficients."
            )


# ============================================================
# MARKET DATA
# ============================================================

with tab_market:

    st.markdown(
        '<div class="section-title">Market Data</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Latest observations and model feature information'
        '</div>',
        unsafe_allow_html=True,
    )

    d1, d2, d3, d4 = st.columns(4)

    with d1:
        st.metric(
            "Rows",
            f"{len(df):,}",
        )

    with d2:
        st.metric(
            "Columns",
            f"{len(df.columns):,}",
        )

    with d3:
        st.metric(
            "Start",
            df["Date"].min().strftime("%d %b %Y"),
        )

    with d4:
        st.metric(
            "End",
            df["Date"].max().strftime("%d %b %Y"),
        )

    st.markdown(
        '<div class="section-title">Latest observations</div>',
        unsafe_allow_html=True,
    )

    display_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in df.columns
    ]

    latest_data = (
        df[available_columns]
        .tail(10)
        .copy()
    )

    if "Date" in latest_data.columns:

        latest_data["Date"] = (
            latest_data["Date"]
            .dt.strftime("%d %b %Y")
        )

    st.dataframe(
        latest_data,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        '<div class="section-title">Saved model features</div>',
        unsafe_allow_html=True,
    )

    feature_df = pd.DataFrame(
        {
            "Feature": feature_columns
        }
    )

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True,
        height=300,
    )


# ============================================================
# SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">Model Summary</div>',
    unsafe_allow_html=True,
)

if price_change > 0:

    st.success(
        f"Model estimate: ${predicted_price:,.2f}, "
        f"representing an estimated change of "
        f"{percentage_change:+.2f}% from the current close."
    )

elif price_change < 0:

    st.warning(
        f"Model estimate: ${predicted_price:,.2f}, "
        f"representing an estimated change of "
        f"{percentage_change:+.2f}% from the current close."
    )

else:

    st.info(
        "The model estimates little change from the current close."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">
<strong>NEXA | Stock Intelligence</strong><br>
Machine-learning dashboard for educational and analytical purposes.<br>
Predictions are model estimates based on historical market data and engineered features,
not financial advice.
</div>
""",
    unsafe_allow_html=True,
)
