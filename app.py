# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
from datetime import datetime
import calendar

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide", page_title="Interactive Weather Dashboard 🌍")

# ---------------------------
# Helpers & Utilities
# ---------------------------

# Helper to convert month number to abbreviation
def calendar_month(m):
    return calendar.month_abbr[m]

@st.cache_data
def load_data(path="weather_filled.csv"):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # parse datetime
    if "last_updated" in df.columns:
        df["last_updated_dt"] = pd.to_datetime(df["last_updated"], errors="coerce")
    elif "last_updated_epoch" in df.columns:
        df["last_updated_dt"] = pd.to_datetime(df["last_updated_epoch"], unit="s", errors="coerce")
    else:
        df["last_updated_dt"] = pd.NaT

    # create date column
    df["date"] = pd.to_datetime(df["last_updated_dt"]).dt.date

    # coerce numeric columns
    numeric_cols = [
        "temperature_celsius", "temperature_fahrenheit", "wind_mph", "wind_kph",
        "pressure_mb", "precip_mm", "precip_in", "humidity", "cloud",
        "feels_like_celsius", "visibility_km", "uv_index", "gust_mph", "gust_kph",
        "air_quality_Carbon_Monoxide", "air_quality_Ozone",
        "air_quality_Nitrogen_dioxide", "air_quality_Sulphur_dioxide",
        "air_quality_PM2.5", "air_quality_PM10",
        "air_quality_us-epa-index", "air_quality_gb-defra-index", "latitude", "longitude"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # compute minutes since midnight for sunrise/sunset
    for col in ["sunrise", "sunset"]:
        if col in df.columns:
            times = pd.to_datetime(df[col], errors="coerce")
            df[f"{col}_min"] = times.dt.hour * 60 + times.dt.minute
    return df

def hhmm_from_minutes(m):
    if pd.isna(m):
        return None
    try:
        m = int(round(m))
        h = m // 60
        mm = m % 60
        return f"{h:02d}:{mm:02d}"
    except Exception:
        return None

def daily_aggregate(df, var):
    ser = df[[ "last_updated_dt", var ]].dropna()
    if ser.empty:
        return pd.Series(dtype="float64")
    ser = ser.set_index("last_updated_dt")
    daily = ser.resample("D").mean()
    return daily[var]

def seasonal_filter(df, season):
    seasons = {
        "All": list(range(1,13)),
        "Monsoon": [6,7,8,9],
        "Summer": [3,4,5],
        "Winter": [11,12,1,2],
        "Autumn": [9,10,11],
        "Spring": [2,3,4]
    }
    months = seasons.get(season, list(range(1,13)))
    if "last_updated_dt" not in df.columns:
        return df
    return df[df["last_updated_dt"].dt.month.isin(months)]

def safe_size_series(s):
    s = s.fillna(0).abs()
    maxv = s.max() if s.max() > 0 else 1.0
    scaled = (s / maxv) * 30 + 3
    return scaled

# ---------------------------
# Load Data
# ---------------------------
df = load_data("weather_filled.csv")

# ---------------------------
# Sidebar Filters & Navigation
# ---------------------------
st.sidebar.title("🌦 Filters & Navigation")

min_dt = df["last_updated_dt"].min()
max_dt = df["last_updated_dt"].max()
date_range = st.sidebar.date_input("Date range", [min_dt.date(), max_dt.date()] if not pd.isna(min_dt) else [])

country_list = ["All"]
if "country" in df.columns:
    country_list += sorted(df["country"].dropna().unique().tolist())
country_sel = st.sidebar.selectbox("Country", country_list)

location_list = ["All"]
if "location_name" in df.columns:
    location_list += sorted(df["location_name"].dropna().unique().tolist())
location_sel = st.sidebar.selectbox("Location (optional)", location_list)

page = st.sidebar.radio("Go to", [
    "Overview", "Temperature", "Rainfall & Humidity", "Precipitation", "Air Quality", "Maps", "Forecast", "Comparisons"
])

# apply filters
df_filt = df.copy()
if date_range and len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
    df_filt = df_filt[(df_filt["last_updated_dt"] >= start) & (df_filt["last_updated_dt"] < end)]
if country_sel != "All":
    df_filt = df_filt[df_filt["country"] == country_sel]
if location_sel != "All":
    df_filt = df_filt[df_filt["location_name"] == location_sel]

# ---------------------------
# Overview Page
# ---------------------------
if page == "Overview":
    st.title("📊 Dataset Overview")
    left, mid, right = st.columns(3)
    if "temperature_celsius" in df_filt.columns:
        left.metric("Avg Temp (°C)", f"{df_filt['temperature_celsius'].mean():.2f}")
        mid.metric("Max Temp (°C)", f"{df_filt['temperature_celsius'].max():.2f}")
        right.metric("Min Temp (°C)", f"{df_filt['temperature_celsius'].min():.2f}")
    st.markdown("#### Sample rows")
    st.dataframe(df_filt.head(200))

    st.markdown("#### Missing values")
    st.write(df_filt.isnull().sum())

    st.markdown("#### Correlation heatmap")
    num = df_filt.select_dtypes("number")
    if num.shape[1] > 1:
        fig = px.imshow(num.corr(), text_auto=True, color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough numeric columns for correlation.")

# ---------------------------
# Temperature Dashboard
# ---------------------------
elif page == "Temperature":
    st.title("🌡 Temperature Dashboard")
    freq = st.selectbox("View frequency:", ["Daily", "Monthly"], index=0)

    col1, col2, col3, col4 = st.columns(4)
    if "temperature_celsius" in df_filt.columns:
        temps = df_filt["temperature_celsius"].dropna()
        col1.metric("Avg Temp (°C)", f"{temps.mean():.2f}" if not temps.empty else "N/A")
        col2.metric("Max Temp (°C)", f"{temps.max():.2f}" if not temps.empty else "N/A")
        col3.metric("Min Temp (°C)", f"{temps.min():.2f}" if not temps.empty else "N/A")
        col4.metric("Std Dev (°C)", f"{temps.std():.2f}" if not temps.empty else "N/A")

    # Daily / Monthly line chart
    if freq == "Daily":
        ts = daily_aggregate(df_filt, "temperature_celsius")
        if not ts.empty:
            fig = px.line(ts.reset_index(), x="last_updated_dt", y="temperature_celsius",
                          title="Daily Avg Temperature (°C)", labels={"last_updated_dt":"Date","temperature_celsius":"°C"})
            st.plotly_chart(fig, use_container_width=True)
    else:
        if "last_updated_dt" in df_filt.columns:
            temp_month = df_filt.copy()
            temp_month["year"] = temp_month["last_updated_dt"].dt.year
            temp_month["month"] = temp_month["last_updated_dt"].dt.month
            grp = temp_month.groupby(["year","month"])["temperature_celsius"].mean().reset_index()
            grp["ym"] = pd.to_datetime(grp["year"].astype(str) + "-" + grp["month"].astype(str) + "-01")
            fig = px.line(grp, x="ym", y="temperature_celsius", title="Monthly Avg Temperature (°C)", labels={"ym":"Month"})
            st.plotly_chart(fig, use_container_width=True)

    # Heatmap: Month vs Year
    st.subheader("Heatmap: Avg Temp (Month vs Year)")
    if "last_updated_dt" in df_filt.columns and "temperature_celsius" in df_filt.columns:
        tmp = df_filt.copy()
        tmp["year"] = tmp["last_updated_dt"].dt.year
        tmp["month"] = tmp["last_updated_dt"].dt.month
        heat = tmp.groupby(["year","month"])["temperature_celsius"].mean().reset_index()
        if not heat.empty:
            pivot = heat.pivot(index="month", columns="year", values="temperature_celsius")
            pivot = pivot.reindex(index=list(range(1,13)))  # ensure Jan-Dec order
            fig = px.imshow(
                pivot,
                labels=dict(x="Year", y="Month", color="Avg Temp (°C)"),
                x=list(pivot.columns),
                y=[calendar_month(m) for m in pivot.index]
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for heatmap.")

# ---------------------------
# Rainfall & Humidity Dashboard
# ---------------------------
elif page == "Rainfall & Humidity":
    st.title("💧 Rainfall & Humidity Dashboard")
    # Season selector
    season = st.selectbox("Select Season:", ["All", "Monsoon", "Summer", "Winter", "Autumn", "Spring"], index=0)
    df_season = seasonal_filter(df_filt, season)

    # KPI cards
    c1, c2, c3 = st.columns(3)
    if "precip_mm" in df_season.columns:
        c1.metric("Total Rainfall (filtered)", f"{df_season['precip_mm'].sum(skipna=True):.2f} mm")
    else:
        c1.metric("Total Rainfall (filtered)", "N/A")
    if "humidity" in df_season.columns:
        c2.metric("Avg Humidity (filtered)", f"{df_season['humidity'].mean():.2f}%")
        c3.metric("Peak Humidity", f"{df_season['humidity'].max():.2f}%")
    else:
        c2.metric("Avg Humidity (filtered)", "N/A")
        c3.metric("Peak Humidity", "N/A")

    # Bar Chart: total rainfall by month
    st.subheader("Total Rainfall by Month")
    if "precip_mm" in df_season.columns and "last_updated_dt" in df_season.columns:
        tmp = df_season.copy()
        tmp["month"] = tmp["last_updated_dt"].dt.month
        monthly_rain = tmp.groupby("month")["precip_mm"].sum().reset_index()
        fig = px.bar(monthly_rain, x="month", y="precip_mm", labels={"month":"Month","prec_mm":"Total Rainfall (mm)"},
                     title="Total Rainfall by Month")
        fig.update_layout(xaxis=dict(tickmode="array", tickvals=list(monthly_rain["month"]),
                                     ticktext=[calendar_month(m) for m in monthly_rain["month"]]))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No precipitation or date data available for monthly rainfall.")

    # Humidity trend
    st.subheader("Humidity Trends (daily avg)")
    hum_ts = daily_aggregate(df_season, "humidity")
    if not hum_ts.empty:
        fig = px.line(hum_ts.reset_index(), x="last_updated_dt", y="humidity", title="Daily Avg Humidity")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough humidity data.")

    # Combo chart: rainfall (bars) + humidity (line)
    st.subheader("Combo: Rainfall (bars) + Humidity (line)")
    if "precip_mm" in df_season.columns and "humidity" in df_season.columns:
        tmp = df_season.copy()
        tmp["date_only"] = tmp["last_updated_dt"].dt.date
        agg = tmp.groupby("date_only").agg({"precip_mm":"sum", "humidity":"mean"}).reset_index()
        if not agg.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=agg["date_only"], y=agg["precip_mm"], name="Daily Rainfall (mm)", yaxis="y1"))
            fig.add_trace(go.Scatter(x=agg["date_only"], y=agg["humidity"], name="Daily Avg Humidity (%)", yaxis="y2", mode="lines"))
            fig.update_layout(
                yaxis=dict(title="Rainfall (mm)"),
                yaxis2=dict(title="Humidity (%)", overlaying="y", side="right"),
                title="Rainfall & Humidity"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough combined data.")
    else:
        st.info("Need both precip_mm and humidity columns for combo chart.")

# ---------------------------
# Precipitation Page (detailed)
# ---------------------------
elif page == "Precipitation":
    st.title("🌧 Precipitation Details")
    if "precip_mm" in df_filt.columns and "last_updated_dt" in df_filt.columns:
        st.subheader("Precipitation over Time")
        fig = px.bar(df_filt.sort_values("last_updated_dt"), x="last_updated_dt", y="precip_mm", hover_data=["location_name","country"])
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Rainy vs Non-rainy")
        df_filt["is_rainy"] = df_filt["precip_mm"].fillna(0) > 0
        counts = df_filt["is_rainy"].value_counts().rename(index={True:"Rainy", False:"Non-rainy"})
        fig2 = px.pie(values=counts.values, names=counts.index, title="Rainy vs Non-rainy records")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No precipitation data available.")

# ---------------------------
# Air Quality Page
# ---------------------------
elif page == "Air Quality":
    st.title("🌬 Air Quality Dashboard")
    aq_cols = [c for c in df_filt.columns if c.lower().startswith("air_quality")]
    if not aq_cols:
        st.info("No air quality columns found.")
    else:
        # KPI cards for PM2.5/PM10 if present
        c1, c2, c3 = st.columns(3)
        if "air_quality_PM2.5" in df_filt.columns:
            c1.metric("Avg PM2.5", f"{df_filt['air_quality_PM2.5'].mean():.2f}")
        if "air_quality_PM10" in df_filt.columns:
            c2.metric("Avg PM10", f"{df_filt['air_quality_PM10'].mean():.2f}")
        if "air_quality_us-epa-index" in df_filt.columns:
            c3.metric("Avg US-EPA Index", f"{df_filt['air_quality_us-epa-index'].mean():.2f}")

        # scatter matrix if multiple
        if len(aq_cols) > 1:
            fig = px.scatter_matrix(df_filt, dimensions=aq_cols, title="Air Quality Variables")
            st.plotly_chart(fig, use_container_width=True)
        # time series for PM2.5
        if "air_quality_PM2.5" in df_filt.columns:
            fig2 = px.line(df_filt.sort_values("last_updated_dt"), x="last_updated_dt", y="air_quality_PM2.5", title="PM2.5 over Time")
            st.plotly_chart(fig2, use_container_width=True)

# ---------------------------
# Maps Dashboard
# ---------------------------
elif page == "Maps":
    st.title("🗺 Maps & Geo Analysis")
    if ("latitude" in df_filt.columns) and ("longitude" in df_filt.columns):
        df_map = df_filt.dropna(subset=["latitude","longitude"]).copy()
        # pick bubble size & color
        size_choice = st.selectbox("Bubble size:", ["temperature_celsius","humidity","precip_mm","air_quality_PM2.5"])
        color_choice = st.selectbox("Bubble color:", ["condition_text","temperature_celsius","humidity","air_quality_PM2.5"])
        size_series = df_map[size_choice] if size_choice in df_map.columns else pd.Series(1, index=df_map.index)
        size_vals = safe_size_series(size_series)
        fig = px.scatter_geo(df_map, lat="latitude", lon="longitude",
                             hover_name="location_name",
                             hover_data=["country","temperature_celsius","humidity","precip_mm","air_quality_PM2.5"],
                             size=size_vals, color=color_choice,
                             projection="natural earth", title="Global Weather Points")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Choropleth selection
        st.subheader("Country-level Choropleth")
        choropleth_var = st.selectbox("Choropleth variable:",
                                      ["temperature_celsius","humidity","precip_mm","air_quality_PM2.5","air_quality_PM10","sunrise_min","sunset_min"])
        df_map_c = df_map.copy()
        if choropleth_var in ["sunrise_min","sunset_min"]:
            # compute average minutes and show HH:MM as hover text
            avg = df_map_c.groupby("country", as_index=False)[choropleth_var].mean()
            avg["hhmm"] = avg[choropleth_var].apply(hhmm_from_minutes)
            fig2 = px.choropleth(avg, locations="country", locationmode="country names",
                                 color=choropleth_var, hover_name="country",
                                 hover_data=["hhmm"], title=f"Avg {choropleth_var} by Country")
            st.plotly_chart(fig2, use_container_width=True, height=500)
        else:
            if choropleth_var in df_map_c.columns:
                avg = df_map_c.groupby("country", as_index=False)[choropleth_var].mean()
                fig2 = px.choropleth(avg, locations="country", locationmode="country names",
                                     color=choropleth_var, hover_name="country",
                                     title=f"Avg {choropleth_var} by Country")
                st.plotly_chart(fig2, use_container_width=True, height=500)
            else:
                st.info("Selected variable not available for choropleth.")

    else:
        st.info("Latitude/Longitude columns not found - maps cannot be shown.")

# ---------------------------
# Forecast Dashboard (ARIMA / SARIMA)
# ---------------------------
elif page == "Forecast":
    st.title("🔮 Forecast Dashboard (ARIMA / SARIMA)")
    # choice of variable
    var_options = [c for c in df.columns if df[c].dtype.kind in "fi"]
    if not var_options:
        st.error("No numeric variables available for forecasting.")
    else:
        var = st.selectbox("Select variable to forecast:", var_options, index=var_options.index("temperature_celsius") if "temperature_celsius" in var_options else 0)
        model_type = st.selectbox("Model type:", ["ARIMA","SARIMA"])
        horizon_choice = st.selectbox("Forecast horizon:", ["7 days","14 days","30 days","90 days"])
        horizon = int(horizon_choice.split()[0])

        run = st.button("Run Forecast")

        if run:
            # prepare timeseries: daily aggregation, drop NaT
            ser = df[[ "last_updated_dt", var ]].dropna().set_index("last_updated_dt")
            if ser.empty:
                st.error("No data for selected variable.")
            else:
                # aggregate duplicates -> daily mean
                ts_daily = ser[var].resample("D").mean()
                # simple interpolation for missing days
                ts_daily = ts_daily.interpolate(limit_direction="both")

                # build model
                try:
                    if model_type == "ARIMA":
                        model = ARIMA(ts_daily, order=(2,1,2))
                        res = model.fit()
                        forecast_res = res.get_forecast(steps=horizon)
                        pred = forecast_res.predicted_mean
                        ci = forecast_res.conf_int()
                    else:
                        # seasonal_period set to 7 (weekly) or 12 if monthly? use 7 for daily seasonality fallback
                        model = SARIMAX(ts_daily, order=(2,1,2), seasonal_order=(1,1,1,7), enforce_stationarity=False, enforce_invertibility=False)
                        res = model.fit(disp=False)
                        forecast_res = res.get_forecast(steps=horizon)
                        pred = forecast_res.predicted_mean
                        ci = forecast_res.conf_int()

                    # KPI cards for next 7 days (or horizon)
                    next_days = min(7,horizon)
                    kdf = pred.head(next_days)
                    avg_pred = kdf.mean()
                    # try to get rainfall/humidity avg predictions if present
                    col1, col2, col3 = st.columns(3)
                    col1.metric(f"Avg {var} (next {next_days} days)", f"{avg_pred:.2f}")
                    if "precip_mm" in df.columns:
                        # if precip not the var, optionally forecast precip separately? For now compute mean historical daily precip to show as rough KPI
                        col2.metric("Avg historical daily precip (filtered)", f"{df_filt['precip_mm'].mean(skipna=True):.2f} mm")
                    if "humidity" in df.columns:
                        col3.metric("Avg historical humidity (filtered)", f"{df_filt['humidity'].mean(skipna=True):.2f}%")

                    # Plot actual + predicted + conf interval
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=ts_daily.index, y=ts_daily.values, mode="lines", name="Historical"))
                    fig.add_trace(go.Scatter(x=pred.index, y=pred.values, mode="lines+markers", name="Forecast"))
                    # CI shading
                    fig.add_traces([
                        go.Scatter(
                            x=list(ci.index) + list(ci.index[::-1]),
                            y=list(ci.iloc[:,0]) + list(ci.iloc[:,1][::-1]),
                            fill='toself',
                            fillcolor='rgba(200,200,200,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            hoverinfo="skip",
                            showlegend=True,
                            name="Confidence Interval"
                        )
                    ])
                    fig.update_layout(title=f"{model_type} forecast for {var}", xaxis_title="Date", yaxis_title=var)
                    st.plotly_chart(fig, use_container_width=True)

                    # Show forecast table
                    out_df = pd.DataFrame({"date":pred.index, f"pred_{var}":pred.values})
                    out_df = out_df.reset_index(drop=True)
                    st.subheader("Forecast values")
                    st.dataframe(out_df.head(horizon))

                except Exception as e:
                    st.error(f"Forecast failed: {e}")

# ---------------------------
# Comparisons Page
# ---------------------------
elif page == "Comparisons":
    st.title("🔎 Comparisons & Scatter Matrix")
    numeric_cols = df_filt.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns available.")
    else:
        cols = st.multiselect("Choose columns to compare (2-8 recommended):", numeric_cols, default=numeric_cols[:5])
        if len(cols) >= 2:
            fig = px.scatter_matrix(df_filt, dimensions=cols, title="Scatter Matrix")
            st.plotly_chart(fig, use_container_width=True, height=800)
        else:
            st.info("Select at least 2 numeric columns to compare.")

# ---------------------------
# Footer / Tips
# ---------------------------
st.markdown("---")
st.markdown("Tips: Use the sidebar filters (date, country, location) to narrow data. Forecasts use ARIMA/SARIMA and require reasonable historic time series (daily aggregated). Sunrise/Sunset choropleth uses minutes since midnight — hover to see HH:MM where available.")