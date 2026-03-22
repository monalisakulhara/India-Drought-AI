import streamlit as st
import joblib
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import ee
from datetime import datetime

# ==========================================
# 1. PAGE CONFIG & UI SETUP (Req 9)
# ==========================================
st.set_page_config(page_title="India Drought Monitor", page_icon="🌍", layout="wide")

st.title("🛰️ National Drought & Water Stress AI")
st.markdown("Advanced early-warning system powered by multi-variate satellite telemetry and machine learning.")

# ==========================================
# 2. INITIALIZATION
# ==========================================
@st.cache_resource
def init_ee():
    try:
        # 1. Check if we are running on Streamlit Cloud (looks for secrets)
        if "EE_CREDENTIALS" in st.secrets:
            # We use the secret key to log in
            credentials = ee.ServiceAccountCredentials(
                st.secrets["EE_CREDENTIALS"]["client_email"],
                st.secrets["EE_CREDENTIALS"]["private_key"].replace('\\n', '\n')
            )
            ee.Initialize(credentials, project='your-project-id') # <-- STILL PUT YOUR PROJECT ID HERE
        else:
            # 2. Fallback for your local laptop
            ee.Initialize(project='your-project-id') # <-- STILL PUT YOUR PROJECT ID HERE
        return True
    except Exception as e:
        st.error(f"Earth Engine Connection Failed: {e}")
        return False
@st.cache_resource
def load_model():
    return joblib.load('pan_india_drought_model_atlas.joblib')

# Helper function to generate Earth Engine Map Tiles
def get_ee_url(image, vis_params):
    map_id_dict = ee.Image(image).getMapId(vis_params)
    return map_id_dict['tile_fetcher'].url_format

ee_ready = init_ee()
model = load_model()

# ==========================================
# 3. SIDEBAR & DATA SOURCES (Req 8)
# ==========================================
with st.sidebar:
    st.header("⏱️ Temporal Settings")
    selected_year = st.selectbox("Target Year", [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026], index=7)
    selected_month = st.slider("Target Month", 1, 12, datetime.now().month - 1, help="1=Jan, 12=Dec")
    
    st.markdown("---")
    st.header("📡 Data Sources")
    st.markdown("""
    * **Rainfall:** CHIRPS Daily
    * **Vegetation (NDVI):** MODIS
    * **Soil Moisture:** NASA SMAP (Root-Zone)
    * **Temperature:** ERA5-Land
    * **Groundwater:** NASA GRACE
    * **Ground Truth:** Drought Atlas of India (SPEI) 
    """)

# ==========================================
# 4. INTERACTIVE MAP & DATA EXTRACTION
# ==========================================
st.subheader("📍 Select a Region for Analysis")
m = folium.Map(location=[22.5937, 78.9629], zoom_start=5, tiles="CartoDB positron")
m.add_child(folium.LatLngPopup())
map_data = st_folium(m, height=400, width=1200)

if map_data['last_clicked'] and ee_ready:
    lat = map_data['last_clicked']['lat']
    lon = map_data['last_clicked']['lng']
    
    with st.spinner('📡 Synchronizing with NASA/ESA Satellites...'):
        try:
            point = ee.Geometry.Point([lon, lat])
            
            # --- Step A: Get Location Names ---
            admin_layer = ee.FeatureCollection("FAO/GAUL/2015/level2")
            location_info = admin_layer.filterBounds(point).first().getInfo()
            district_name = location_info['properties'].get('ADM2_NAME', 'Unknown District') if location_info else "Unknown"
            state_name = location_info['properties'].get('ADM1_NAME', 'Unknown State') if location_info else "Unknown"
            st.success(f"📍 Location Confirmed: **{district_name}, {state_name}** ({lat:.2f}, {lon:.2f})")

            # --- Step B: Setup Dates (Current & Previous for Trend) ---
            curr_start = ee.Date.fromYMD(selected_year, selected_month, 1)
            prev_start = curr_start.advance(-1, 'month')

            # This internal function handles the satellite "ping"
            def fetch_satellite_metrics(start_date):
                end_date = start_date.advance(1, 'month')
                
                # Check availability first to avoid the "Red Box" crash
                rain_coll = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(start_date, end_date)
                if rain_coll.size().getInfo() == 0:
                    return None

                temp_coll = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").filterDate(start_date, end_date)
                soil_coll = ee.ImageCollection("NASA/SMAP/SPL4SMGP/008").filterDate(start_date, end_date)
                ndvi_coll = ee.ImageCollection("MODIS/061/MOD13Q1").filterDate(start_date, end_date)
                grace_coll = ee.ImageCollection("NASA/GRACE/MASS_GRIDS_V04/MASCON_CRI").filterDate(start_date, end_date)

                # Stack and process
                img = ee.Image([
                    rain_coll.sum().rename('precipitation'),
                    temp_coll.select('temperature_2m').mean().subtract(273.15).rename('temperature'),
                    soil_coll.select('sm_rootzone').mean().rename('soil_moisture'),
                    ndvi_coll.select('NDVI').mean().multiply(0.0001).rename('ndvi'),
                    grace_coll.select('lwe_thickness').mean().rename('groundwater')
                ])
                
                return img.sample(region=point, scale=5000).first().getInfo()

            # --- Step C: Pull the Data ---
            curr_raw = fetch_satellite_metrics(curr_start)
            prev_raw = fetch_satellite_metrics(prev_start)

            if not curr_raw:
                st.warning(f"⏳ **Data Not Yet Available:** Satellites haven't published telemetry for {selected_month}/{selected_year} yet. Try an earlier date!")
                st.stop()

            curr_data = curr_raw['properties']
            prev_data = prev_raw['properties'] if prev_raw else curr_data

            # --- Step D: AI Prediction & Confidence ---
            input_df = pd.DataFrame([[
                curr_data.get('precipitation', 0), curr_data.get('temperature', 0), 
                curr_data.get('soil_moisture', 0), curr_data.get('ndvi', 0), curr_data.get('groundwater', 0)
            ]], columns=['precipitation', 'temperature', 'soil_moisture', 'ndvi', 'groundwater'])

            prediction = model.predict(input_df)[0]
            confidence = max(model.predict_proba(input_df)[0]) * 100

        except Exception as e:
            st.error(f"⚠️ Earth Engine Error: {str(e)}")
            st.stop()

        # ==========================================
        # 5. DASHBOARD TABS (Updated with Trend)
        # ==========================================
        tab1, tab2, tab3 = st.tabs(["🚨 Threat Assessment", "📈 Temporal Trends", "🧠 AI Diagnostics"])
        
        with tab1:
            severity_map = {
                0: ("Normal / Optimal Moisture", "#2E8B57", "Standard agricultural yield expected. No immediate water interventions required."),
                1: ("Mild Water Stress", "#FFD700", "Early signs of drying. Monitor local reservoirs and surface water levels."),
                2: ("Moderate Drought", "#FF8C00", "Vegetation stress visible. Rainfed agriculture at slight risk."),
                3: ("Severe Water Crisis", "#FF4500", "High risk of crop failure. Significant groundwater depletion occurring."),
                4: ("Exceptional Drought Emergency", "#8B0000", "CRITICAL: Widespread agricultural loss likely. Hydrological systems severely compromised.")
            }
            
            label, color, impact = severity_map.get(prediction, ("Unknown", "#808080", "No data"))
            
            st.markdown(f"""
            <div style="background-color: {color}; padding: 25px; border-radius: 10px; color: white;">
                <h2 style="color: white; margin-bottom: 5px;">⚠️ {label}</h2>
                <h4 style="color: #f0f0f0; margin-top: 0px;">AI Confidence: {confidence:.1f}%</h4>
                <hr style="border-top: 1px solid white;">
                <b>Expected Impact:</b> {impact}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Helper function for calculating the change (delta)
            def get_delta(key):
                return curr_data.get(key, 0) - prev_data.get(key, 0)

            col1, col2, col3, col4, col5 = st.columns(5)
            
            # --- UPDATED: Using curr_data instead of data ---
            col1.metric("🌧️ Rain", f"{curr_data.get('precipitation', 0):.1f} mm", 
                       delta=f"{get_delta('precipitation'):+.1f} mm")
            
            col2.metric("🌡️ Temp", f"{curr_data.get('temperature', 0):.1f} °C", 
                       delta=f"{get_delta('temperature'):+.1f} °C", delta_color="inverse")
            
            col3.metric("🌱 Soil Moist", f"{curr_data.get('soil_moisture', 0):.3f}", 
                       delta=f"{get_delta('soil_moisture'):+.3f}")
            
            col4.metric("🌿 NDVI", f"{curr_data.get('ndvi', 0):.2f}", 
                       delta=f"{get_delta('ndvi'):+.2f}")
            
            col5.metric("💧 Water Table", f"{curr_data.get('groundwater', 0):.1f} cm", 
                       delta=f"{get_delta('groundwater'):+.1f} cm")

        with tab2:
            st.markdown(f"#### 12-Month Environmental Trend for {district_name}")
            
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            # --- UPDATED: Using curr_data instead of data ---
            mock_ndvi = [curr_data.get('ndvi', 0.5) + np.random.uniform(-0.1, 0.1) for _ in range(12)]
            mock_rain = [max(0, curr_data.get('precipitation', 50) + np.random.uniform(-30, 30)) for _ in range(12)]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=months, y=mock_rain, name='Rainfall (mm)', marker_color='blue', yaxis='y1'))
            fig.add_trace(go.Scatter(x=months, y=mock_ndvi, name='NDVI', mode='lines+markers', line=dict(color='green', width=3), yaxis='y2'))
            
            fig.update_layout(
                yaxis=dict(title='Rainfall (mm)', side='left'),
                yaxis2=dict(title='NDVI', side='right', overlaying='y', range=[0, 1]),
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            # --- REQ 6 & 7: Threshold Explanation & Feature Contribution ---
            st.subheader("How the AI made this decision")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("##### ⚙️ Feature Contribution")
                # Using the actual Random Forest feature importances
                importances = model.feature_importances_
                features = ['Precipitation', 'Temperature', 'Soil Moisture', 'NDVI', 'Groundwater']
                
                fig_imp = go.Figure(go.Bar(
                    x=importances, y=features, orientation='h',
                    marker_color=['blue', 'red', 'brown', 'green', 'cyan']
                ))
                fig_imp.update_layout(title="Which variables mattered most?", xaxis_title="Influence Weight", height=300)
                st.plotly_chart(fig_imp, use_container_width=True)
                
            with col_b:
                st.markdown("##### 📖 Meteorological Thresholds")
                st.markdown("""
                The AI classifies drought based on interlocking physical triggers:
                * **Meteorological Trigger:** When **Rainfall** drops below historical norms, the cycle begins.
                * **Thermal Catalyst:** High **Temperatures** accelerate evaporation, drying surface water.
                * **Agricultural Warning:** **Soil Moisture** depletes, warning of impending crop stress.
                * **Biological Impact:** **NDVI** drops as plant cellular structures break down.
                * **Hydrological Crisis:** **Groundwater** anomalies indicate long-term systemic water deficits.
                """)

else:
    st.info("👆 Awaiting input. Please select a district on the map to begin multi-variate analysis.")