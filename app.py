
import streamlit as st
import pandas as pd
import joblib
import sqlite3
import yagmail
from datetime import datetime
import matplotlib.pyplot as plt

# Load model
model = joblib.load("advanced_temperature_model.pkl")

# Safe temperature ranges
product_temp_limits = {
    'milk': (2, 4),
    'curd': (3, 5),
    'butter': (5, 7),
    'cheese': (7, 10),
    'ice-cream': (-22, -18),
    'flavored_beverage': (4, 6)
}

# Create database if not exists
def create_database():
    conn = sqlite3.connect("cold_chain.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            product_type TEXT,
            external_temp REAL,
            current_room_temp REAL,
            humidity REAL,
            volume_kg REAL,
            packaging_type TEXT,
            storage_time_hr REAL,
            airflow_rating TEXT,
            predicted_temp REAL,
            email TEXT,
            alert_sent TEXT
        )
    """)
    conn.commit()
    conn.close()

create_database()

# Page config and banner
st.set_page_config(page_title="Smart Cold Chain", page_icon="🧊", layout="centered")
st.image("https://images.unsplash.com/photo-1570872622029-213dc5f6bb32", use_container_width=True)
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>🧠 AI Cold Chain Temperature Optimizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Smart prediction & alert system for dairy product safety.</p>", unsafe_allow_html=True)
st.markdown("---")

# Form input
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        product = st.selectbox("Product Type", list(product_temp_limits.keys()))
        external_temp = st.slider("External Temperature (°C)", 20, 45, 30)
        humidity = st.slider("Humidity (%)", 30, 90, 60)
        volume_kg = st.slider("Volume (kg)", 1, 100, 10)
        storage_time = st.slider("Storage Duration (hr)", 1, 72, 12)

    with col2:
        current_temp = st.slider("Current Room Temp (°C)", -25, 25, 5)
        packaging = st.selectbox("Packaging Type", ['plastic', 'glass', 'tetrapack', 'metal'])
        airflow = st.selectbox("Airflow Rating", ['poor', 'moderate', 'good'])
        email = st.text_input("Email for Alerts (optional)")

    submitted = st.form_submit_button("📡 Predict Temperature")

if submitted:
    input_data = pd.DataFrame([{
        'product_type': product,
        'external_temp': external_temp,
        'current_room_temp': current_temp,
        'humidity': humidity,
        'volume_kg': volume_kg,
        'packaging_type': packaging,
        'storage_time_hr': storage_time,
        'airflow_rating': airflow
    }])

    predicted_temp = round(model.predict(input_data)[0], 2)
    st.success(f"✅ Ideal Room Temperature: {predicted_temp} °C")

    safe_min, safe_max = product_temp_limits[product]
    alert_needed = predicted_temp < safe_min or predicted_temp > safe_max
    alert_sent = "Yes" if alert_needed else "No"

    # Store to DB
    conn = sqlite3.connect("cold_chain.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (
            timestamp, product_type, external_temp, current_room_temp,
            humidity, volume_kg, packaging_type, storage_time_hr,
            airflow_rating, predicted_temp, email, alert_sent
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        product, external_temp, current_temp,
        humidity, volume_kg, packaging, storage_time,
        airflow, predicted_temp, email, alert_sent
    ))
    conn.commit()
    conn.close()

    # Email Alert
    if email and alert_needed:
        try:
            yag = yagmail.SMTP(user="your_email@gmail.com", password="your_app_password")
            yag.send(
                to=email,
                subject=f"🚨 {product.upper()} Alert: Temp Out of Range!",
                contents=f"""
ALERT: Temperature for {product} is out of range!

🧊 Predicted Temp: {predicted_temp} °C
✅ Safe Range: {safe_min} °C to {safe_max} °C

Please act to preserve product quality!
                """
            )
            st.warning("🚨 Email alert sent successfully!")
        except Exception as e:
            st.error(f"Email error: {e}")

# Historical Plot
st.markdown("---")
st.subheader("📈 Temperature Prediction Trends")
conn = sqlite3.connect("cold_chain.db")
df = pd.read_sql_query("SELECT * FROM predictions", conn)
conn.close()

if not df.empty:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    fig, ax = plt.subplots(figsize=(10, 4))
    for ptype in df['product_type'].unique():
        temp_df = df[df['product_type'] == ptype]
        ax.plot(temp_df['timestamp'], temp_df['predicted_temp'], label=ptype, marker='o')
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Historical Ideal Temperature Predictions")
    ax.legend()
    st.pyplot(fig)
else:
    st.info("No data to display yet.")

# Show database
if st.checkbox("📄 View All Records"):
    st.dataframe(df)
