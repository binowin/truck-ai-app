import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Truck AI", layout="centered")

st.title("🚚 Truck Health Overview")

st.markdown("Upload a CSV file with these columns: **Engine_Temp, Oil_Pressure, RPM, Mileage**")

# File uploader
uploaded_file = st.file_uploader("Choose CSV File", type=["csv"])

# Train sample model
sample_data = {
    'Engine_Temp': [85, 95, 100, 70, 110, 90, 88, 102, 105, 98],
    'Oil_Pressure': [30, 28, 20, 40, 15, 25, 32, 18, 22, 27],
    'RPM': [2000, 2500, 3000, 1800, 3200, 2400, 2100, 2900, 3100, 2600],
    'Mileage': [100000, 120000, 130000, 90000, 140000, 110000, 102000, 135000, 145000, 115000],
    'Failure': [0, 1, 2, 0, 2, 1, 0, 2, 2, 1]
}
df = pd.DataFrame(sample_data)
X = df[['Engine_Temp', 'Oil_Pressure', 'RPM', 'Mileage']]
y = df['Failure']
model = RandomForestClassifier()
model.fit(X, y)
labels = {0: 'Low', 1: 'Medium', 2: 'High'}

# When user uploads a file
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    if {'Engine_Temp', 'Oil_Pressure', 'RPM', 'Mileage'}.issubset(data.columns):
        st.subheader("Uploaded Data:")
        st.dataframe(data)

        predictions = model.predict(data[['Engine_Temp', 'Oil_Pressure', 'RPM', 'Mileage']])
        data['Failure Risk'] = [labels[p] for p in predictions]

        st.subheader("📊 Prediction Result:")
        st.dataframe(data[['Engine_Temp', 'Oil_Pressure', 'RPM', 'Mileage', 'Failure Risk']])
    else:
        st.error("❌ Columns missing. Make sure your file has: Engine_Temp, Oil_Pressure, RPM, Mileage")
else:
    st.info("Awaiting CSV file upload...")
