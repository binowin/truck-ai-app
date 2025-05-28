import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Page setup
st.set_page_config(page_title="Truck Health Dashboard", layout="wide")
st.markdown("<h1 style='text-align: center;'>🚚 Truck Health Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload truck data to get AI-powered health prediction</p>", unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader("📤 Upload CSV File", type=["csv"])

# Sample training data for model
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
emoji_map = {
    'Low': '🟢 Low',
    'Medium': '🟡 Medium',
    'High': '🔴 High'
}

# Truck data analysis
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    if {'Engine_Temp', 'Oil_Pressure', 'RPM', 'Mileage'}.issubset(data.columns):
        predictions = model.predict(data[['Engine_Temp', 'Oil_Pressure', 'RPM', 'Mileage']])
        data['Failure Risk'] = [emoji_map[labels[p]] for p in predictions]

        st.success("✅ Prediction completed!")
        
        st.subheader("📊 Data Overview")
       st.dataframe(data)

