import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="centered")

MODEL_PATH = Path(__file__).parent / "laptop_price_model.pkl"


# --- Feature engineering (identical logic to the training notebook) ---
def to_gb(token):
    m = re.search(r'(\d+(?:\.\d+)?)\s*(TB|GB)', str(token), flags=re.I)
    if not m:
        return 0.0
    value = float(m.group(1))
    return value * 1024 if m.group(2).upper() == 'TB' else value


def memory_features(value):
    out = {'SSD_GB': 0.0, 'HDD_GB': 0.0, 'Flash_GB': 0.0, 'Hybrid_GB': 0.0}
    for part in str(value).split('+'):
        part = part.strip()
        gb = to_gb(part)
        low = part.lower()
        if 'ssd' in low:
            out['SSD_GB'] += gb
        elif 'hdd' in low:
            out['HDD_GB'] += gb
        elif 'flash' in low:
            out['Flash_GB'] += gb
        elif 'hybrid' in low:
            out['Hybrid_GB'] += gb
    return out


def cpu_type(value):
    low = str(value).lower()
    if 'intel core i7' in low: return 'Intel Core i7'
    if 'intel core i5' in low: return 'Intel Core i5'
    if 'intel core i3' in low: return 'Intel Core i3'
    if 'intel' in low: return 'Other Intel Processor'
    if 'amd' in low: return 'AMD Processor'
    return 'Other Processor'


def os_group(value):
    low = str(value).lower()
    if 'windows' in low: return 'Windows'
    if 'mac' in low: return 'Mac'
    if 'linux' in low: return 'Linux'
    if 'chrome' in low: return 'Chrome OS'
    if 'no os' in low: return 'No OS'
    if 'android' in low: return 'Android'
    return 'Other'


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=raw.index)
    x['Company'] = raw['Company'].astype(str)
    x['TypeName'] = raw['TypeName'].astype(str)
    x['Inches'] = pd.to_numeric(raw['Inches'], errors='coerce')
    x['Ram_GB'] = raw['Ram'].astype(str).str.extract(r'(\d+(?:\.\d+)?)')[0].astype(float)
    x['Weight_kg'] = raw['Weight'].astype(str).str.replace('kg', '', regex=False).astype(float)

    screen = raw['ScreenResolution'].astype(str)
    x['Touchscreen'] = screen.str.contains('Touchscreen', case=False).astype(int)
    x['IPS'] = screen.str.contains('IPS', case=False).astype(int)
    wh = screen.str.extract(r'(\d+)x(\d+)').astype(float)
    x['PPI'] = np.sqrt(wh[0] ** 2 + wh[1] ** 2) / x['Inches']

    cpu = raw['Cpu'].astype(str)
    x['CPU_Type'] = cpu.map(cpu_type)
    x['CPU_GHz'] = cpu.str.extract(r'(\d+(?:\.\d+)?)GHz')[0].astype(float)

    x['GPU_Brand'] = raw['Gpu'].astype(str).str.split().str[0]
    x['OpSys_Group'] = raw['OpSys'].astype(str).map(os_group)

    mem = raw['Memory'].apply(memory_features).apply(pd.Series)
    x = pd.concat([x, mem], axis=1)
    x['TotalStorage_GB'] = mem.sum(axis=1)

    x['Cpu_Raw'] = raw['Cpu'].astype(str)
    x['Gpu_Raw'] = raw['Gpu'].astype(str)
    x['ScreenResolution_Raw'] = raw['ScreenResolution'].astype(str)
    x['Memory_Raw'] = raw['Memory'].astype(str)
    return x


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle['model'], bundle.get('model_name', 'model'), bundle.get('metrics', {})


model, model_name, metrics = load_model()

st.title("💻 Laptop Price Predictor")
st.caption(f"Model: **{model_name}** — R² test set: {metrics.get('R2', 0):.3f} | MAE: €{metrics.get('MAE', 0):,.0f}")
st.write("Isi spesifikasi laptop di bawah untuk memperkirakan harganya (dalam Euro).")

companies = ['Apple', 'HP', 'Acer', 'Asus', 'Dell', 'Lenovo', 'Chuwi', 'MSI',
             'Microsoft', 'Toshiba', 'Huawei', 'Xiaomi', 'Vero', 'Razer',
             'Mediacom', 'Samsung', 'Google', 'Fujitsu', 'LG']
types = ['Ultrabook', 'Notebook', 'Netbook', 'Gaming', '2 in 1 Convertible', 'Workstation']
os_options = ['Windows 10', 'Windows 7', 'Windows 10 S', 'macOS', 'Mac OS X',
              'Linux', 'Chrome OS', 'Android', 'No OS']

col1, col2 = st.columns(2)
with col1:
    company = st.selectbox("Company", companies)
    type_name = st.selectbox("Type", types)
    inches = st.slider("Screen size (inches)", 10.0, 18.4, 15.6, 0.1)
    ram = st.selectbox("RAM", ['2GB', '4GB', '6GB', '8GB', '12GB', '16GB', '24GB', '32GB', '64GB'], index=3)
    weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
    opsys = st.selectbox("Operating System", os_options)

with col2:
    resolution = st.selectbox("Screen Resolution", [
        '1366x768', '1440x900', 'Full HD 1920x1080',
        'IPS Panel Full HD 1920x1080', '4K Ultra HD 3840x2160',
        'IPS Panel Retina Display 2560x1600', 'IPS Panel Retina Display 2880x1800',
        'Touchscreen 2560x1440', 'Quad HD+ 3200x1800',
    ])
    cpu = st.selectbox("CPU", [
        'Intel Core i3 6006U 2GHz', 'Intel Core i5 7200U 2.5GHz',
        'Intel Core i5 8250U 1.6GHz', 'Intel Core i7 7700HQ 2.8GHz',
        'Intel Core i7 8550U 1.8GHz', 'Intel Celeron Dual Core N3060 1.6GHz',
        'AMD A9-Series 9420 3GHz', 'AMD Ryzen 1600 3.2GHz',
    ])
    memory = st.selectbox("Storage", [
        '128GB SSD', '256GB SSD', '512GB SSD', '1TB SSD',
        '500GB HDD', '1TB HDD', '2TB HDD',
        '128GB SSD +  1TB HDD', '256GB SSD +  1TB HDD', '256GB Flash Storage',
    ])
    gpu = st.selectbox("GPU", [
        'Intel HD Graphics 620', 'Intel UHD Graphics 620', 'Intel Iris Plus Graphics 640',
        'Nvidia GeForce GTX 1050', 'Nvidia GeForce GTX 1060', 'Nvidia GeForce MX150',
        'AMD Radeon 530', 'AMD Radeon R5',
    ])

if st.button("Prediksi Harga", type="primary", use_container_width=True):
    raw = pd.DataFrame([{
        'Company': company, 'TypeName': type_name, 'Inches': inches,
        'ScreenResolution': resolution, 'Cpu': cpu, 'Ram': ram, 'Memory': memory,
        'Gpu': gpu, 'OpSys': opsys, 'Weight': f'{weight}kg',
    }])
    features = engineer_features(raw)
    pred_log = model.predict(features)[0]
    price = float(np.expm1(pred_log))
    st.success(f"### Estimasi harga: € {price:,.2f}")
    with st.expander("Lihat detail fitur yang dipakai model"):
        st.dataframe(features.T.rename(columns={0: 'value'}))
