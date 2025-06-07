import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

# Memuat model Random Forest dari file .pkl
model = pickle.load(open('random_forest_model.pkl', 'rb'))

# Judul dan deskripsi
st.title("Aplikasi Prediksi Penyakit Terkait COVID-19")
st.write("Masukkan data pasien di bawah ini untuk memprediksi kemungkinan penyakit terkait COVID-19 (mis. gejala Long COVID).")

# --- Input Data Pasien ---
st.header("Input Data Pasien")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Umur", min_value=0, max_value=120, value=30)
    gender = st.radio("Jenis Kelamin", ["Pria", "Wanita"])
    region = st.selectbox("Daerah", ["Hovedstaden", "Sjælland", "Syddanmark", "Nordjylland", "Midtjylland"])
    occupation = st.selectbox("Pekerjaan", ["Tenaga Kesehatan", "Pelajar/Mahasiswa", "Pegawai Kantoran", "Sopir", "Guru", "Pengangguran"])
    smoking = st.selectbox("Status Merokok", ["Tidak Pernah", "Mantan Perokok", "Saat Ini"])
    bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=25.0, format="%.1f")
with col2:
    precond = st.selectbox("Kondisi Penyerta", ["Tidak ada", "Asthma", "Obesity", "Hypertension", "Cardiovascular", "Diabetes"])
    inf_date = st.date_input("Tanggal Infeksi", value=datetime.today())
    strain = st.selectbox("Varian COVID", ["Alpha", "Beta", "Delta", "Omicron", "XBB.1.5"])
    symp = st.selectbox("Gejala Awal", ["Mild", "Moderate", "Severe"])
    severity = st.selectbox("Tingkat Keparahan", ["Low", "Moderate", "High", "Critical"])

# --- Input Status Perawatan ---
st.header("Info Perawatan & Outcome")

hospitalized = st.radio("Dirawat di RS?", ["Ya", "Tidak"])
if hospitalized == "Ya":
    admit_date = st.date_input("Tanggal Masuk RS", value=datetime.today())
    discharge_date = st.date_input("Tanggal Keluar RS", value=datetime.today())
else:
    admit_date = None
    discharge_date = None

icu = st.radio("Masuk ICU?", ["Ya", "Tidak"])
ventilator = st.radio("Ventilator?", ["Ya", "Tidak"])
recovered = st.radio("Pasien Sembuh dari COVID-19?", ["Ya", "Tidak"])
if recovered == "Ya":
    recovery_date = st.date_input("Tanggal Sembuh", value=datetime.today())
else:
    recovery_date = None

reinfection = st.radio("Pernah Terinfeksi Ulang (Reinfeksi)?", ["Ya", "Tidak"])
if reinfection == "Ya":
    reinf_date = st.date_input("Tanggal Reinfeksi", value=datetime.today())
else:
    reinf_date = None

# --- Input Status Vaksinasi ---
st.header("Info Vaksinasi")

vaccinated = st.radio("Status Vaksinasi COVID-19?", ["Ya", "Tidak"])
if vaccinated == "Ya":
    vaccine_type = st.selectbox("Jenis Vaksin", ["Pfizer", "Moderna", "AstraZeneca", "Janssen", "Lainnya"])
    doses = st.number_input("Jumlah Dosis Vaksin", min_value=0, max_value=10, value=2)
    last_dose_date = st.date_input("Tanggal Dosis Terakhir", value=datetime.today())
else:
    vaccine_type = "None"
    doses = 0
    last_dose_date = None

# --- Prediksi ---
if st.button("Prediksi"):
    # Mempersiapkan data input ke dalam DataFrame
    input_df = pd.DataFrame({
        "Age": [age],
        "Gender": ["Male" if gender == "Pria" else "Female"],
        "Region": [region],
        "Preexisting_Condition": [ "None" if precond == "Tidak ada" else precond ],
        "Date_of_Infection": [inf_date.strftime("%Y-%m-%d")],
        "COVID_Strain": [strain],
        "Symptoms": [symp],
        "Severity": [severity],
        "Hospitalized": ["Yes" if hospitalized == "Ya" else "No"],
        "Hospital_Admission_Date": [admit_date.strftime("%Y-%m-%d") if admit_date else None],
        "Hospital_Discharge_Date": [discharge_date.strftime("%Y-%m-%d") if discharge_date else None],
        "ICU_Admission": ["Yes" if icu == "Ya" else "No"],
        "Ventilator_Support": ["Yes" if ventilator == "Ya" else "No"],
        "Recovered": ["Yes" if recovered == "Ya" else "No"],
        "Date_of_Recovery": [recovery_date.strftime("%Y-%m-%d") if recovery_date else None],
        "Reinfection": ["Yes" if reinfection == "Ya" else "No"],
        "Date_of_Reinfection": [reinf_date.strftime("%Y-%m-%d") if reinf_date else None],
        "Vaccination_Status": ["Yes" if vaccinated == "Ya" else "No"],
        "Vaccine_Type": [vaccine_type],
        "Doses_Received": [doses],
        "Date_of_Last_Dose": [last_dose_date.strftime("%Y-%m-%d") if last_dose_date else None],
        "Occupation": [occupation if occupation != "Pengangguran" else "Unemployed"],
        "Smoking_Status": ["Never" if smoking == "Tidak Pernah" else ("Former" if smoking == "Mantan Perokok" else "Current")],
        "BMI": [bmi]
    })
    # Melakukan prediksi menggunakan model
    pred = model.predict(input_df)[0]
    # Menentukan label hasil (Positif/Negatif)
    if pred in [0, "0", "No", "None", False]:
        label = "Negatif"
        st.error(f"Hasil Prediksi: **{label}**")
    else:
        label = "Positif"
        st.success(f"Hasil Prediksi: **{label}**")
