import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

# Load model dari file .pkl
model = pickle.load(open('random_forest_model.pkl', 'rb'))

# Judul aplikasi
st.title("Aplikasi Prediksi Penyakit Terkait COVID-19")
st.write("Silakan isi data pasien untuk memprediksi kemungkinan penyakit terkait COVID-19 seperti gejala long COVID.")

# Bagian Input Data Pasien
st.header("1. Data Pasien")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Umur", min_value=0, max_value=120, value=30)
    gender = st.radio("Jenis Kelamin", ["Pria", "Wanita"])
    region = st.selectbox("Wilayah Domisili", ["Hovedstaden", "Sjælland", "Syddanmark", "Nordjylland", "Midtjylland"])
    occupation = st.selectbox("Pekerjaan", ["Tenaga Kesehatan", "Pelajar/Mahasiswa", "Pegawai Kantoran", "Sopir", "Guru", "Pengangguran"])
    smoking = st.selectbox("Status Merokok", ["Tidak Pernah", "Mantan Perokok", "Saat Ini"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, format="%.1f")
with col2:
    precond = st.selectbox("Kondisi Penyerta", ["Tidak ada", "Asthma", "Obesity", "Hypertension", "Cardiovascular", "Diabetes"])
    inf_date = st.date_input("Tanggal Infeksi", value=datetime.today())
    strain = st.selectbox("Varian COVID", ["Alpha", "Beta", "Delta", "Omicron", "XBB.1.5"])
    symp = st.selectbox("Gejala Awal", ["Mild", "Moderate", "Severe"])
    severity = st.selectbox("Tingkat Keparahan", ["Low", "Moderate", "High", "Critical"])

# Bagian Perawatan
st.header("2. Informasi Perawatan")

hospitalized = st.radio("Apakah Dirawat di Rumah Sakit?", ["Ya", "Tidak"])
if hospitalized == "Ya":
    admit_date = st.date_input("Tanggal Masuk RS", value=datetime.today())
    discharge_date = st.date_input("Tanggal Keluar RS", value=datetime.today())
else:
    admit_date = None
    discharge_date = None

icu = st.radio("Masuk ICU?", ["Ya", "Tidak"])
ventilator = st.radio("Menggunakan Ventilator?", ["Ya", "Tidak"])
recovered = st.radio("Sudah Sembuh dari COVID-19?", ["Ya", "Tidak"])
recovery_date = st.date_input("Tanggal Sembuh", value=datetime.today()) if recovered == "Ya" else None
reinfection = st.radio("Pernah Terinfeksi Ulang?", ["Ya", "Tidak"])
reinf_date = st.date_input("Tanggal Reinfeksi", value=datetime.today()) if reinfection == "Ya" else None

# Bagian Vaksinasi
st.header("3. Status Vaksinasi")

vaccinated = st.radio("Apakah Sudah Vaksin COVID-19?", ["Ya", "Tidak"])
if vaccinated == "Ya":
    vaccine_type = st.selectbox("Jenis Vaksin", ["Pfizer", "Moderna", "AstraZeneca", "Janssen", "Lainnya"])
    doses = st.number_input("Jumlah Dosis yang Diterima", min_value=0, max_value=10, value=2)
    last_dose_date = st.date_input("Tanggal Dosis Terakhir", value=datetime.today())
else:
    vaccine_type = "None"
    doses = 0
    last_dose_date = None

# Tombol Prediksi
if st.button("Prediksi"):
    input_df = pd.DataFrame({
        "Age": [age],
        "Gender": ["Male" if gender == "Pria" else "Female"],
        "Region": [region],
        "Preexisting_Condition": ["None" if precond == "Tidak ada" else precond],
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
        "Occupation": ["Unemployed" if occupation == "Pengangguran" else occupation],
        "Smoking_Status": ["Never" if smoking == "Tidak Pernah" else ("Former" if smoking == "Mantan Perokok" else "Current")],
        "BMI": [bmi]
    })

    try:
        pred = model.predict(input_df)[0]
        label = "Positif" if pred not in [0, "0", "No", "None", False] else "Negatif"
        if label == "Positif":
            st.success(f"Hasil Prediksi: **{label}**")
        else:
            st.error(f"Hasil Prediksi: **{label}**")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")

