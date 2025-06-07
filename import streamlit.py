import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
import joblit

# Load model dari file pickle
model = pickle.load(open('random_forest_model_joblit.pkl', 'rb'))

st.title("Prediksi Gejala Long COVID")
st.write("Isi data pasien berikut untuk memprediksi kemungkinan gejala Long COVID.")

# --- Input Data ---
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Umur", 0, 120, 30)
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    region = st.selectbox("Wilayah", ["Hovedstaden", "Sjælland", "Syddanmark", "Nordjylland", "Midtjylland"])
    precond = st.selectbox("Kondisi Penyerta", ["None", "Asthma", "Obesity", "Hypertension", "Cardiovascular", "Diabetes"])
    inf_date = st.date_input("Tanggal Infeksi", value=datetime.today())
    strain = st.selectbox("Varian COVID", ["Alpha", "Beta", "Delta", "Omicron", "XBB.1.5"])
    symptoms = st.selectbox("Gejala Awal", ["Mild", "Moderate", "Severe"])
    severity = st.selectbox("Tingkat Keparahan", ["Low", "Moderate", "High", "Critical"])
    hospitalized = st.radio("Dirawat di RS?", ["Yes", "No"])
    icu = st.radio("Masuk ICU?", ["Yes", "No"])
    ventilator = st.radio("Gunakan Ventilator?", ["Yes", "No"])
with col2:
    admit_date = st.date_input("Tanggal Masuk RS", value=datetime.today()) if hospitalized == "Yes" else None
    discharge_date = st.date_input("Tanggal Keluar RS", value=datetime.today()) if hospitalized == "Yes" else None
    recovered = st.radio("Sudah Sembuh?", ["Yes", "No"])
    recovery_date = st.date_input("Tanggal Sembuh", value=datetime.today()) if recovered == "Yes" else None
    reinfection = st.radio("Pernah Terinfeksi Ulang?", ["Yes", "No"])
    reinf_date = st.date_input("Tanggal Reinfeksi", value=datetime.today()) if reinfection == "Yes" else None
    vaccinated = st.radio("Sudah Vaksin?", ["Yes", "No"])
    vaccine_type = st.selectbox("Jenis Vaksin", ["None", "Pfizer", "Moderna", "AstraZeneca", "Janssen"]) if vaccinated == "Yes" else "None"
    doses = st.number_input("Jumlah Dosis", 0, 10, value=2) if vaccinated == "Yes" else 0
    last_dose_date = st.date_input("Tanggal Dosis Terakhir", value=datetime.today()) if vaccinated == "Yes" else None
    occupation = st.selectbox("Pekerjaan", ["Healthcare", "Student", "Office Worker", "Driver", "Teacher", "Unemployed"])
    smoking = st.selectbox("Status Merokok", ["Never", "Former", "Current"])
    bmi = st.number_input("BMI", 10.0, 60.0, value=25.0)

# Tombol prediksi
if st.button("Prediksi"):
    # Susun data sesuai urutan dan format dataset
    input_df = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Region": [region],
        "Preexisting_Condition": [precond],
        "Date_of_Infection": [inf_date.strftime('%Y-%m-%d')],
        "COVID_Strain": [strain],
        "Symptoms": [symptoms],
        "Severity": [severity],
        "Hospitalized": [hospitalized],
        "Hospital_Admission_Date": [admit_date.strftime('%Y-%m-%d') if admit_date else None],
        "Hospital_Discharge_Date": [discharge_date.strftime('%Y-%m-%d') if discharge_date else None],
        "ICU_Admission": [icu],
        "Ventilator_Support": [ventilator],
        "Recovered": [recovered],
        "Date_of_Recovery": [recovery_date.strftime('%Y-%m-%d') if recovery_date else None],
        "Reinfection": [reinfection],
        "Date_of_Reinfection": [reinf_date.strftime('%Y-%m-%d') if reinf_date else None],
        "Vaccination_Status": [vaccinated],
        "Vaccine_Type": [vaccine_type],
        "Doses_Received": [doses],
        "Date_of_Last_Dose": [last_dose_date.strftime('%Y-%m-%d') if last_dose_date else None],
        "Occupation": [occupation],
        "Smoking_Status": [smoking],
        "BMI": [bmi]
    })

    # Prediksi dengan model
    try:
        pred = model.predict(input_df)[0]
        st.success(f"Hasil Prediksi Gejala Long COVID: **{pred}**")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")

