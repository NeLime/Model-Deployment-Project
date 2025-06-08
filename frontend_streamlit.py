import streamlit as st
import pandas as pd
import cloudpickle
import requests

st.set_page_config(page_title="Prediksi Hospitalisasi COVID", layout="centered")

st.title("🩺 Denmark COVID Patient Outcome Analyzer")

# Load model (sebagai fallback jika API tidak tersedia)
try:
    with open('random_forest_pipeline.pkl', 'rb') as f:
        local_model = cloudpickle.load(f)
    st.success("✅ Model lokal berhasil dimuat (sebagai fallback).")
except Exception as e:
    st.warning(f"⚠️ Gagal memuat model lokal: {e}")
    local_model = None

# Form input
def user_input():
    age = st.slider('Umur', 0, 100, 30)
    doses = st.slider('Jumlah Dosis Vaksin yang Diterima', 0, 3, 2)
    bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=22.0)

    gender = st.selectbox('Jenis Kelamin', ['Male', 'Female'])
    region = st.selectbox('Region', ['Hovedstaden', 'SjÃ¦lland', 'Syddanmark', 'Nordjylland', 'Midtjylland'])
    preexisting = st.selectbox('Penyakit Penyerta', ['Cardiovascular', 'Obesity', 'Diabetes', 'Hypertension', 'Asthma'])
    strain = st.selectbox('Varian COVID', ['Alpha', 'Beta', 'Delta', 'Omicron', 'XBB.1.5'])
    symptoms = st.selectbox('Gejala', ['Mild', 'Moderate', 'Severe'])
    severity = st.selectbox('Tingkat Keparahan', ['Low', 'Moderate', 'High', 'Critical'])
    icu = st.selectbox('Dirawat di ICU', ['Yes', 'No'])
    ventilator = st.selectbox('Dibantu Ventilator', ['Yes', 'No'])
    recovered = st.selectbox('Sudah Sembuh', ['Yes', 'No'])
    reinfection = st.selectbox('Pernah Terinfeksi Ulang', ['Yes', 'No'])
    vac_status = st.selectbox('Status Vaksinasi', ['Yes','No'])
    occupation = st.selectbox('Pekerjaan', ['Student', 'Office Worker', 'Unemployed', 'Driver', 'Teacher', 'Healthcare'])
    smoking = st.selectbox('Perokok', ['Never', 'Current', 'Former'])

    return {
        'Age': age,
        'Doses_Received': doses,
        'BMI': bmi,
        'Gender': gender,
        'Region': region,
        'Preexisting_Condition': preexisting,
        'COVID_Strain': strain,
        'Symptoms': symptoms,
        'Severity': severity,
        'ICU_Admission': icu,
        'Ventilator_Support': ventilator,
        'Recovered': recovered,
        'Reinfection': reinfection,
        'Vaccination_Status': vac_status,
        'Occupation': occupation,
        'Smoking_Status': smoking
    }

input_data = user_input()

if st.button('Prediksi'):
    try:
        # Coba prediksi melalui API terlebih dahulu
        response = requests.post("http://127.0.0.1:8001/predict/", json=input_data)
        
        if response.status_code == 200:
            result = response.json()
            st.subheader("Hasil Prediksi (via API)")
            st.write(f"**Prediksi:** {result['prediction']}")
            st.write(f"**Confidence:** {result['confidence']:.2%}")
            st.write("**Probabilitas:**")
            st.json(result['probabilities'])
        else:
            raise Exception(f"API Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.warning("⚠️ Tidak dapat terhubung ke API, menggunakan model lokal...")
        if local_model:
            try:
                # Fallback ke model lokal
                df_input = pd.DataFrame([input_data])
                pred = local_model.predict(df_input)
                proba = local_model.predict_proba(df_input)
                
                st.subheader("Hasil Prediksi (Model Lokal)")
                result_text = '🛏️ Dirawat (Hospitalized)' if pred[0] == 1 else '🏠 Tidak Dirawat'
                st.success(f"**Hasil Prediksi:** {result_text}")
                st.write(f"**Confidence:** {max(proba[0]):.2%}")
                st.write("**Probabilitas:**")
                st.json({
                    "Tidak Dirawat": float(proba[0][0]),
                    "Dirawat": float(proba[0][1])
                })
            except Exception as e:
                st.error(f"❌ Error prediksi model lokal: {e}")
        else:
            st.error("❌ Tidak ada model lokal yang tersedia sebagai fallback")
            
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {e}")