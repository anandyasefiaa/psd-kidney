import streamlit as st
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from web_functions import predict, train_model, train_naive_bayes

def app(df, x, y):
    st.title("Halaman Klasifikasi")

    # Pilihan metode klasifikasi
    st.subheader("Pilih Metode Klasifikasi")
    model_choice = st.selectbox("Metode", ["Decision Tree", "Naive Bayes"])

    # Input fitur
    col1, col2, col3 = st.columns(3)

    with col1:
        bp = st.text_input ('Input Nilai Blood Pressure')
        sg = st.text_input ('Input Nilai Specific Gravity')
        al = st.text_input ('Input Nilai Albumin')
        su = st.text_input ('Input Nilai Sugar')
        rbc = st.text_input ('Input Nilai Red Blood Cells')
        pc = st.text_input ('Input Nilai Pus Cell')
        pcc = st.text_input ('Input Nilai Pus Cell Clumps')
        ba = st.text_input ('Input Nilai Bacteria')

    with col2:
        bgr = st.text_input ('Input Nilai Blood Glucose Random')
        bu = st.text_input ('Input Nilai Blood Urea')
        sc = st.text_input ('Input Nilai Serum Creatinine')
        sod = st.text_input ('Input Nilai Sodium')
        pot = st.text_input ('Input Nilai Potassium')
        hemo = st.text_input ('Input Nilai Hemoglobin')
        pcv = st.text_input ('Input Nilai Packed Cell Volume ') 
        wc = st.text_input ('Input Nilai White Cell Count')

    with col3:
        rc = st.text_input ('Input Nilai rc')
        htn = st.text_input ('Input Nilai Hypertension')
        dm = st.text_input ('Input Nilai Diabetes Mellitus')
        cad = st.text_input ('Input Nilai cad')
        appet = st.text_input ('Input Nilai appet')
        pe = st.text_input ('Input Nilai pe')
        ane = st.text_input ('Input Nilai ane')

    # Masukkan fitur ke dalam list dan ubah ke tipe numerik
    features = [bp,sg,al,su,rbc,pc,pcc,ba,bgr,bu,sc,sod,pot,hemo,pcv,wc,rc,htn,dm,cad,appet,pe,ane]
    numeric_features = []

    for feature in features:
        try:
            # Konversi input menjadi float, ganti nilai kosong atau tidak valid dengan 0
            numeric_features.append(float(feature) if feature.strip() else 0.0)
        except ValueError:
            numeric_features.append(0.0)  # Jika tidak bisa dikonversi, ganti dengan 0

    # Prediksi ketika tombol diklik
    if st.button("Klasifikasi"):
        if model_choice == "Decision Tree":
            prediction, score = predict(x, y, numeric_features)
        elif model_choice == "Naive Bayes":
            model, score = train_naive_bayes(x, y)
            prediction = model.predict([numeric_features])  # Pastikan numeric_features adalah 2D array

        st.info("Klasifikasi Sukses...")

        if prediction == 1:
            st.warning("Orang tersebut rentan terkena penyakit ginjal.")
        else:
            st.success("Orang tersebut relatif aman dari penyakit ginjal.")

        st.write("Model yang digunakan memiliki tingkat akurasi", (score * 100), "%")
