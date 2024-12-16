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
        age = st.text_input('Input Usia')
        bp = st.text_input ('Input Nilai Tekanan Darah')
        sg = st.text_input ('Input Nilai Berat jenis urin')
        al = st.text_input ('Input Nilai Kadar Albumin dalam urin')
        su = st.text_input ('Input Nilai Kadar gula dalam urin')
        rbc = st.text_input ('Input Nilai sel darah merah')
        pc = st.text_input ('Input Nilai Sel nanah')
        pcc = st.text_input ('Input Nilai Gumpalan Sel Nanah')
        ba = st.text_input ('Input Nilai Bakteri')

    with col2:
        bgr = st.text_input ('Input Nilai Gula Darah Acak')
        bu = st.text_input ('Input Nilai Kadar dalam darah')
        sc = st.text_input ('Input Nilai Kreatinin Serum')
        sod = st.text_input ('Input Nilai Natrium')
        pot = st.text_input ('Input Nilai Kalium')
        hemo = st.text_input ('Input Nilai kadar Hemoglobin')
        pcv = st.text_input ('Input Nilai Volume Sel Darah yang terkemas ') 
        wc = st.text_input ('Input Nilai Jumlah sel darah putih')

    with col3:
        rc = st.text_input ('Input Nilai Jumlah sel darah merah')
        htn = st.text_input ('Input Nilai Hipertensi')
        dm = st.text_input ('Input Nilai Diabetes Mellitus')
        cad = st.text_input ('Input Nilai Penyakit Arteri Koroner')
        appet = st.text_input ('Input Nilai Nafsu Makan')
        pe = st.text_input ('Input Nilai Edema pada kaki')
        ane = st.text_input ('Input Nilai anemia') 

    # Masukkan fitur ke dalam list dan ubah ke tipe numerik
    features = [age,bp,sg,al,su,rbc,pc,pcc,ba,bgr,bu,sc,sod,pot,hemo,pcv,wc,rc,htn,dm,cad,appet,pe,ane]
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
