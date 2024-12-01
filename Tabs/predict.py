import streamlit as st
from web_functions import predict, train_model, train_naive_bayes

def app(df, x, y):
    st.title("Halaman Prediksi")

    # Pilihan metode prediksi
    st.subheader("Pilih Metode Prediksi")
    model_choice = st.selectbox("Metode", ["Decision Tree", "Naive Bayes"])

    # Input fitur
    col1, col2, col3 = st.columns(3)

    with col1:
        bp = st.text_input('Input Nilai bp')
        sg = st.text_input('Input Nilai sg')
        al = st.text_input('Input Nilai al')
        su = st.text_input('Input Nilai su')
        rbc = st.text_input('Input Nilai rbc')
        pc = st.text_input('Input Nilai pc')
        pcc = st.text_input('Input Nilai pcc')
        ba = st.text_input('Input Nilai ba')

    with col2:
        bgr = st.text_input('Input Nilai bgr')
        bu = st.text_input('Input Nilai bu')
        sc = st.text_input('Input Nilai sc')
        sod = st.text_input('Input Nilai sod')
        pot = st.text_input('Input Nilai pot')
        hemo = st.text_input('Input Nilai hemo')
        pcv = st.text_input('Input Nilai pcv')
        wc = st.text_input('Input Nilai wc')

    with col3:
        rc = st.text_input('Input Nilai rc')
        htn = st.text_input('Input Nilai htn')
        dm = st.text_input('Input Nilai dm')
        cad = st.text_input('Input Nilai cad')
        appet = st.text_input('Input Nilai appet')
        pe = st.text_input('Input Nilai pe')
        ane = st.text_input('Input Nilai ane')

    # Masukkan fitur ke dalam list
    features = [bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane]

    # Prediksi ketika tombol diklik
    if st.button("Prediksi"):
        if model_choice == "Decision Tree":
            prediction, score = predict(x, y, features)
        elif model_choice == "Naive Bayes":
            model, score = train_naive_bayes(x, y)
            prediction = model.predict([features])

        st.info("Prediksi Sukses...")

        if prediction == 1:
            st.warning("Orang tersebut rentan terkena penyakit ginjal.")
        else:
            st.success("Orang tersebut relatif aman dari penyakit ginjal.")

        st.write("Model yang digunakan memiliki tingkat akurasi", (score * 100), "%")
