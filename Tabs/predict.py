import streamlit as st
from web_functions import predict, train_model, train_naive_bayes

def app(df, x, y):
    st.title("Halaman Prediksi")

    # Pilihan metode prediksi
    st.subheader("Pilih Metode Prediksi")
    model_choice = st.selectbox("Metode", ["Decision Tree", "Naive Bayes"])

    # Pastikan data preprocessing selesai
    if 'preprocessed_data' in st.session_state and 'Imp_features' in st.session_state:
        preprocessed_data = st.session_state['preprocessed_data']
        Imp_features = st.session_state['Imp_features']

        if not Imp_features:
            st.error("Tidak ada fitur yang memenuhi syarat korelasi. Pastikan seleksi fitur telah dilakukan.")
            return

        # Input fitur
        st.subheader("Input Data Prediksi")
        user_inputs = {}

        for feature in Imp_features:
            user_input = st.text_input(f"Masukkan nilai untuk fitur '{feature}'", value="0")
            user_inputs[feature] = user_input

        # Konversi input menjadi tipe numerik
        numeric_features = []
        for key, value in user_inputs.items():
            try:
                numeric_features.append(float(value) if value.strip() else 0.0)
            except ValueError:
                numeric_features.append(0.0)  # Jika input invalid, ganti dengan 0.0

        # Prediksi ketika tombol diklik
        if st.button("Prediksi"):
            # Split data untuk pelatihan
            X = preprocessed_data[Imp_features]
            y = preprocessed_data['classification_notckd']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            if model_choice == "Decision Tree":
                model = DecisionTreeClassifier()
            elif model_choice == "Naive Bayes":
                model = GaussianNB()

            # Latih model
            model.fit(X_train, y_train)

            # Prediksi berdasarkan input pengguna
            try:
                prediction = model.predict([numeric_features])  # Pastikan input dalam bentuk 2D array
                accuracy = model.score(X_test, y_test)

                st.subheader("Hasil Prediksi")
                if prediction[0] == 1:
                    st.warning("Orang tersebut rentan terkena penyakit ginjal.")
                else:
                    st.success("Orang tersebut relatif aman dari penyakit ginjal.")

                st.write(f"Model {model_choice} memiliki tingkat akurasi: {accuracy * 100:.2f}%")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi: {e}")
    else:
        st.error("Lakukan preprocessing dan seleksi fitur terlebih dahulu pada halaman Preprocessing.")
