import streamlit as st 

from web_functions import predict 

def app(df, x, y):

    st.title("Halaman Prediksi")

    col1, col2, col3 = st.columns(3)

    with col1:
        bp = st.text_input ('Input Nilai bp')
    with col1:
        sg = st.text_input ('Input Nilai sg')
    with col1:
        al = st.text_input ('Input Nilai al')
    with col1:
        su = st.text_input ('Input Nilai su')
    with col1:
        rbc = st.text_input ('Input Nilai rbc')
    with col1:
        pc = st.text_input ('Input Nilai pc')
    with col1:
        pcc = st.text_input ('Input Nilai pcc')
    with col1:
        ba = st.text_input ('Input Nilai ba')
    with col2:
        bgr = st.text_input ('Input Nilai bgr')
    with col2:
        bu = st.text_input ('Input Nilai bu')
    with col2:
        sc = st.text_input ('Input Nilai sc')
    with col2:
        sod = st.text_input ('Input Nilai sod')
    with col2:
        pot = st.text_input ('Input Nilai pot')
    with col2:
        hemo = st.text_input ('Input Nilai hemo')
    with col2:
        pcv = st.text_input ('Input Nilai pcv') 
    with col2:
        wc = st.text_input ('Input Nilai wc')
    with col3:
        rc = st.text_input ('Input Nilai rc')
    with col3:
        htn = st.text_input ('Input Nilai htn')
    with col3:
        dm = st.text_input ('Input Nilai dm')
    with col3:
        cad = st.text_input ('Input Nilai cad')
    with col3:
        appet = st.text_input ('Input Nilai appet')
    with col3:
        pe = st.text_input ('Input Nilai pe')
    with col3:
        ane = st.text_input ('Input Nilai ane')

    features = [bp,sg,al,su,rbc,pc,pcc,ba,bgr,bu,sc,sod,pot,hemo,pcv,wc,rc,htn,dm,cad,appet,pe,ane]

    if st.button("Prediksi"):
        prediction, score = predict(x,y,features)
        score = score
        st.info("Prediksi Sukses...")

        if(prediction == 1):
            st.warning("Orang tersebut rentan terkena penyakit ginjal")
        else:
            st.success("Orang tersebut relatif aman dari penyakit ginjal")
        
        st.write("Model yang digunakan memiliki tingkat akurasi", (score*100),"%")