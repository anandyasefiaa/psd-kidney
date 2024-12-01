import warnings
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import tree
import streamlit as st 

from web_functions import train_model

def app(df, x, y):

    warnings.filterwarnings('ignore')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title("Visualisasi Prediksi")

    # Konversi x dan y ke array numerik
    x_numeric = x.apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy()
    y_numeric = y.values.ravel()  # pastikan y menjadi array 1D

    if st.checkbox("Plot Confusion Matrix"):
        model, score = train_model(x_numeric, y_numeric)
        plt.figure(figsize=(10, 6))
        disp = ConfusionMatrixDisplay.from_estimator(
            model, x_numeric, y_numeric, cmap=plt.cm.Blues
        )
        st.pyplot()

    if st.checkbox("Plot Decision Tree"):
        model, score = train_model(x_numeric, y_numeric)
        dot_data = tree.export_graphviz(
            decision_tree=model, max_depth=3, out_file=None, filled=True, rounded=True,
            feature_names=x.columns, class_names=['nockd', 'ckd']
        )
        st.graphviz_chart(dot_data)
