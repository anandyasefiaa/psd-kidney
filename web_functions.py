import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import streamlit as st

@st.cache()
def load_data():
    df = pd.read_csv('kidney_disease.csv')

    x = df[[ "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", 
         "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", 
         "wc", "rc", "htn", "dm", "cad", "appet", "pe", "ane" ]]
    y = df[[ "classification" ]]
    
    return df, x, y

@st.cache
def train_naive_bayes(x, y):
    model = GaussianNB()
    model.fit(x, y.values.ravel())  # Naive Bayes membutuhkan y dalam format array 1D
    score = model.score(x, y)
    return model, score

@st.cache
def train_model(x,y):
    model = DecisionTreeClassifier(
        ccp_alpha=0.0, class_weight=None, criterion='entropy', 
        max_depth=4, max_features=None, max_leaf_nodes=None, 
        min_impurity_decrease=0.0, min_samples_leaf=1, 
        min_samples_split=2, min_weight_fraction_leaf=0.0, 
        random_state=42, splitter='best'
    )

    model.fit(x,y)

    score = model.score(x,y)

    return model, score


def predict(x,y, features):
    model, score = train_model(x,y)
    prediction = model.predict(np.array(features).reshape(1,-1))

    return prediction, score
