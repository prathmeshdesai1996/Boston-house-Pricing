import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

# Load the saved model
reg_model = pickle.load(open('regmodel.pkl', 'rb'))

# Load the Boston House Pricing Dataset
boston = load_boston()
boston_data = pd.DataFrame(boston.data, columns=boston.feature_names)

# Define a function for user input
def get_user_input():
    crim = st.sidebar.slider('CRIM (per capita crime rate by town)', float(boston_data.CRIM.min()), float(boston_data.CRIM.max()), float(boston_data.CRIM.mean()))
    zn = st.sidebar.slider('ZN (proportion of residential land zoned for lots over 25,000 sq.ft.)', float(boston_data.ZN.min()), float(boston_data.ZN.max()), float(boston_data.ZN.mean()))
    indus = st.sidebar.slider('INDUS (proportion of non-retail business acres per town)', float(boston_data.INDUS.min()), float(boston_data.INDUS.max()), float(boston_data.INDUS.mean()))
    chas = st.sidebar.slider('CHAS (Charles River dummy variable, 1 if tract bounds river; 0 otherwise)', float(boston_data.CHAS.min()), float(boston_data.CHAS.max()), float(boston_data.CHAS.mean()))
    nox = st.sidebar.slider('NOX (nitric oxides concentration (parts per 10 million))', float(boston_data.NOX.min()), float(boston_data.NOX.max()), float(boston_data.NOX.mean()))
    rm = st.sidebar.slider('RM (average number of rooms per dwelling)', float(boston_data.RM.min()), float(boston_data.RM.max()), float(boston_data.RM.mean()))
    age = st.sidebar.slider('AGE (proportion of owner-occupied units built prior to 1940)', float(boston_data.AGE.min()), float(boston_data.AGE.max()), float(boston_data.AGE.mean()))
    dis = st.sidebar.slider('DIS (weighted distances to five Boston employment centres)', float(boston_data.DIS.min()), float(boston_data.DIS.max()), float(boston_data.DIS.mean()))
    rad = st.sidebar.slider('RAD (index of accessibility to radial highways)', float(boston_data.RAD.min()), float(boston_data.RAD.max()), float(boston_data.RAD.mean()))
    tax = st.sidebar.slider('TAX (full-value property-tax rate per $10,000)', float(boston_data.TAX.min()), float(boston_data.TAX.max()), float(boston_data.TAX.mean()))
    ptratio = st.sidebar.slider('PTRATIO (pupil-teacher ratio by town)', float(boston_data.PTRATIO.min()), float(boston_data.PTRATIO.max()), float(boston_data.PTRATIO.mean()))
    b = st.sidebar.slider('B (1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town)', float(boston_data.B.min()), float(boston_data.B.max()), float(boston_data.B.mean()))
    lstat = st.sidebar.slider('LSTAT (% lower status of the population)', float(boston_data.LSTAT.min()), float(boston_data.LSTAT.max()), float(boston_data.LSTAT.mean()))
    
    # Store the user input in a dictionary
    data = {'CRIM': crim,
            'ZN': zn,
            'INDUS': indus,
            'CHAS': chas,
            'NOX': nox,
            'RM': rm,
            'AGE': age,
            'DIS': dis,
            'RAD': rad,
            'TAX': tax,
            'PTRATIO': ptratio,
            'B': b,
            'LSTAT': lstat}
    features = pd.DataFrame(data, index=[0])
    return features


# Collecting user input in the dataframe
df = get_user_input()

# Displaying the input values
st.write('The input values are:', df)

# Predicting the output value for the input
prediction = reg_model.predict(df)

# Displaying the predicted output
st.write('The predicted house price is:', prediction[0])