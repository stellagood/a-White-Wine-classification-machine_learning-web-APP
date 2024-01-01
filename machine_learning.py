# python -m streamlit run 10890091_Stella_hw2.py
# pip install streamlit


import pandas as pd
import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons
import numpy as np 
import math 
from sklearn.svm import SVC 


df = pd.read_csv("./winequality-white.csv",  sep=';')
print(df.head(5))
print()

X = df.drop('quality', axis=1)
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

# Suppose we train 3 models
model_list = ['Decision Tree (DT)', 'Support Vector Machine (SVM)', 'RandomForest(RF)']
# we can create a selection box inside the APP
classifiers = st.selectbox("Which machine learning classifier model is used?", model_list)

# Now, let' trained the model selected by the user from the model list
if classifiers == 'Decision Tree (DT)':
    # train the decision tree
    dt = DecisionTreeClassifier()
    # we train the model with fit() function 
    dt.fit(X_train, y_train)
    # after training the model, we can evaluate the model performance with test dataset
    # we can calculate the prediction accuracy
    accuracy = dt.score(X_test, y_test)
    # we can also show the evaluation results on the APP
    st.write("Classification accuracy: ", accuracy)

    # we can make prediction with trained model
    pred_dt = dt.predict(X_test)

    # then, we can calculate the confusion matrix with predicted values and real values
    con_matrix_dt = accuracy_score(y_test, pred_dt)
    # lastly, we can show the confusion matrix results on the APP
    st.write("Confusion Matrix: ", con_matrix_dt)
    print("The decision tree model is trained successfully!")
    print()
elif classifiers == 'Support Vector Machine (SVM)':
    # train the SVM
    svc = SVC()
    # we train the model with fit() function 
    svc.fit(X_train, y_train)
    # after training the model, we can evaluate the model performance with test dataset
    # we can calculate the prediction accuracy
    accuracy = svc.score(X_test, y_test)
    # we can also show the evaluation results on the APP
    st.write("Classification accuracy: ", accuracy)

    # we can make prediction with trained model
    pred_svc = svc.predict(X_test)

    # then, we can calculate the confusion matrix with predicted values and real values
    con_matrix_svc = accuracy_score(y_test, pred_svc)
    # lastly, we can show the confusion matrix results on the APP
    st.write("Confusion Matrix: ", con_matrix_svc)
    print("SVC is trained successfully!")
    print()
else:
    # train an ensemble bagging learner (RandomForest)
    rf_model = RandomForestClassifier()
    # we train the model with fit() function 
    rf_model.fit(X_train, y_train)
    # after training the model, we can evaluate the model performance with test dataset
    # we can calculate the prediction accuracy
    accuracy = rf_model.score(X_test, y_test)
    # we can also show the evaluation results on the APP
    st.write("Classification accuracy: ", accuracy)
    pred_rf_model = rf_model.predict(X_test)
    con_matrix_rf = accuracy_score(y_test, pred_rf_model)
    st.write("Confusion Matrix: ", con_matrix_rf)
    print("The ensemble bagging learner - RandomForrest is trained successfully!")
    print()


# main panel
st.title("White wine machine learning prediction!")

# side panel
st.sidebar.header("User Input Parameters.")


def user_input_features():
    fixed_acidity = st.sidebar.slider("Fixed Acidity", 6.0, 9.0, 7.0)
    volatile_acidity = st.sidebar.slider("Volatile Acidity", 0.2, 0.4, 0.27)
    citric_acid = st.sidebar.slider("Citric Acid", 0.3, 0.5, 0.36)
    residual_sugar = st.sidebar.slider("Residual Sugar", 0.1, 25.0, 20.7)
    chlorides = st.sidebar.slider("Chlorides", 0.04, 0.06)
    free_sulfur_dioxide = st.sidebar.slider("Free Sulfur Dioxide", 17, 200, 170)
    total_sulfur_dioxide = st.sidebar.slider("Total Sulfur Dioxide", 0.1, 3.0, 1.001)
    density = st.sidebar.slider("Density", 0.8, 2.0, 1.001)
    pH = st.sidebar.slider("pH", 1.0, 7.0, 3.0)
    sulphates = st.sidebar.slider("Sulphates", 0.3, 0.5, 0.45)
    alcohol = st.sidebar.slider("Alcohol", 1, 10, 6)

    # create a data dictionary
    data = {
        "fixed_acidity": float(fixed_acidity),
        "volatile_acidity": float(volatile_acidity),
        "citric_acid": float(citric_acid),
        "residual_sugar": float(residual_sugar),
        "chlorides": float(chlorides),
        "free_sulfur_dioxide": float(free_sulfur_dioxide),
        "total_sulfur_dioxide": float(total_sulfur_dioxide),
        "density": float(density),
        "pH": float(pH),
        "sulphates": float(sulphates),
        "alcohol": float(alcohol)
    }
    # create a dataframe with data stored in the dictionary
    features_df = pd.DataFrame(data=data, index=[0])
    return features_df

# let's call the function and create the users' inputs
user_inputs = user_input_features()

# show the inputs on the side panel
st.subheader("Users' Inputs")
st.write(user_inputs)

