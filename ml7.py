import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Define the smaller Iris dataset
iris_data = {
    "data": np.array([
        [5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2],
        [4.7, 3.2, 1.3, 0.2],
        [4.6, 3.1, 1.5, 0.2],
        [5.0, 3.6, 1.4, 0.2],
        [5.4, 3.9, 1.7, 0.4],
        [4.6, 3.4, 1.4, 0.3],
        [5.0, 3.4, 1.5, 0.2],
        [4.4, 2.9, 1.4, 0.2],
        [4.9, 3.1, 1.5, 0.1]
    ]),
    "target": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "target_names": np.array(['setosa'])
}

# Create a DataFrame
df = pd.DataFrame(data=iris_data['data'], columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris_data['data'], iris_data['target'], test_size=0.2, random_state=42)

# Initialize the KNN classifier
k = 3
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X_train, y_train)

# Streamlit app
st.title('Iris Flower Species Prediction')
st.sidebar.header('User Input Parameters')

# Function to get user inputs
def get_user_input():
    sepal_length = st.sidebar.slider('Sepal length', float(df['sepal_length'].min()), float(df['sepal_length'].max()), float(df['sepal_length'].mean()))
    sepal_width = st.sidebar.slider('Sepal width', float(df['sepal_width'].min()), float(df['sepal_width'].max()), float(df['sepal_width'].mean()))
    petal_length = st.sidebar.slider('Petal length', float(df['petal_length'].min()), float(df['petal_length'].max()), float(df['petal_length'].mean()))
    petal_width = st.sidebar.slider('Petal width', float(df['petal_width'].min()), float(df['petal_width'].max()), float(df['petal_width'].mean()))
    return np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Get user input
user_input = get_user_input()

# Predicting the output
prediction = knn_classifier.predict(user_input)
prediction_proba = knn_classifier.predict_proba(user_input)

# Displaying the user input and prediction
st.subheader('User Input Parameters')
st.write(df)
st.subheader('Prediction')
st.write(iris_data['target_names'][prediction][0])

# Display the probability of each class
st.subheader('Prediction Probability')
st.write(iris_data['target_names'][0], prediction_proba[0][0])
