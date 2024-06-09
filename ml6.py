import streamlit as st
import pandas as pd
import numpy as np

# Streamlit app title
st.title(" CYBER CENTURIONS-Heart Disease Prediction using Logistic Regression")

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load data from CSV
    heart_disease = pd.read_csv(uploaded_file)
    st.write("The first 5 values of the dataset:")
    st.write(heart_disease.head())

    # Preprocess the data
    label_encoders = {}
    categorical_columns = ['age', 'Gender', 'Family', 'diet', 'Lifestyle', 'cholestrol']
    
    # Define standard categorical values
    categorical_values = {
        'age': ['SuperSeniorCitizen', 'SeniorCitizen', 'MiddleAged', 'Youth', 'Teen'],
        'Gender': ['Male', 'Female'],
        'Family': ['Yes', 'No'],
        'diet': ['High', 'Medium'],
        'Lifestyle': ['Athlete', 'Active', 'Moderate', 'Sedentary'],
        'cholestrol': ['High', 'BorderLine', 'Normal']
    }
    
    for column in categorical_columns:
        label_encoders[column] = {val: idx for idx, val in enumerate(categorical_values[column])}
        heart_disease[column] = heart_disease[column].map(label_encoders[column])

    # Manually split the data into training and testing sets
    np.random.seed(42)
    mask = np.random.rand(len(heart_disease)) < 0.8
    train_data = heart_disease[mask]
    test_data = heart_disease[~mask]

    X_train = train_data.drop('heartdisease', axis=1).values
    y_train = train_data['heartdisease'].values
    X_test = test_data.drop('heartdisease', axis=1).values
    y_test = test_data['heartdisease'].values

    # Implement logistic regression
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def initialize_parameters(dim):
        w = np.zeros((dim, 1))
        b = 0
        return w, b

    def propagate(w, b, X, Y):
        m = X.shape[0]
        
        # Forward propagation
        A = sigmoid(np.dot(X, w) + b)
        cost = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        
        # Backward propagation
        dw = 1/m * np.dot(X.T, (A - Y))
        db = 1/m * np.sum(A - Y)
        
        cost = np.squeeze(cost)
        grads = {"dw": dw, "db": db}
        
        return grads, cost

    def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
        costs = []
        
        for i in range(num_iterations):
            grads, cost = propagate(w, b, X, Y)
            
            # Retrieve derivatives from grads
            dw = grads["dw"]
            db = grads["db"]
            
            # Update rule
            w = w - learning_rate * dw
            b = b - learning_rate * db
            
            # Record the costs
            if i % 100 == 0:
                costs.append(cost)
            
            # Print the cost every 100 training iterations
            if print_cost and i % 100 == 0:
                st.write(f"Cost after iteration {i}: {cost}")
        
        params = {"w": w, "b": b}
        grads = {"dw": dw, "db": db}
        
        return params, grads, costs

    def predict(w, b, X):
        m = X.shape[0]
        Y_prediction = np.zeros((m, 1))
        w = w.reshape(X.shape[1], 1)
        
        # Compute vector "A" predicting the probabilities of a heart disease
        A = sigmoid(np.dot(X, w) + b)
        
        for i in range(A.shape[0]):
            # Convert probabilities A[0,i] to actual predictions p[0,i]
            Y_prediction[i, 0] = 1 if A[i, 0] > 0.5 else 0
        
        return Y_prediction

    def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
        w, b = initialize_parameters(X_train.shape[1])
        
        # Gradient descent
        parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
        
        # Retrieve parameters w and b from dictionary "parameters"
        w = parameters["w"]
        b = parameters["b"]
        
        # Predict test/train set examples
        Y_prediction_test = predict(w, b, X_test)
        Y_prediction_train = predict(w, b, X_train)
        
        # Print train/test Errors
        st.write("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        st.write("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
        
        d = {"costs": costs,
             "Y_prediction_test": Y_prediction_test, 
             "Y_prediction_train" : Y_prediction_train, 
             "w" : w, 
             "b" : b,
             "learning_rate" : learning_rate,
             "num_iterations": num_iterations}
        
        return d

    # Train the model
    d = model(X_train, y_train.reshape(-1, 1), X_test, y_test.reshape(-1, 1), num_iterations=2000, learning_rate=0.005, print_cost=True)

    # User inputs
    st.write('### Enter the following details:')
    
    age = st.selectbox('Age', list(categorical_values['age']))
    gender = st.selectbox('Gender', list(categorical_values['Gender']))
    family_history = st.selectbox('Family History', list(categorical_values['Family']))
    diet = st.selectbox('Diet', list(categorical_values['diet']))
    lifestyle = st.selectbox('Lifestyle', list(categorical_values['Lifestyle']))
    cholestrol = st.selectbox('Cholestrol', list(categorical_values['cholestrol']))

    # Convert user inputs to appropriate format
    user_input = np.array([label_encoders['age'][age], label_encoders['Gender'][gender],
                           label_encoders['Family'][family_history], label_encoders['diet'][diet],
                           label_encoders['Lifestyle'][lifestyle], label_encoders['cholestrol'][cholestrol]]).reshape(1, -1)

    # Predict heart disease
    if st.button("Predict"):
        prediction = predict(d["w"], d["b"], user_input)
        st.write("Prediction for Heart Disease:")
        st.write('Yes' if prediction[0, 0] == 1 else 'No')
else:
    st.write("Please upload a CSV file to proceed.")
