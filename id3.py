
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.tree import export_text

# Define the Streamlit app
def main():
    st.title("ID3 Algorithm Demo")

    # Upload dataset
    st.subheader("Upload Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data)

        # Preprocessing: Convert categorical variables into numerical
        data = pd.get_dummies(data)

        # Model Training
        st.subheader("Training")
        X = data.drop(columns=['target'])
        y = data['target']
        clf = DecisionTreeClassifier(criterion="entropy")
        clf.fit(X, y)

        # Display decision tree rules
        st.subheader("Decision Tree Rules")
        rules = export_text(clf, feature_names=X.columns.tolist())
        st.text_area("Decision Tree Rules", rules, height=300)

        # Make a prediction
        st.subheader("Make Prediction")
        input_features = {}
        for feature in X.columns:
            input_features[feature] = st.number_input(f"Enter {feature}", value=0.0)

        if st.button("Predict"):
            instance = pd.DataFrame([input_features])
            prediction = clf.predict(instance)
            st.success(f"The predicted class is {prediction[0]}")

# Run the app
if __name__ == "__main__":
    main()
