import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define the Streamlit app
def main():
    st.title("ID3 Algorithm Demo")

    # Upload dataset
    st.subheader("Upload Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data)

        # Choose target column
        target_column = st.selectbox("Select the target column", data.columns)

        # Split data into features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create decision tree model
        clf = DecisionTreeClassifier(criterion="entropy")
        clf.fit(X_train, y_train)

        # Display decision tree rules
        st.subheader("Decision Tree Rules")
        rules = export_text(clf, feature_names=X.columns.tolist())
        st.text_area("Decision Tree Rules", rules, height=300)

        # Make predictions
        st.subheader("Make Predictions")
        new_data = {}
        for feature in X.columns:
            new_data[feature] = st.number_input(f"Enter {feature}", value=0.0)

        if st.button("Predict"):
            instance = pd.DataFrame([new_data])
            prediction = clf.predict(instance)
            st.success(f"The predicted class is {prediction[0]}")

        # Evaluate model
        st.subheader("Model Evaluation")
        accuracy = accuracy_score(y_test, clf.predict(X_test))
        st.write(f"Accuracy: {accuracy:.2f}")

# Run the app
if __name__ == "__main__":
    main()
