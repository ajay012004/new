Updates to keyboard shortcuts … On Thursday, August 1, 2024, Drive keyboard shortcuts will be updated to give you first-letters navigation.Learn more
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

# Streamlit app
st.title("Document Classifier")
st.write("This app classifies text documents using a Multinomial Naive Bayes classifier.")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    # Load data
    msg = pd.read_csv(uploaded_file, names=['message', 'label'])
    st.write("Total Instances of Dataset: ", msg.shape[0])
    st.write("### The first 5 values of data")
    st.write(msg.head())

    # Map labels to numbers
    msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})

    X = msg.message
    y = msg.labelnum

    # Split the data
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20, random_state=42)

    # Vectorize the data
    count_v = CountVectorizer()
    Xtrain_dm = count_v.fit_transform(Xtrain)
    Xtest_dm = count_v.transform(Xtest)

    df = pd.DataFrame(Xtrain_dm.toarray(), columns=count_v.get_feature_names_out())
    st.write("### Feature Names and Sample Data")
    st.write(df.head())

    # Train the classifier
    clf = MultinomialNB()
    clf.fit(Xtrain_dm, ytrain)
    pred = clf.predict(Xtest_dm)

    # Display predictions
    st.write("### Predictions")
    for doc, p in zip(Xtest, pred):
        p_label = 'pos' if p == 1 else 'neg'
        st.write(f"{doc} -> {p_label}")

    # Calculate and display accuracy metrics
    accuracy = accuracy_score(ytest, pred)
    recall = recall_score(ytest, pred)
    precision = precision_score(ytest, pred)
    confusion = confusion_matrix(ytest, pred)

    st.write("### Accuracy Metrics")
    st.write(f"Accuracy: {accuracy}")
    st.write(f"Recall: {recall}")
    st.write(f"Precision: {precision}")
    st.write("Confusion Matrix:")
    st.write(confusion)
else:
    st.write("Please upload a CSV file.")
