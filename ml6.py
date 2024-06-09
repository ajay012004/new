
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

def main():
    st.title('Sentiment Analysis with Naive Bayes Classifier')
    
    # Default file path
    default_file_path = r"C:\Users\Asus\Documents\ml6\ml6\document (1).csv"
    
    # File uploader for user to upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read the CSV file from uploader
        msg = pd.read_csv(uploaded_file, names=['message', 'label'])
    else:
        # Read the CSV file from default path
        msg = pd.read_csv(default_file_path, names=['message', 'label'])
    
    st.write("Total Instances of Dataset:", msg.shape[0])
    
    # Map labels to numerical values
    msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})
    
    # Split data into train and test sets
    X = msg.message
    y = msg.labelnum
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Check for NaN or infinite values in ytrain and remove them
    if np.isnan(ytrain).any() or np.isinf(ytrain).any():
        mask = ~np.isnan(ytrain) & ~np.isinf(ytrain)
        Xtrain = Xtrain[mask]
        ytrain = ytrain[mask]
    
    # Vectorize the text data
    count_v = CountVectorizer()
    Xtrain_dm = count_v.fit_transform(Xtrain)
    Xtest_dm = count_v.transform(Xtest)
    
    # Convert to DataFrame for display
    df = pd.DataFrame(Xtrain_dm.toarray(), columns=count_v.get_feature_names_out())
    st.write("Sample of Vectorized Training Data:")
    st.write(df.head())
    
    # Train Naive Bayes classifier
    clf = MultinomialNB()
    clf.fit(Xtrain_dm, ytrain)
    pred = clf.predict(Xtest_dm)
    
    # Display sample predictions
    st.write('Sample Predictions:')
    for doc, p in zip(Xtest, pred):
        p = 'pos' if p == 1 else 'neg'
        st.write(f"{doc} -> {p}")
    
    # Display accuracy metrics
    st.write('Accuracy Metrics:')
    st.write('Accuracy:', accuracy_score(ytest, pred))
    st.write('Recall:', recall_score(ytest, pred))
    st.write('Precision:', precision_score(ytest, pred))
    st.write('Confusion Matrix:\n', confusion_matrix(ytest, pred))

if __name__ == '__main__':
    main()
