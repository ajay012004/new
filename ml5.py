import streamlit as st
import pandas as pd
import numpy as np

# Sample dataset
data = [
    ("I love this sandwich", "pos"),
    ("This is an amazing place", "pos"),
    ("I feel very good about these beers", "pos"),
    ("This is my best work", "pos"),
    ("What an awesome view", "pos"),
    ("I do not like this restaurant", "neg"),
    ("I am tired of this stuff", "neg"),
    ("I can't deal with this", "neg"),
    ("He is my sworn enemy", "neg"),
    ("My boss is horrible", "neg"),
    ("This is an awesome place", "pos"),
    ("I do not like the taste of this juice", "neg"),
    ("I love to dance", "pos"),
    ("I am sick and tired of this place", "neg"),
    ("What a great holiday", "pos"),
    ("That is a bad locality to stay", "neg"),
    ("We will have good fun tomorrow", "pos"),
    ("I went to my enemy's house today", "neg")
]

# Preprocess the data
def preprocess_data(data):
    df = pd.DataFrame(data, columns=['text', 'label'])
    df['text'] = df['text'].str.lower().str.split()
    return df

# Train Naive Bayes model
def train_naive_bayes(df):
    num_pos = (df['label'] == 'pos').sum()
    num_neg = len(df) - num_pos
    total_docs = len(df)
    
    p_pos = num_pos / total_docs
    p_neg = num_neg / total_docs
    
    p_word_given_pos = {}
    p_word_given_neg = {}
    
    for index, row in df.iterrows():
        for word in row['text']:
            if row['label'] == 'pos':
                p_word_given_pos[word] = p_word_given_pos.get(word, 0) + 1
            else:
                p_word_given_neg[word] = p_word_given_neg.get(word, 0) + 1
    
    vocab_size = len(set(df['text'].sum()))
    p_word_given_pos_smooth = {word: (count + 1) / (num_pos + vocab_size) for word, count in p_word_given_pos.items()}
    p_word_given_neg_smooth = {word: (count + 1) / (num_neg + vocab_size) for word, count in p_word_given_neg.items()}
    
    return p_pos, p_neg, p_word_given_pos_smooth, p_word_given_neg_smooth

# Classify a document
def classify_document(document, p_pos, p_neg, p_word_given_pos, p_word_given_neg):
    p_pos_given_doc = p_pos
    p_neg_given_doc = p_neg
    
    for word in document:
        p_pos_given_doc *= p_word_given_pos.get(word, 1 / (len(p_word_given_pos) + 1))
        p_neg_given_doc *= p_word_given_neg.get(word, 1 / (len(p_word_given_neg) + 1))
    
    return 'pos' if p_pos_given_doc > p_neg_given_doc else 'neg'

# Evaluate the model
def evaluate_model(df, p_pos, p_neg, p_word_given_pos, p_word_given_neg):
    y_true = df['label']
    y_pred = [classify_document(doc, p_pos, p_neg, p_word_given_pos, p_word_given_neg) for doc in df['text']]
    
    accuracy = np.mean([y_true[i] == y_pred[i] for i in range(len(y_true))])
    precision_pos = sum((np.array(y_pred) == 'pos') & (np.array(y_true) == 'pos')) / sum(np.array(y_pred) == 'pos')
    precision_neg = sum((np.array(y_pred) == 'neg') & (np.array(y_true) == 'neg')) / sum(np.array(y_pred) == 'neg')
    recall_pos = sum((np.array(y_pred) == 'pos') & (np.array(y_true) == 'pos')) / sum(np.array(y_true) == 'pos')
    recall_neg = sum((np.array(y_pred) == 'neg') & (np.array(y_true) == 'neg')) / sum(np.array(y_true) == 'neg')
    
    return accuracy, precision_pos, precision_neg, recall_pos, recall_neg

# Streamlit app
def main():
    st.title("Naive Bayes Document Classification")

    # Display dataset
    df = preprocess_data(data)
    st.write("Dataset:")
    st.write(df)

    # Train the model
    if st.button("Train Model"):
        p_pos, p_neg, p_word_given_pos, p_word_given_neg = train_naive_bayes(df)
        st.success("Model trained successfully!")

        # Evaluate the model
        accuracy, precision_pos, precision_neg, recall_pos, recall_neg = evaluate_model(df, p_pos, p_neg, p_word_given_pos, p_word_given_neg)

        st.write("Model Evaluation Results:")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"Precision (Positive): {precision_pos:.2f}")
        st.write(f"Precision (Negative): {precision_neg:.2f}")
        st.write(f"Recall (Positive): {recall_pos:.2f}")
        st.write(f"Recall (Negative): {recall_neg:.2f}")

if __name__ == "__main__":
    main()
