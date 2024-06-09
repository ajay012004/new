
import streamlit as st
import pandas as pd
import csv

def find_s_algorithm(data):
    hypo = ['%', '%', '%', '%', '%', '%']
    positive_examples = []

    for row in data:
        if row[len(row) - 1].upper() == "YES":
            positive_examples.append(row)

    TotalExamples = len(positive_examples)
    d = len(positive_examples[0]) - 1

    hypo = positive_examples[0][:d]

    for i in range(1, TotalExamples):
        for k in range(d):
            if hypo[k] != positive_examples[i][k]:
                hypo[k] = '?'

    return hypo, positive_examples

# Streamlit app
st.title(" TITTLE: Cyber centurions")
st,subheater("TOPIC: Find-S Algorithm")
st.write("This app runs the Find-S algorithm to find the maximally specific hypothesis from given training data.")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    data = list(csv.reader(uploaded_file))
    
    st.write("### The given training examples are:")
    st.write(pd.DataFrame(data))

    hypo, positive_examples = find_s_algorithm(data)

    st.write("### The positive examples are:")
    st.write(pd.DataFrame(positive_examples))

    st.write("### The steps of the Find-S algorithm are:")
    st.write(hypo)

    st.write("### The maximally specific Find-S hypothesis for the given training examples is:")
    st.write(hypo)
else:
    st.write("Please upload a CSV file.")
