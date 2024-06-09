
import streamlit as st
import numpy as np

# Define the neural network class
class Neural_Network(object):
    def _init_(self):
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)

    def forward(self, X):
        self.z = np.dot(X, self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2)
        o = self.sigmoid(self.z3)
        return o 

    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))

    def sigmoidPrime(self, s):
        return s * (1 - s)
    
    def backward(self, X, y, o):
        self.o_error = y - o
        self.o_delta = self.o_error * self.sigmoidPrime(o)
        self.z2_error = self.o_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.o_delta)

    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)

# Initialize data
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)     # X = (hours sleeping, hours studying)
y = np.array(([92], [86], [89]), dtype=float)           # y = score on test

# Scale units
X = X / np.amax(X, axis=0)        # maximum of X array
y = y / 100                       # max test score is 100

# Initialize the neural network
NN = Neural_Network()

# Streamlit app
def main():
    st.title("Neural Network for Predicting Test Scores")

    if st.button("Train Neural Network"):
        # Train the neural network
        NN.train(X, y)

        # Forward pass to get the prediction
        predicted_output = NN.forward(X)

        # Calculate the loss
        loss = np.mean(np.square(y - predicted_output))

        # Display the results
        st.write("### Input Data (Hours Sleeping, Hours Studying):")
        st.write(X)
        st.write("### Actual Output (Test Scores):")
        st.write(y)
        st.write("### Predicted Output (Test Scores):")
        st.write(predicted_output)
        st.write("### Loss:")
        st.write(loss)

if _name_ == '_main_':
    main()
