import streamlit as st
import numpy as np

# Node class for the decision tree
class Node:
    def _init_(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature          # Index of feature to split on
        self.threshold = threshold      # Threshold value for numerical features
        self.left = left                # Left subtree
        self.right = right              # Right subtree
        self.value = value              # Majority class label for leaf node

# Function to calculate entropy
def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

# Function to find the best split
def find_best_split(X, y):
    best_entropy = float('inf')
    best_feature = None
    best_threshold = None
    
    # Iterate over each feature
    for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            left_indices = np.where(X[:, feature] <= threshold)[0]
            right_indices = np.where(X[:, feature] > threshold)[0]
            
            left_entropy = entropy(y[left_indices])
            right_entropy = entropy(y[right_indices])
            total_entropy = (len(left_indices) * left_entropy + len(right_indices) * right_entropy) / len(y)
            
            if total_entropy < best_entropy:
                best_entropy = total_entropy
                best_feature = feature
                best_threshold = threshold
    
    return best_feature, best_threshold

# Function to build the decision tree
def build_tree(X, y, max_depth):
    # Check for stopping criteria
    if max_depth == 0 or len(np.unique(y)) == 1:
        return Node(value=np.argmax(np.bincount(y)))
    
    best_feature, best_threshold = find_best_split(X, y)
    
    if best_feature is None:
        return Node(value=np.argmax(np.bincount(y)))
    
    # Split the data
    left_indices = np.where(X[:, best_feature] <= best_threshold)[0]
    right_indices = np.where(X[:, best_feature] > best_threshold)[0]
    
    left_subtree = build_tree(X[left_indices], y[left_indices], max_depth - 1)
    right_subtree = build_tree(X[right_indices], y[right_indices], max_depth - 1)
    
    return Node(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

# Function to predict using the decision tree
def predict(tree, X):
    if tree.value is not None:
        return tree.value
    
    if X[tree.feature] <= tree.threshold:
        return predict(tree.left, X)
    else:
        return predict(tree.right, X)

# Main Streamlit app
def main():
    st.title(' Cybe Centurions ID3 Algorithm Demo')
    
    # Dummy dataset
    X = np.array([[2, 3], [1, 2], [3, 1], [4, 2], [3, 5], [6, 5]])
    y = np.array([0, 0, 1, 1, 0, 1])
    
    # Build the decision tree
    max_depth = st.sidebar.slider('Max Depth', min_value=1, max_value=10, value=2)
    tree = build_tree(X, y, max_depth)
    
    # Test the decision tree
    test_point = st.sidebar.text_input('Test Point (comma-separated)', '3,4')
    test_point = np.array(list(map(float, test_point.split(','))))
    
    prediction = predict(tree, test_point)
    st.write(f'Prediction for test point {test_point}: Class {prediction}')

if _name_ == '_main_':
    main()
