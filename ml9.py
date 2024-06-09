import numpy as np
import pandas as pd
import streamlit as st

# Function to compute weights
def get_weights(X, x_query, tau):
    m = X.shape[0]
    W = np.eye(m)
    for i in range(m):
        xi = X[i]
        W[i, i] = np.exp(-np.dot((xi - x_query), (xi - x_query).T) / (2 * tau ** 2))
    return W

# Locally Weighted Regression function
def locally_weighted_regression(X, y, x_query, tau):
    X_ = np.c_[np.ones((X.shape[0], 1)), X]  # Add intercept term
    x_query_ = np.r_[1, x_query]  # Add intercept term
    W = get_weights(X_, x_query_, tau)
    theta = np.linalg.pinv(X_.T @ W @ X_) @ (X_.T @ W @ y)
    return np.dot(x_query_, theta)

# Streamlit App
st.title("CLOUD STROMS - Locally Weighted Regression")

# Upload dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head())

    # Select feature and target columns
    feature_col = st.selectbox("Select Feature Column", data.columns)
    target_col = st.selectbox("Select Target Column", data.columns)

    # Get X and y values
    X = data[[feature_col]].values
    y = data[target_col].values

    # Input value for bandwidth parameter tau
    tau = st.slider("Select Tau (Bandwidth)", 0.01, 1.0, 0.1)

    # Generate predictions
    X_test = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    y_pred = [locally_weighted_regression(X, y, x_query, tau) for x_query in X_test]

    # Prepare data for plotting
    plot_data = pd.DataFrame({
        feature_col: X_test.flatten(),
        'LWR Fit': np.array(y_pred).flatten()
    })

    # Plotting
    st.line_chart(plot_data.rename(columns={'LWR Fit': target_col}), x=feature_col, y=target_col)
    scatter_data = pd.DataFrame({
        feature_col: X.flatten(),
        target_col: y.flatten()
    })
    st.scatter_chart(scatter_data)

    st.write("### Combined Plot")
    combined_data = pd.concat([plot_data.set_index(feature_col), scatter_data.set_index(feature_col)], axis=1).reset_index()
    combined_data.columns = [feature_col, 'LWR Fit', 'Actual']  # Rename columns to avoid conflicts
    st.line_chart(combined_data, x=feature_col, y=['LWR Fit', 'Actual'])  # Corrected line
