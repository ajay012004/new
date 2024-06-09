import streamlit as st
import numpy as np

# Function to generate synthetic data
def generate_data(num_samples, num_features):
    np.random.seed(0)
    X = np.random.randn(num_samples, num_features)
    return X

# Function for basic K-means clustering
def kmeans_clustering(X, num_clusters, max_iter=100):
    np.random.seed(0)
    centroids = X[np.random.choice(range(len(X)), num_clusters, replace=False)]
    for _ in range(max_iter):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(num_clusters)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, labels

# Function to plot clustering results
def plot_clusters(X, labels, centers, title, xlabel, ylabel):
    st.write(f"## {title}")
    if X.shape[1] == 2:
        # 2D plot
        st.write(f"### {xlabel} vs {ylabel}")
        for i in range(len(centers)):
            cluster_points = X[labels == i]
            st.write(f"Cluster {i + 1}")
            st.table(cluster_points)  # Display cluster points
            st.write(f"Center {i + 1}")
            st.write(centers[i])
    else:
        # If number of features is not 2, show basic data output
        st.write("Clustering result:")
        st.write("Cluster labels:")
        st.write(labels)

# Main Streamlit app
def main():
    st.title('CLOUD STROMS - K-NEAREST NEIGHBOUR ALGORITHM')

    # Sidebar to input parameters
    st.sidebar.header('Parameters')
    num_samples = st.sidebar.slider('Number of samples', min_value=50, max_value=200, value=100)
    num_features = st.sidebar.slider('Number of features', min_value=2, max_value=5, value=2)
    num_clusters = st.sidebar.slider('Number of clusters (K-means)', min_value=2, max_value=5, value=3)

    # Generate synthetic data
    X = generate_data(num_samples, num_features)

    # Perform KMeans clustering
    kmeans_centers, kmeans_labels = kmeans_clustering(X, num_clusters)

    # Display K-means results
    st.subheader('K-means Clustering')
    st.write('Centroids:')
    st.table(kmeans_centers)  # Display centroids

    plot_clusters(X, kmeans_labels, kmeans_centers, 'K-means Clustering', 'Feature 1', 'Feature 2')

if __name__ == '__main__':
    main()
