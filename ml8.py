
import streamlit as st
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to plot the figures
def plot_clusters(X, y, predY, y_cluster_gmm):
    colormap = np.array(['red', 'lime', 'black'])
    
    # Create a figure with three subplots
    fig, axs = plt.subplots(1, 3, figsize=(21, 7))
    
    # REAL PLOT
    axs[0].scatter(X.Petal_Length, X.Petal_Width, c=colormap[y.Targets], s=40)
    axs[0].set_title('Real')

    # K-PLOT
    axs[1].scatter(X.Petal_Length, X.Petal_Width, c=colormap[predY], s=40)
    axs[1].set_title('KMeans')

    # GMM PLOT
    axs[2].scatter(X.Petal_Length, X.Petal_Width, c=colormap[y_cluster_gmm], s=40)
    axs[2].set_title('GMM Classification')
    
    return fig

def main():
    st.title('Iris Dataset Clustering')

    dataset = load_iris()

    X = pd.DataFrame(dataset.data)
    X.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']
    y = pd.DataFrame(dataset.target)
    y.columns = ['Targets']

    model = KMeans(n_clusters=3)
    model.fit(X)
    predY = np.choose(model.labels_, [0, 1, 2]).astype(np.int64)

    scaler = preprocessing.StandardScaler()
    scaler.fit(X)
    xsa = scaler.transform(X)
    xs = pd.DataFrame(xsa, columns=X.columns)
    gmm = GaussianMixture(n_components=3)
    gmm.fit(xs)
    y_cluster_gmm = gmm.predict(xs)

    st.write("### Clustering of Iris Dataset")
    fig = plot_clusters(X, y, predY, y_cluster_gmm)
    st.pyplot(fig)

if __name__ == "__main__":
    main()
