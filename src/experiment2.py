import numpy as np
import matplotlib.pyplot as plt
import os
from DensityPeaksClustering import DensityPeaksClustering
from ClusteringAggregation import ClusteringAggregation
#from DPCV import DPCV
from DPCVnew import DPCV
from FLAME import FLAME
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, davies_bouldin_score
import pandas as pd
from scipy.stats import mode

def plot_decision_graph(rho, delta, center_indices, cluster_colors, output_path):
    """
    Plot the decision graph (rho-delta plot) with cluster centers highlighted.
    
    Parameters:
    -----------
    rho : array-like
        Local density values
    delta : array-like
        Minimum distance to points of higher density
    center_indices : array-like
        Indices of cluster centers
    cluster_colors : list
        List of colors for each cluster center
    output_path : str
        Path where to save the plot
    """
    plt.figure(figsize=(5, 4))
    
    # Plot all points
    plt.scatter(rho, delta, c='k', marker='o', s=25)
    
    # Highlight centers with different colors
    for i, center_idx in enumerate(center_indices):
        plt.scatter(rho[center_idx], delta[center_idx], c=[cluster_colors[i]], marker='o', s=80)
    
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$\delta$')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_aggregation():
    """Run DensityPeaksClustering and Clustering Aggregation on the Aggregation dataset."""
    # Read data from Aggregation.txt
    data_file = "../data/experiment2/Aggregation.txt"
    data = np.loadtxt(data_file, delimiter='\t')
    
    # Extract features (X, Y) and true labels
    X = data[:, :2]  # First two columns are X, Y coordinates
    true_labels = data[:, 2].astype(int)  # Third column contains the true labels
    
    # Apply DensityPeaksClustering
    dpc = DensityPeaksClustering(n_clusters=7, density_estimator='gaussian')
    predicted_labels_dpc = dpc.fit_predict(X)
    
    # Apply Clustering Aggregation
    ca = ClusteringAggregation(n_clusters=7)
    aggregated_labels = ca.fit_predict(X)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=7, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)
    
    # Compute clustering metrics
    metrics = {
        'Algorithm': ['DensityPeaks', 'ClusteringAggregation', 'K-means'],
        'ARI': [
            adjusted_rand_score(true_labels, predicted_labels_dpc),
            adjusted_rand_score(true_labels, aggregated_labels),
            adjusted_rand_score(true_labels, kmeans_labels)
        ],
        'NMI': [
            normalized_mutual_info_score(true_labels, predicted_labels_dpc),
            normalized_mutual_info_score(true_labels, aggregated_labels),
            normalized_mutual_info_score(true_labels, kmeans_labels)
        ],
        'Silhouette': [
            silhouette_score(X, predicted_labels_dpc),
            silhouette_score(X, aggregated_labels),
            silhouette_score(X, kmeans_labels)
        ],
        'DBI': [
            davies_bouldin_score(X, predicted_labels_dpc),
            davies_bouldin_score(X, aggregated_labels),
            davies_bouldin_score(X, kmeans_labels)
        ]
    }
    
    # Create DataFrame and save to CSV
    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.round(4)  # Round to 4 decimal places
    results_dir = "../results/experiment2"
    os.makedirs(results_dir, exist_ok=True)
    metrics_df.to_csv(f"{results_dir}/Aggregation_metrics.csv", index=False)
    print(f"\nClustering metrics for Aggregation dataset:")
    print(metrics_df.to_string(index=False))
    
    # Define color palette with hex values
    color_hex = {
        'yellow': '#F2EB17',
        'green': '#69BD45',
        'red': '#EE1F23', 
        'blue': '#4077BC',
        'orange': '#F57F21',
        'darkblue': '#3853A3',
        'black': '#040503'
    }
    
    # Define color ordering for original clusters
    original_colors = ['blue', 'darkblue', 'orange', 'green', 'red', 'black', 'yellow']
    
    # Define color ordering for DensityPeaks clusters
    dpc_colors = ['red', 'yellow', 'orange', 'darkblue', 'black', 'blue', 'green']
    
    # Define color ordering for Clustering Aggregation
    agg_colors = ['red', 'yellow', 'darkblue', 'orange', 'black', 'green', 'blue']
    
    # Define color ordering for K-means
    kmeans_colors = ['black', 'darkblue', 'orange', 'green', 'yellow', 'blue', 'red']
    
    # Plot decision graph for DensityPeaksClustering
    dpc_hex_colors = [color_hex[color] for color in dpc_colors]
    plot_decision_graph(
        dpc.rho_,
        dpc.delta_,
        dpc.centers_,
        dpc_hex_colors,
        os.path.join(results_dir, 'Aggregation_decision_graph.png')
    )
    
    # Plot results
    fig = plt.figure(figsize=(12, 10))
    plt.subplots_adjust(bottom=0.1)  # Add more space at the bottom for captions
    
    # Original labels
    ax1 = plt.subplot(2, 2, 1)
    for label in np.unique(true_labels):
        mask = true_labels == label
        color_name = original_colors[label % len(original_colors)]
        plt.scatter(X[mask, 0], X[mask, 1], c=color_hex[color_name], s=30, alpha=0.8)
    plt.title("Original Labels")
    plt.axis('off')
    # Add border
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('black')
    # Add caption
    plt.text(0.5, -0.1, '(a)', transform=ax1.transAxes, fontsize=12, ha='center')
    
    # DensityPeaksClustering labels
    ax2 = plt.subplot(2, 2, 2)
    for label in np.unique(predicted_labels_dpc):
        mask = predicted_labels_dpc == label
        color_name = dpc_colors[label % len(dpc_colors)]
        plt.scatter(X[mask, 0], X[mask, 1], c=color_hex[color_name], s=30, alpha=0.8)
    plt.title("DensityPeaksClustering Prediction")
    plt.axis('off')
    # Add border
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('black')
    # Add caption
    plt.text(0.5, -0.1, '(b)', transform=ax2.transAxes, fontsize=12, ha='center')
    
    # Clustering Aggregation labels
    ax3 = plt.subplot(2, 2, 3)
    for label in np.unique(aggregated_labels):
        mask = aggregated_labels == label
        color_name = agg_colors[label % len(agg_colors)]
        plt.scatter(X[mask, 0], X[mask, 1], c=color_hex[color_name], s=30, alpha=0.8)
    plt.title("Clustering Aggregation Prediction")
    plt.axis('off')
    # Add border
    for spine in ax3.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('black')
    # Add caption
    plt.text(0.5, -0.1, '(c)', transform=ax3.transAxes, fontsize=12, ha='center')
    
    # K-means clustering labels
    ax4 = plt.subplot(2, 2, 4)
    for label in np.unique(kmeans_labels):
        mask = kmeans_labels == label
        color_name = kmeans_colors[label % len(kmeans_colors)]
        plt.scatter(X[mask, 0], X[mask, 1], c=color_hex[color_name], s=30, alpha=0.8)
    plt.title("K-means Clustering")
    plt.axis('off')
    # Add border
    for spine in ax4.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('black')
    # Add caption
    plt.text(0.5, -0.1, '(d)', transform=ax4.transAxes, fontsize=12, ha='center')
    
    plt.tight_layout()
    # Adjust the layout to make room for captions
    plt.subplots_adjust(hspace=0.3)
    
    # Save the plot
    plt.savefig(f"{results_dir}/Aggregation.png", dpi=300, bbox_inches='tight')
    print(f"Plot saved to {results_dir}/Aggregation.png")
    plt.close()

def run_flame():
    """Run DensityPeaksClustering and FLAME on the Flame dataset."""
    # Read data from flame.txt
    data_file = "../data/experiment2/flame.txt"
    data = np.loadtxt(data_file, delimiter='\t')
    
    # Extract features (X, Y) and true labels
    X = data[:, :2]
    true_labels = data[:, 2].astype(int)
    
    # Apply DensityPeaksClustering
    dpc = DensityPeaksClustering(n_clusters=2, percent=3.0, density_estimator='gaussian')
    dpc_labels = dpc.fit_predict(X)
    
    # Apply FLAME clustering
    flame = FLAME(metric="minkowski", cluster_neighbors=20, iteration_neighbors=20, 
                       max_iter=2000, eps=1.6e-05, thd=-2.2, verbose=1)
    flame_labels = flame.fit_predict(X)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)
    
    # Compute clustering metrics
    metrics = {
        'Algorithm': ['DensityPeaks', 'FLAME', 'K-means'],
        'ARI': [
            adjusted_rand_score(true_labels, dpc_labels),
            adjusted_rand_score(true_labels, flame_labels),
            adjusted_rand_score(true_labels, kmeans_labels)
        ],
        'NMI': [
            normalized_mutual_info_score(true_labels, dpc_labels),
            normalized_mutual_info_score(true_labels, flame_labels),
            normalized_mutual_info_score(true_labels, kmeans_labels)
        ],
        'Silhouette': [
            silhouette_score(X, dpc_labels),
            silhouette_score(X, flame_labels),
            silhouette_score(X, kmeans_labels)
        ],
        'DBI': [
            davies_bouldin_score(X, dpc_labels),
            davies_bouldin_score(X, flame_labels),
            davies_bouldin_score(X, kmeans_labels)
        ]
    }
    
    # Create DataFrame and save to CSV
    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.round(4)  # Round to 4 decimal places
    results_dir = "../results/experiment2"
    os.makedirs(results_dir, exist_ok=True)
    metrics_df.to_csv(f"{results_dir}/Flame_metrics.csv", index=False)
    print(f"\nClustering metrics for Flame dataset:")
    print(metrics_df.to_string(index=False))
    
    color_hex = {
        'blue': '#3853A3',
        'red': '#EE1F23'
    }
    
    # Plot decision graph for DensityPeaksClustering
    dpc_colors = ['red', 'blue']  # Colors for the two clusters
    dpc_hex_colors = [color_hex[color] for color in dpc_colors]
    plot_decision_graph(
        dpc.rho_,
        dpc.delta_,
        dpc.centers_,
        dpc_hex_colors,
        os.path.join(results_dir, 'Flame_decision_graph.png')
    )
    
    # Plot results
    fig = plt.figure(figsize=(12, 10))
    plt.subplots_adjust(bottom=0.1)  # Add more space at the bottom for captions
    
    # Original labels
    ax1 = plt.subplot(2, 2, 1)
    for label in np.unique(true_labels):
        mask = true_labels == label
        color_name = 'blue' if label == 1 else 'red'
        plt.scatter(X[mask, 0], X[mask, 1], c=color_hex[color_name], s=30, alpha=0.8)
    plt.title("Original Labels")
    plt.axis('off')
    # Add border
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('black')
    # Add caption
    plt.text(0.5, -0.1, '(a)', transform=ax1.transAxes, fontsize=12, ha='center')
    
    # DPC labels
    ax2 = plt.subplot(2, 2, 2)
    for label in np.unique(dpc_labels):
        mask = dpc_labels == label
        color_name = 'red' if label == 0 else 'blue'
        plt.scatter(X[mask, 0], X[mask, 1], c=color_hex[color_name], s=30, alpha=0.8)
    plt.title("DensityPeaksClustering Prediction")
    plt.axis('off')
    # Add border
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('black')
    # Add caption
    plt.text(0.5, -0.1, '(b)', transform=ax2.transAxes, fontsize=12, ha='center')
    
    # FLAME labels
    ax3 = plt.subplot(2, 2, 3)
    for label in np.unique(flame_labels):
        mask = flame_labels == label
        if label == -1:
            plt.scatter(X[mask, 0], X[mask, 1], c='black', marker='x', s=50, label='Outliers')
        else:
            color_name = 'red' if label == 0 else 'blue'
            plt.scatter(X[mask, 0], X[mask, 1], c=color_hex[color_name], s=30, alpha=0.8)
    
    plt.title("FLAME Clustering Prediction")
    plt.axis('off')
    # Add border
    for spine in ax3.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('black')
    # Add caption
    plt.text(0.5, -0.1, '(c)', transform=ax3.transAxes, fontsize=12, ha='center')
    
    # K-means labels
    ax4 = plt.subplot(2, 2, 4)
    for label in np.unique(kmeans_labels):
        mask = kmeans_labels == label
        color_name = 'red' if label == 0 else 'blue' 
        plt.scatter(X[mask, 0], X[mask, 1], c=color_hex[color_name], s=30, alpha=0.8)
    plt.title("K-means Clustering")
    plt.axis('off')
    # Add border
    for spine in ax4.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('black')
    # Add caption
    plt.text(0.5, -0.1, '(d)', transform=ax4.transAxes, fontsize=12, ha='center')
    
    plt.tight_layout()
    # Adjust the layout to make room for captions
    plt.subplots_adjust(hspace=0.3)
    
    # Save the plot
    plt.savefig(f"{results_dir}/Flame.png", dpi=300, bbox_inches='tight')
    print(f"Plot saved to {results_dir}/Flame.png")
    plt.close()

def run_spiral():
    """Run DensityPeaksClustering, Spectral Clustering and K-means on the Spiral dataset."""
    # Create results directory
    results_dir = "../results/experiment2"
    os.makedirs(results_dir, exist_ok=True)
    
    # Read data from spiral.txt
    data_file = "../data/experiment2/spiral.txt"
    data = np.loadtxt(data_file, delimiter='\t')
    
    # Extract features (X, Y) and true labels
    X = data[:, :2]
    true_labels = data[:, 2].astype(int)
    
    # Define color palette with hex values
    color_hex = {
        'green': '#69BD45',
        'red': '#EE1F23',
        'blue': '#4077BC'
    }
    
    # Define color mappings for each algorithm to ensure consistency
    original_colors = ['blue', 'green', 'red']  # Original labels are 1-indexed
    dpc_colors = ['red', 'green', 'blue']  # DPC labels
    spectral_colors = ['red', 'green', 'blue']  # Spectral labels
    kmeans_colors = ['red', 'green', 'blue']  # K-means labels
    
    # Apply DensityPeaksClustering
    dpc = DensityPeaksClustering(n_clusters=3, percent=3.0, density_estimator='gaussian')
    predicted_labels_dpc = dpc.fit_predict(X)
    
    # Plot decision graph for DensityPeaksClustering
    plot_decision_graph(
        dpc.rho_,
        dpc.delta_,
        dpc.centers_,
        [color_hex[color] for color in dpc_colors],
        os.path.join(results_dir, 'Spiral_decision_graph.png')
    )
    
    # Apply Spectral Clustering with optimized parameters
    spectral = SpectralClustering(n_clusters=3, gamma=0.34, n_neighbors=5, assign_labels='kmeans', 
                                 random_state=42, affinity='nearest_neighbors', eigen_tol=1.59)
    predicted_labels_spectral = spectral.fit_predict(X)
    
    # Apply K-means
    kmeans = KMeans(n_clusters=3, random_state=42)
    predicted_labels_kmeans = kmeans.fit_predict(X)
    
    # Compute clustering metrics
    metrics = {
        'Algorithm': ['DensityPeaks', 'Spectral', 'K-means'],
        'ARI': [
            adjusted_rand_score(true_labels, predicted_labels_dpc),
            adjusted_rand_score(true_labels, predicted_labels_spectral),
            adjusted_rand_score(true_labels, predicted_labels_kmeans)
        ],
        'NMI': [
            normalized_mutual_info_score(true_labels, predicted_labels_dpc),
            normalized_mutual_info_score(true_labels, predicted_labels_spectral),
            normalized_mutual_info_score(true_labels, predicted_labels_kmeans)
        ],
        'Silhouette': [
            silhouette_score(X, predicted_labels_dpc),
            silhouette_score(X, predicted_labels_spectral),
            silhouette_score(X, predicted_labels_kmeans)
        ],
        'DBI': [
            davies_bouldin_score(X, predicted_labels_dpc),
            davies_bouldin_score(X, predicted_labels_spectral),
            davies_bouldin_score(X, predicted_labels_kmeans)
        ]
    }
    
    # Create DataFrame and save to CSV
    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.round(4)  # Round to 4 decimal places
    metrics_df.to_csv(f"{results_dir}/Spiral_metrics.csv", index=False)
    print(f"\nClustering metrics for Spiral dataset:")
    print(metrics_df.to_string(index=False))
    
    # Plot results
    fig = plt.figure(figsize=(12, 10))
    plt.subplots_adjust(bottom=0.1)  # Add more space at the bottom for captions
    
    # Original labels
    ax1 = plt.subplot(2, 2, 1)
    for i, label in enumerate(np.unique(true_labels)):
        mask = true_labels == label
        color_name = original_colors[i]  # Use consistent color mapping
        plt.scatter(X[mask, 0], X[mask, 1], c=color_hex[color_name], s=30, alpha=0.8)
    plt.title("Original Labels")
    plt.axis('off')
    # Add border
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('black')
    # Add caption
    plt.text(0.5, -0.1, '(a)', transform=ax1.transAxes, fontsize=12, ha='center')
    
    # DPC labels
    ax2 = plt.subplot(2, 2, 2)
    for i, label in enumerate(np.unique(predicted_labels_dpc)):
        mask = predicted_labels_dpc == label
        color_name = dpc_colors[i]  # Use consistent color mapping
        plt.scatter(X[mask, 0], X[mask, 1], c=color_hex[color_name], s=30, alpha=0.8)
    plt.title("DensityPeaksClustering Prediction")
    plt.axis('off')
    # Add border
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('black')
    # Add caption
    plt.text(0.5, -0.1, '(b)', transform=ax2.transAxes, fontsize=12, ha='center')
    
    # Spectral Clustering labels
    ax3 = plt.subplot(2, 2, 3)
    for i, label in enumerate(np.unique(predicted_labels_spectral)):
        mask = predicted_labels_spectral == label
        color_name = spectral_colors[i]  # Use consistent color mapping
        plt.scatter(X[mask, 0], X[mask, 1], c=color_hex[color_name], s=30, alpha=0.8)
    plt.title("Spectral Clustering Prediction")
    plt.axis('off')
    # Add border
    for spine in ax3.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('black')
    # Add caption
    plt.text(0.5, -0.1, '(c)', transform=ax3.transAxes, fontsize=12, ha='center')
    
    # K-means labels
    ax4 = plt.subplot(2, 2, 4)
    for i, label in enumerate(np.unique(predicted_labels_kmeans)):
        mask = predicted_labels_kmeans == label
        color_name = kmeans_colors[i]  # Use consistent color mapping
        plt.scatter(X[mask, 0], X[mask, 1], c=color_hex[color_name], s=30, alpha=0.8)
    plt.title("K-means Clustering")
    plt.axis('off')
    # Add border
    for spine in ax4.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('black')
    # Add caption
    plt.text(0.5, -0.1, '(d)', transform=ax4.transAxes, fontsize=12, ha='center')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    
    # Save the plot
    plt.savefig(f"{results_dir}/Spiral.png", dpi=300, bbox_inches='tight')
    print(f"Plot saved to {results_dir}/Spiral.png")
    plt.close()

def get_cluster_color_mapping(predicted_labels, true_labels, canonical_color_names, color_hex):
    """
    Determines the color for each predicted cluster based on the most frequent true label within it.

    Parameters:
    -----------
    predicted_labels : array-like
        Labels predicted by the clustering algorithm.
    true_labels : array-like
        Ground truth labels.
    canonical_color_names : list
        List of color names used for the original labels.
    color_hex : dict
        Dictionary mapping color names to hex values.

    Returns:
    --------
    dict
        A dictionary mapping each unique predicted label to its assigned hex color.
    """
    color_map = {}
    unique_predicted = np.unique(predicted_labels)
    unique_true = np.unique(true_labels)
    
    # Create a map from true label to its canonical color name
    true_label_to_color_name = {label: canonical_color_names[i % len(canonical_color_names)] 
                                for i, label in enumerate(unique_true)}

    for pred_label in unique_predicted:
        mask = predicted_labels == pred_label
        # Find the most frequent true label in this predicted cluster
        most_frequent_true_label = mode(true_labels[mask], keepdims=True)[0][0]
        
        # Get the canonical color name for that true label
        color_name = true_label_to_color_name.get(most_frequent_true_label, 'black') # Default to black if not found
        
        # Map the predicted label to the hex color
        color_map[pred_label] = color_hex[color_name]
        
    return color_map

def run_s3():
    """Run DensityPeaksClustering, DPCV, and K-means on the S3 dataset."""
    # Create results directory
    results_dir = "../results/experiment2"
    os.makedirs(results_dir, exist_ok=True)
    
    # Read data from s3.txt (X and Y coordinates)
    data_file = "../data/experiment2/s3.txt"
    X = np.loadtxt(data_file)
    
    # Read labels from s3-label.txt
    label_file = "../data/experiment2/s3-label.txt"
    with open(label_file, 'r') as f:
        lines = f.readlines()
    true_labels = np.array([int(line.strip()) for line in lines if line.strip()])
    
    n_clusters_s3 = 15

    # Define color palette with hex values
    color_hex = {
        'blue': '#3853A3',
        'darkblue': '#2A318D',
        'darkestblue': '#272873',
        'green': '#69BD45',
        'darkgreen': '#0D803F',
        'orange': '#F57F21',
        'red': '#EE1F23',
        'yellow': '#F2EB17',
        'pink': '#B9519F',
        'purple': '#873A88',
        'brown': '#7F1315',
        'lightbrown': '#7F8133',
        'black': '#040503',
        'turquoise': '#0C8081',
        'cyan': '#00A0B0'
    }
    
    # Define canonical color ordering - used for original labels and mapping predictions
    canonical_color_names = ['black', 'darkblue', 'green', 'brown', 'blue', 
                             'orange', 'darkestblue', 'pink', 'turquoise', 'cyan', 
                             'red', 'lightbrown', 'darkgreen', 'purple', 'yellow']
    
    original_colors = canonical_color_names # Use canonical order for original labels
    dpcv_colors = ['black', 'darkblue', 'green', 'brown', 'blue', 
                             'orange', 'darkestblue', 'pink', 'turquoise', 'cyan', 
                             'red', 'lightbrown', 'darkgreen', 'purple', 'yellow']

    # Apply DensityPeaksClustering
    dpc = DensityPeaksClustering(n_clusters=n_clusters_s3, percent=2.5, density_estimator='cutoff')
    predicted_labels_dpc = dpc.fit_predict(X)

    # Apply DPCV
    dpcv = DPCV(n_clusters=n_clusters_s3, n_neighbors=1)
    predicted_labels_dpcv = dpcv.fit_predict(X)

    # Apply K-means
    kmeans = KMeans(n_clusters=n_clusters_s3, random_state=42, n_init=10) # Added n_init explicitly
    predicted_labels_kmeans = kmeans.fit_predict(X)

    # Get color mappings based on matching with true labels (for DPC and K-means)
    dpc_color_map = get_cluster_color_mapping(predicted_labels_dpc, true_labels, canonical_color_names, color_hex)
    # dpcv_color_map = get_cluster_color_mapping(predicted_labels_dpcv, true_labels, canonical_color_names, color_hex) # Removed for DPCV
    kmeans_color_map = get_cluster_color_mapping(predicted_labels_kmeans, true_labels, canonical_color_names, color_hex)

    # Compute clustering metrics for S3 dataset
    s3_metrics = {
        'Algorithm': ['DensityPeaks', 'DPCV', 'K-means'],
        'ARI': [
            adjusted_rand_score(true_labels, predicted_labels_dpc),
            adjusted_rand_score(true_labels, predicted_labels_dpcv),
            adjusted_rand_score(true_labels, predicted_labels_kmeans)
        ],
        'NMI': [
            normalized_mutual_info_score(true_labels, predicted_labels_dpc),
            normalized_mutual_info_score(true_labels, predicted_labels_dpcv),
            normalized_mutual_info_score(true_labels, predicted_labels_kmeans)
        ],
        'Silhouette': [
            silhouette_score(X, predicted_labels_dpc),
            silhouette_score(X, predicted_labels_dpcv),
            silhouette_score(X, predicted_labels_kmeans)
        ],
        'DBI': [
            davies_bouldin_score(X, predicted_labels_dpc),
            davies_bouldin_score(X, predicted_labels_dpcv),
            davies_bouldin_score(X, predicted_labels_kmeans)
        ]
    }
    
    # Create DataFrame and save to CSV
    s3_metrics_df = pd.DataFrame(s3_metrics)
    s3_metrics_df = s3_metrics_df.round(4) # Round to 4 decimal places
    s3_metrics_df.to_csv(f"{results_dir}/S3_metrics.csv", index=False)
    print(f"\nClustering metrics for S3 dataset:")
    print(s3_metrics_df.to_string(index=False))
    
    # Plot decision graph for DensityPeaksClustering
    # We still need colors for the centers in the decision graph.
    # Let's map the center indices back to predicted labels, then use the dpc_color_map.
    center_labels_dpc = predicted_labels_dpc[dpc.centers_]
    dpc_decision_graph_hex_colors = [dpc_color_map[label] for label in center_labels_dpc]
    
    plot_decision_graph(
        dpc.rho_,
        dpc.delta_,
        dpc.centers_,
        dpc_decision_graph_hex_colors, # Use mapped colors
        os.path.join(results_dir, 'S3_decision_graph.png')
    )
    
    # Plot results
    fig = plt.figure(figsize=(12, 10))
    plt.subplots_adjust(bottom=0.1)  # Add more space at the bottom for captions
    
    # Original labels
    ax1 = plt.subplot(2, 2, 1)
    # Map unique labels to colors
    unique_labels = np.unique(true_labels)
    true_label_to_color_name = {label: canonical_color_names[i % len(canonical_color_names)] 
                                for i, label in enumerate(unique_labels)}
    for label in unique_labels:
        mask = true_labels == label
        color_name = true_label_to_color_name[label]
        plt.scatter(X[mask, 0], X[mask, 1], c=color_hex[color_name], s=15, alpha=1)
    plt.title("Original Labels")
    plt.axis('off')
    # Add border
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('black')
    # Add caption
    plt.text(0.5, -0.1, '(a)', transform=ax1.transAxes, fontsize=12, ha='center')
    
    # DensityPeaksClustering Predicted labels
    ax2 = plt.subplot(2, 2, 2)
    # Iterate through unique predicted labels and use the mapped color
    unique_predicted_dpc = np.unique(predicted_labels_dpc)
    for label in unique_predicted_dpc:
        mask = predicted_labels_dpc == label
        cluster_color_hex = dpc_color_map[label] # Get color from the map
        plt.scatter(X[mask, 0], X[mask, 1], c=cluster_color_hex, s=15, alpha=1)
    plt.title("DensityPeaksClustering Prediction")
    plt.axis('off')
    # Add border
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('black')
    # Add caption
    plt.text(0.5, -0.1, '(b)', transform=ax2.transAxes, fontsize=12, ha='center')

    # DPCV Predicted labels
    ax3 = plt.subplot(2, 2, 3)
    unique_predicted_dpcv = np.unique(predicted_labels_dpcv)
    # Iterate using enumerate to assign colors sequentially from dpcv_colors list
    for i, label in enumerate(unique_predicted_dpcv):
        mask = predicted_labels_dpcv == label
        # cluster_color_hex = dpcv_color_map[label] # Reverted: Get color from the map
        color_name = dpcv_colors[i % len(dpcv_colors)] # Use sequential assignment from dpcv_colors
        cluster_color_hex = color_hex[color_name]      # Get hex value
        plt.scatter(X[mask, 0], X[mask, 1], c=cluster_color_hex, s=15, alpha=1)
    plt.title("DPCV Prediction")
    plt.axis('off')
    # Add border
    for spine in ax3.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('black')
    # Add caption
    plt.text(0.5, -0.1, '(c)', transform=ax3.transAxes, fontsize=12, ha='center')

    # K-means Predicted labels
    ax4 = plt.subplot(2, 2, 4)
    unique_predicted_kmeans = np.unique(predicted_labels_kmeans)
    for label in unique_predicted_kmeans:
        mask = predicted_labels_kmeans == label
        cluster_color_hex = kmeans_color_map[label] # Get color from the map
        plt.scatter(X[mask, 0], X[mask, 1], c=cluster_color_hex, s=15, alpha=1)
    plt.title("K-means Prediction")
    plt.axis('off')
    # Add border
    for spine in ax4.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('black')
    # Add caption
    plt.text(0.5, -0.1, '(d)', transform=ax4.transAxes, fontsize=12, ha='center')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    
    # Save the plot
    plt.savefig(f"{results_dir}/S3.png", dpi=300, bbox_inches='tight')
    print(f"Plot saved to {results_dir}/S3.png")
    plt.close()

def run_experiment():
    """Main function to run all experiment test cases."""
    results_dir = "../results/experiment2"
    os.makedirs(results_dir, exist_ok=True)
    
    #run_aggregation()
    run_s3()
    #run_flame()
    #run_spiral()
    
if __name__ == "__main__":
    run_experiment()
