import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from DensityPeaksClustering import DensityPeaksClustering

def read_distance_matrix(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    data = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            i, j, dist = int(parts[0]), int(parts[1]), float(parts[2])
            data.append((i, j, dist))
    
    # Find maximum index to determine matrix size
    max_idx = max(max(i, j) for i, j, _ in data)
    
    # Initialize distance matrix
    dist_matrix = np.zeros((max_idx, max_idx))
    
    # Fill the matrix with distances
    for i, j, dist in data:
        i_idx, j_idx = i-1, j-1
        dist_matrix[i_idx, j_idx] = dist
        dist_matrix[j_idx, i_idx] = dist
    
    return dist_matrix

def plot_decision_graph(rho, delta, center_indices, labels, cluster_colors, output_path):
    plt.figure(figsize=(5, 4))
    
    # Plot all points
    plt.scatter(rho, delta, c='k', marker='o', s=25)
    
    # Highlight centers with different colors
    for i, center_idx in enumerate(center_indices):
        plt.scatter(rho[center_idx], delta[center_idx], c=[cluster_colors[i]], marker='o', s=80)
    
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$\delta$')
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    
def plot_mds_visualization(points, labels, halo, cluster_colors, output_path):
    plt.figure(figsize=(8, 8))
    
    # Get unique cluster labels
    unique_labels = np.unique(labels)
    
    # Plot all points as black circles
    plt.scatter(points[:, 0], points[:, 1], c='k', marker='o', s=25)
    
    # Plot clusters
    for i, label in enumerate(unique_labels):
        if label == -1:
            continue
            
        cluster_points = points[labels == label]
        halo_mask = halo[labels == label]
        
        # Plot non-halo points with color
        if np.any(~halo_mask):
            plt.scatter(
                cluster_points[~halo_mask, 0],
                cluster_points[~halo_mask, 1],
                c=cluster_colors[i],
                s=25,
                marker='o'
            )
    
    plt.xlabel('')
    plt.ylabel('')
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    
    x_range = np.ptp(points[:, 0])
    y_range = np.ptp(points[:, 1])
    x_mean = np.mean(points[:, 0])
    y_mean = np.mean(points[:, 1])
    margin = 0.15
    plt.xlim(x_mean - x_range * (0.5 + margin), x_mean + x_range * (0.5 + margin))
    plt.ylim(y_mean - y_range * (0.5 + margin), y_mean + y_range * (0.5 + margin))
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_experiment(distance_matrix, n_points, plots_dir, suffix):
    if n_points < distance_matrix.shape[0]:
        np.random.seed(42)
        indices = np.random.choice(distance_matrix.shape[0], n_points, replace=False)
        indices.sort()
        sampled_matrix = distance_matrix[indices][:, indices]
    else:
        sampled_matrix = distance_matrix
        indices = np.arange(distance_matrix.shape[0])
    
    dpc = DensityPeaksClustering(percent=2.5, n_clusters=5, density_estimator='gaussian')
    
    # Use MDS to convert distances to coordinates for fitting
    temp_mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    temp_points = temp_mds.fit_transform(sampled_matrix)
    
    dpc.fit(temp_points)
    labels = dpc.labels_
    halo = dpc.halo_
    
    # Define colors (Blue, Red, Orange, Pink, Green)
    current_colors = ['#3853A3', '#EE1F23', '#F57F21', '#ED107F', '#0D803F']
    if n_points == 1000:
        # Blue, Red, Green, Orange, Pink
        current_colors = ['#3853A3', '#EE1F23', '#0D803F', '#F57F21', '#ED107F']
    
    # Generate visualizations
    plot_decision_graph(
        dpc.rho_, 
        dpc.delta_, 
        dpc.centers_, 
        labels, 
        current_colors,
        os.path.join(plots_dir, f'decision_graph_{suffix}.png')
    )
    
    mds = MDS(n_components=2, dissimilarity='precomputed', metric=True, 
              n_init=1, random_state=None, normalized_stress='auto')
    points = mds.fit_transform(sampled_matrix)
    
    plot_mds_visualization(
        points,
        labels,
        halo,
        current_colors,
        os.path.join(plots_dir, f'mds_visualization_{suffix}.png')
    )
    
    return n_points, len(np.unique(labels)), np.sum(~halo), np.sum(halo)

def main():
    print("Running Experiment 1...")
    
    # Create output directories
    plots_dir = os.path.join("../results", "experiment1")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Read the input file and convert to distance matrix
    input_file = "../data/experiment1/example_distances.dat"
    distance_matrix = read_distance_matrix(input_file)
    
    # Run experiments for both full and sampled datasets
    experiments = [
        (distance_matrix.shape[0], '2000'),
        (1000, '1000')
    ]
    
    print("\nResults Summary:")
    print("===============")
    
    for n_points, suffix in experiments:
        points, n_clusters, n_core, n_halo = run_experiment(distance_matrix, n_points, plots_dir, suffix)
        print(f"\nDataset: {suffix} points")
        print(f"Number of points: {points}")
        print(f"Number of clusters: {n_clusters}")
        print(f"Core points: {n_core} ({n_core / points * 100:.1f}%)")
        print(f"Halo points: {n_halo} ({n_halo / points * 100:.1f}%)")
    
    print(f"\nPlots saved to: {plots_dir}")

if __name__ == "__main__":
    main() 