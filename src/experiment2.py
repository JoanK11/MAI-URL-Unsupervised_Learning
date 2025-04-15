import numpy as np
import matplotlib.pyplot as plt
import os
from DensityPeaksClustering import DensityPeaksClustering

def run_aggregation():
    """Run DensityPeaksClustering on the Aggregation dataset."""
    # Read data from Aggregation.txt
    data_file = "../data/experiment2/Aggregation.txt"
    data = np.loadtxt(data_file, delimiter='\t')
    
    # Extract features (X, Y) and true labels
    X = data[:, :2]  # First two columns are X, Y coordinates
    true_labels = data[:, 2].astype(int)  # Third column contains the true labels
    
    # Apply DensityPeaksClustering
    dpc = DensityPeaksClustering(n_clusters=7, density_estimator='gaussian')
    predicted_labels = dpc.fit_predict(X)
    
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
    
    # Define color ordering for predicted clusters
    predicted_colors = ['red', 'yellow', 'orange', 'darkblue', 'black', 'blue', 'green']
    
    # Plot results
    fig = plt.figure(figsize=(12, 5))
    
    # Original labels
    ax1 = plt.subplot(1, 2, 1)
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
    
    # Predicted labels
    ax2 = plt.subplot(1, 2, 2)
    for label in np.unique(predicted_labels):
        mask = predicted_labels == label
        color_name = predicted_colors[label % len(predicted_colors)]
        plt.scatter(X[mask, 0], X[mask, 1], c=color_hex[color_name], s=30, alpha=0.8)
    plt.title("DensityPeaksClustering Prediction")
    plt.axis('off')
    # Add border
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('black')
    
    plt.tight_layout()
    
    # Save the plot
    results_dir = "../results/experiment2"
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(f"{results_dir}/Aggregation.png", dpi=300, bbox_inches='tight')
    print(f"Plot saved to {results_dir}/Aggregation.png")
    plt.close()

def run_flame():
    """Run DensityPeaksClustering on the Flame dataset."""
    # Read data from flame.txt
    data_file = "../data/experiment2/flame.txt"
    data = np.loadtxt(data_file, delimiter='\t')
    
    # Extract features (X, Y) and true labels
    X = data[:, :2]
    true_labels = data[:, 2].astype(int)
    
    # Apply DensityPeaksClustering
    dpc = DensityPeaksClustering(n_clusters=2, percent=3.0, density_estimator='gaussian')
    predicted_labels = dpc.fit_predict(X)
    
    color_hex = {
        'blue': '#3853A3',
        'red': '#EE1F23'
    }
    
    # Plot results
    fig = plt.figure(figsize=(12, 5))
    
    # Original labels
    ax1 = plt.subplot(1, 2, 1)
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
    
    # Predicted labels
    ax2 = plt.subplot(1, 2, 2)
    for label in np.unique(predicted_labels):
        mask = predicted_labels == label
        color_name = 'red' if label == 0 else 'blue'
        plt.scatter(X[mask, 0], X[mask, 1], c=color_hex[color_name], s=30, alpha=0.8)
    plt.title("DensityPeaksClustering Prediction")
    plt.axis('off')
    # Add border
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('black')
    
    plt.tight_layout()
    
    # Save the plot
    results_dir = "../results/experiment2"
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(f"{results_dir}/Flame.png", dpi=300, bbox_inches='tight')
    print(f"Plot saved to {results_dir}/Flame.png")
    plt.close()

def run_spiral():
    """Run DensityPeaksClustering on the Spiral dataset."""
    # Read data from spiral.txt
    data_file = "../data/experiment2/spiral.txt"
    data = np.loadtxt(data_file, delimiter='\t')
    
    # Extract features (X, Y) and true labels
    X = data[:, :2]
    true_labels = data[:, 2].astype(int)
    
    # Apply DensityPeaksClustering
    dpc = DensityPeaksClustering(n_clusters=3, percent=3.0, density_estimator='gaussian')
    predicted_labels = dpc.fit_predict(X)
    
    # Define color palette with hex values
    color_hex = {
        'green': '#69BD45',
        'red': '#EE1F23',
        'blue': '#4077BC'
    }
    
    # Plot results
    fig = plt.figure(figsize=(12, 5))
    
    # Original labels
    ax1 = plt.subplot(1, 2, 1)
    for label in np.unique(true_labels):
        mask = true_labels == label
        if label == 1:
            color_name = 'blue'
        elif label == 2:
            color_name = 'green'
        else:  # label == 3
            color_name = 'red'
        plt.scatter(X[mask, 0], X[mask, 1], c=color_hex[color_name], s=30, alpha=0.8)
    plt.title("Original Labels")
    plt.axis('off')
    # Add border
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('black')
    
    # Predicted labels
    ax2 = plt.subplot(1, 2, 2)
    for label in np.unique(predicted_labels):
        mask = predicted_labels == label
        if label == 0:
            color_name = 'red'
        elif label == 1:
            color_name = 'green'
        else:
            color_name = 'blue'
        plt.scatter(X[mask, 0], X[mask, 1], c=color_hex[color_name], s=30, alpha=0.8)
    plt.title("DensityPeaksClustering Prediction")
    plt.axis('off')
    # Add border
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('black')
    
    plt.tight_layout()
    
    # Save the plot
    results_dir = "../results/experiment2"
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(f"{results_dir}/Spiral.png", dpi=300, bbox_inches='tight')
    print(f"Plot saved to {results_dir}/Spiral.png")
    plt.close()

def run_s3():
    """Run DensityPeaksClustering on the S3 dataset."""
    # Read data from s3.txt (X and Y coordinates)
    data_file = "../data/experiment2/s3.txt"
    X = np.loadtxt(data_file)
    
    # Read labels from s3-label.txt
    label_file = "../data/experiment2/s3-label.txt"
    with open(label_file, 'r') as f:
        lines = f.readlines()
    true_labels = np.array([int(line.strip()) for line in lines if line.strip()])
    
    # Apply DensityPeaksClustering
    dpc = DensityPeaksClustering(n_clusters=15, percent=2.5, density_estimator='cutoff')
    predicted_labels = dpc.fit_predict(X)
    
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
    
    # Map unique labels to colors
    unique_labels = np.unique(true_labels)
    color_mapping = {}
    colors_list = ['black', 'darkblue', 'green', 'brown', 'blue', 
                  'orange', 'darkestblue', 'pink', 'turquoise', 'cyan', 
                  'red', 'lightbrown', 'darkgreen', 'purple', 'yellow']
    #colors_list = ['blue', 'darkblue', 'darkestblue', 'green', 'darkgreen', 
     #             'orange', 'red', 'yellow', 'pink', 'purple', 
      #            'brown', 'lightbrown', 'black', 'turquoise', 'cyan']
    
    
    for i, label in enumerate(unique_labels):
        color_mapping[label] = colors_list[i % len(colors_list)]
    
    # Plot results
    fig = plt.figure(figsize=(12, 5))
    
    # Original labels
    ax1 = plt.subplot(1, 2, 1)
    for label in unique_labels:
        mask = true_labels == label
        color_name = color_mapping[label]
        plt.scatter(X[mask, 0], X[mask, 1], c=color_hex[color_name], s=15, alpha=1)
    plt.title("Original Labels")
    plt.axis('off')
    # Add border
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('black')
    
    # Predicted labels
    ax2 = plt.subplot(1, 2, 2)
    for label in np.unique(predicted_labels):
        mask = predicted_labels == label
        color_name = colors_list[label % len(colors_list)]
        plt.scatter(X[mask, 0], X[mask, 1], c=color_hex[color_name], s=15, alpha=1)
    plt.title("DensityPeaksClustering Prediction")
    plt.axis('off')
    # Add border
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('black')
    
    plt.tight_layout()
    
    # Save the plot
    results_dir = "../results/experiment2"
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(f"{results_dir}/S3.png", dpi=300, bbox_inches='tight')
    print(f"Plot saved to {results_dir}/S3.png")
    plt.close()

def run_experiment():
    """Main function to run all experiment test cases."""
    results_dir = "../results/experiment2"
    os.makedirs(results_dir, exist_ok=True)
    
    # run_aggregation()
    run_s3()
    # run_flame()
    # run_spiral()
    
if __name__ == "__main__":
    run_experiment()
