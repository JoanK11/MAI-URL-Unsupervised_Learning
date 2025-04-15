import numpy as np
import matplotlib.pyplot as plt
import os
from DensityPeaksClustering import DensityPeaksClustering

def run_s3_variation(percent, density_estimator='cutoff'):
    """Run DensityPeaksClustering on the S3 dataset with specific parameters."""
    # Read data from s3.txt (X and Y coordinates)
    data_file = "../data/experiment2/s3.txt"
    X = np.loadtxt(data_file)
    
    # Read labels from s3-label.txt
    label_file = "../data/experiment2/s3-label.txt"
    with open(label_file, 'r') as f:
        lines = f.readlines()
    true_labels = np.array([int(line.strip()) for line in lines if line.strip()])
    
    # Apply DensityPeaksClustering
    dpc = DensityPeaksClustering(n_clusters=15, percent=percent, density_estimator=density_estimator)
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
    
    # Define color ordering
    colors_list = ['black', 'darkblue', 'green', 'brown', 'blue', 
                  'orange', 'darkestblue', 'pink', 'turquoise', 'cyan', 
                  'red', 'lightbrown', 'darkgreen', 'purple', 'yellow']
    
    # Plot results
    fig = plt.figure(figsize=(12, 5))
    
    # Original labels
    ax1 = plt.subplot(1, 2, 1)
    for label in np.unique(true_labels):
        mask = true_labels == label
        color_name = colors_list[label - 1]  # labels start from 1
        plt.scatter(X[mask, 0], X[mask, 1], c=color_hex[color_name], s=15, alpha=1)
    plt.title("Original Labels")
    plt.axis('off')
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
    plt.title(f"DPC Prediction (percent={percent:.2f}, {density_estimator})")
    plt.axis('off')
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('black')
    
    plt.tight_layout()
    
    # Save the plot
    results_dir = f"../results/temp/s3_{density_estimator}"
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(f"{results_dir}/percent_{percent:.2f}.png", dpi=300, bbox_inches='tight')
    print(f"Plot saved to {results_dir}/percent_{percent:.2f}.png")
    plt.close()

def run_experiment():
    """Generate variations of S3 clustering with different parameters."""
    # Create results directories
    for estimator in ['gaussian', 'cutoff']:
        os.makedirs(f"../results/temp/s3_{estimator}", exist_ok=True)
    
    # Generate variations
    for percent in np.arange(0.1, 1.05, 0.05):
        # Run with gaussian estimator
        run_s3_variation(percent, 'gaussian')
        # Run with cutoff estimator
        run_s3_variation(percent, 'cutoff')

if __name__ == "__main__":
    run_experiment() 