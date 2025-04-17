import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.preprocessing import StandardScaler
from DensityPeaksClustering import DensityPeaksClustering


def plot_decision_graph(rho, delta, center_indices, cluster_colors, output_path):
    plt.figure(figsize=(6, 5))
    # all points
    plt.scatter(rho, delta, c='k', marker='o', s=25)
    # highlight centers
    for i, idx in enumerate(center_indices):
        plt.scatter(rho[idx], delta[idx], c=[cluster_colors[i]], marker='o', s=100)
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$\delta$')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_montage(images, labels, centers, cluster_colors, output_path):
    rows, cols = 5, 20
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5))
    # for each row, two persons (10 images each)
    for row in range(rows):
        for p in range(2):
            person = 2 * row + p
            start_idx = person * 10
            for k in range(10):
                idx = start_idx + k
                ax = axes[row, p * 10 + k]
                # show grayscale face
                ax.imshow(images[idx], cmap='gray', interpolation='nearest')
                # overlay cluster color with transparency
                overlay = np.ones(images[idx].shape + (4,))
                color = cluster_colors[labels[idx]]
                overlay[..., :3] = np.array(color[:3])
                overlay[..., 3] = 0.5
                ax.imshow(overlay, interpolation='nearest')
                ax.axis('off')
                # mark cluster centers with a white circle at top-right
                if idx in centers:
                    ax.scatter(0.9, 0.9, s=80, c='white', marker='o', transform=ax.transAxes, edgecolors='none')
                # Remove subplot borders
                for spine in ax.spines.values():
                    spine.set_visible(False)
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    # 1. Load Olivetti faces
    faces = fetch_olivetti_faces()
    X_all = faces.data           # shape (400, 4096)
    images = faces.images        # shape (400, 64, 64)
    targets = faces.target       # true person labels

    # 2. Preprocess: standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # 3. Use only the first 100 faces (10 persons x 10 images)
    n = 100
    X = X_scaled[:n]
    images = images[:n]
    targets = targets[:n]

    # determine number of clusters (10 persons)
    n_clusters = len(np.unique(targets))

    # 4. Cluster using DensityPeaksClustering
    dpc = DensityPeaksClustering(n_clusters=n_clusters, dc=0.07, similarity_metric='cw-ssim', density_estimator='gaussian')
    labels = dpc.fit_predict(images)

    # 5. Plot decision graph
    results_dir = os.path.join('..', 'results', 'experiment3')
    dg_path = os.path.join(results_dir, 'Olivetti_decision_graph.png')
    # choose distinct colors for clusters
    cmap = plt.cm.get_cmap('tab10', n_clusters)
    cluster_colors = [cmap(i) for i in range(n_clusters)]
    plot_decision_graph(dpc.rho_, dpc.delta_, dpc.centers_, cluster_colors, dg_path)
    print(f"Decision graph saved to {dg_path}")

    # 6. Create and save montage of 100 faces colored by cluster
    montage_path = os.path.join(results_dir, 'Olivetti.png')
    create_montage(images, labels, list(dpc.centers_), cluster_colors, montage_path)
    print(f"Montage image saved to {montage_path}")


if __name__ == '__main__':
    main()
