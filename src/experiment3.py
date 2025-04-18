import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.preprocessing import StandardScaler
from DensityPeaksClustering import DensityPeaksClustering


def plot_decision_graph(rho, delta, center_indices, cluster_colors, output_path):
    plt.figure(figsize=(6, 5))
    plt.scatter(rho, delta, c='k', marker='o', s=25)
    for i, idx in enumerate(center_indices):
        plt.scatter(rho[idx], delta[idx], c=[cluster_colors[i]], marker='o', s=100)
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$\delta$')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_decision_graph_no_axes(rho, delta, center_indices, cluster_colors, output_path):
    plt.figure(figsize=(5, 4))
    plt.scatter(rho, delta, c='k', marker='o', s=25)
    for i, idx in enumerate(center_indices):
        plt.scatter(rho[idx], delta[idx], c=[cluster_colors[i]], marker='o', s=80)
    plt.axis('off')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_montage(images, labels, centers, cluster_colors, output_path, show_centers=True, unlabeled_color=(0.5, 0.5, 0.5)):
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
                # apply color for labeled clusters or a uniform color for unlabeled
                if labels[idx] == -1:
                    overlay[..., :3] = np.array(unlabeled_color)
                else:
                    color = cluster_colors[labels[idx]]
                    overlay[..., :3] = np.array(color[:3])
                overlay[..., 3] = 0.5
                ax.imshow(overlay, interpolation='nearest')
                ax.axis('off')
                # mark cluster centers with a white circle at top-right
                if show_centers and idx in centers:
                    ax.scatter(0.9, 0.9, s=80, c='white', marker='o', transform=ax.transAxes, edgecolors='none')
    plt.subplots_adjust(wspace=0, hspace=0)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    # 1. Load Olivetti faces
    faces = fetch_olivetti_faces()
    X_all = faces.data
    images = faces.images
    targets = faces.target

    # 2. Preprocess: standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # 3. Use only the first 100 faces (10 persons x 10 images)
    n = 100
    X = X_scaled[:n]
    images = images[:n]
    targets = targets[:n]

    n_clusters = len(np.unique(targets))

    results_dir = os.path.join('..', 'results', 'experiment3')
    os.makedirs(results_dir, exist_ok=True)
    cmap = plt.get_cmap('tab10', n_clusters)
    cluster_colors = [cmap(i) for i in range(n_clusters)]

    metrics = [
        ('cw-ssim', 'CW-SSIM', 'Olivetti_cw_ssim.png'),
        ('ssim', 'SSIM', 'Olivetti_ssim.png'),
    ]
    montage_paths = {}
    for sim_metric, title, file_name in metrics:
        dpc_m = DensityPeaksClustering(n_clusters=n_clusters, dc=0.07, similarity_metric=sim_metric, density_estimator='gaussian')
        labels_m = dpc_m.fit_predict(images)
        montage_path = os.path.join(results_dir, file_name)
        create_montage(images, labels_m, list(dpc_m.centers_), cluster_colors, montage_path)
        print(f"Montage image for {title} saved to {montage_path}")
        montage_paths[sim_metric] = montage_path
        
        # Add decision graph for CW-SSIM
        if sim_metric == 'cw-ssim':
            decision_graph_path = os.path.join(results_dir, 'Olivetti_decision_graph.png')
            plot_decision_graph(dpc_m.rho_, dpc_m.delta_, dpc_m.centers_, cluster_colors, decision_graph_path)
            print(f"Decision graph for {title} saved to {decision_graph_path}")

    # restrictive cw-ssim clustering montage
    dpc_r = DensityPeaksClustering(n_clusters=n_clusters, dc=0.07, similarity_metric='cw-ssim', density_estimator='gaussian', restrictive=True)
    labels_r = dpc_r.fit_predict(images)
    restrictive_montage_path = os.path.join(results_dir, 'Olivetti_restrictive_cw_ssim.png')
    create_montage(images, labels_r, list(dpc_r.centers_), cluster_colors, restrictive_montage_path, show_centers=False)
    print(f"Montage image for Restrictive CW-SSIM saved to {restrictive_montage_path}")

    # composite 2x2 figure
    final_fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    final_fig.subplots_adjust(wspace=0.2, hspace=0.2)
    img_list = [
        (os.path.join('..', 'results', 'experiment3', 'olivetti_paper.png'), '(A) Original Results'),
        (montage_paths['cw-ssim'], '(B) Non-Restrictive CW-SSIM'),
        (montage_paths['ssim'], '(C) Non-Restrictive SSIM'),
        (restrictive_montage_path, '(D) Restrictive CW-SSIM'),
    ]

    for ax, (img_path, title) in zip(axes.flatten(), img_list):
        img = plt.imread(img_path)
        ax.imshow(img, aspect='auto')
        ax.set_title(title, pad=10)
        ax.axis('off')

    final_composite_path = os.path.join(results_dir, 'Olivetti.png')
    final_fig.savefig(final_composite_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(final_fig)
    print(f"Composite image saved to {final_composite_path}")
    
    # Delete individual montage images after composite is created
    for _, path in montage_paths.items():
        if os.path.exists(path):
            os.remove(path)
            print(f"Removed temporary file: {path}")
    
    if os.path.exists(restrictive_montage_path):
        os.remove(restrictive_montage_path)
        print(f"Removed temporary file: {restrictive_montage_path}")


if __name__ == '__main__':
    main()