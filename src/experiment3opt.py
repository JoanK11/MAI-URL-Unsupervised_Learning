import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.preprocessing import StandardScaler
from DensityPeaksClustering import DensityPeaksClustering
from sklearn.metrics import adjusted_rand_score
from skopt import gp_minimize
from skopt.space import Integer, Real
import json


def save_hyperparameters(dataset_name, hyperparameters, results_dir):
    """Save optimized hyperparameters to JSON in results_dir."""
    filepath = os.path.join(results_dir, 'dpc_olivetti_hyperparameters.json')
    try:
        with open(filepath, 'r') as f:
            all_params = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_params = {}
    all_params[dataset_name] = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k,v in hyperparameters.items()}
    with open(filepath, 'w') as f:
        json.dump(all_params, f, indent=4)
    return filepath


def optimize_dpc_olivetti(images, targets, n_clusters, results_dir):
    """Bayesian optimize DPC parameters to maximize ARI."""
    # prepare cluster colors for visualizations
    cmap = plt.cm.get_cmap('tab10', n_clusters)
    cluster_colors = [cmap(i) for i in range(n_clusters)]
    
    # search space: optimize dc, cw_levels, cw_K, cw_guardb
    space = [
        Real(0.01, 0.2, prior='log-uniform', name='dc'),
        Integer(1, 6, name='cw_levels'),
        Real(1e-10, 1e-6, prior='log-uniform', name='cw_K'),
        Integer(0, 3, name='cw_guardb')
    ]
    
    def objective(params):
        dc, cw_levels, cw_K, cw_guardb = params
        # instantiate model with given parameters
        model = DensityPeaksClustering(
            n_clusters=n_clusters,
            dc=dc,
            density_estimator='gaussian',
            similarity_metric='cw-ssim',
            cwssim_level=cw_levels,
            cwssim_K=cw_K,
            cwssim_guardb=cw_guardb
        )
        # fit model and obtain labels
        model.fit(images)
        labels = model.labels_
        # save visual results for this configuration
        dg_path = os.path.join(results_dir, f"Olivetti_decision_graph_dc{dc:.3f}_levels{cw_levels}_K{cw_K:.0e}_guardb{cw_guardb}.png")
        plot_decision_graph(model.rho_, model.delta_, model.centers_, cluster_colors, dg_path)
        montage_path = os.path.join(results_dir, f"Olivetti_montage_dc{dc:.3f}_levels{cw_levels}_K{cw_K:.0e}_guardb{cw_guardb}.png")
        create_montage(images, labels, list(model.centers_), cluster_colors, montage_path)
        # maximize ARI -> minimize negative ARI
        return -adjusted_rand_score(targets, labels)

    res = gp_minimize(objective, space, n_calls=30, random_state=42)
    # retrieve best parameters
    best_dc, best_levels, best_K, best_guardb = res.x
    # build best model with optimized parameters
    best_model = DensityPeaksClustering(
        n_clusters=n_clusters,
        dc=best_dc,
        density_estimator='gaussian',
        similarity_metric='cw-ssim',
        cwssim_level=best_levels,
        cwssim_K=best_K,
        cwssim_guardb=best_guardb
    )
    best_model.fit(images)
    # save optimized hyperparameters
    save_hyperparameters('olivetti', {
        'dc': best_dc,
        'cw_levels': best_levels,
        'cw_K': best_K,
        'cw_guardb': best_guardb
    }, results_dir)
    print(f"Saved optimized hyperparameters to {results_dir}")
    return best_model


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
    plt.subplots_adjust(wspace=0, hspace=0)
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

    # choose distinct colors for clusters
    cmap = plt.cm.get_cmap('tab10', n_clusters)
    cluster_colors = [cmap(i) for i in range(n_clusters)]

    # --- optimization step ---
    opt_dir = os.path.join('..', 'results', 'experiment3opt')
    os.makedirs(opt_dir, exist_ok=True)
    best_dpc = optimize_dpc_olivetti(images, targets, n_clusters, opt_dir)
    # use best model to get labels and decision graph
    opt_labels = best_dpc.labels_
    plot_decision_graph(best_dpc.rho_, best_dpc.delta_, best_dpc.centers_, cluster_colors,
                        os.path.join(opt_dir, 'Olivetti_decision_graph_opt.png'))
    create_montage(images, opt_labels, list(best_dpc.centers_), cluster_colors,
                   os.path.join(opt_dir, 'Olivetti_opt.png'))
    print(f"Optimized plots saved to {opt_dir}")

    # 4. Cluster using DensityPeaksClustering
    dpc = DensityPeaksClustering(n_clusters=n_clusters, dc=0.07, similarity_metric='cw-ssim', density_estimator='gaussian')
    labels = dpc.fit_predict(images)

    # 5. Plot decision graph
    results_dir = os.path.join('..', 'results', 'experiment3')
    dg_path = os.path.join(results_dir, 'Olivetti_decision_graph.png')
    plot_decision_graph(dpc.rho_, dpc.delta_, dpc.centers_, cluster_colors, dg_path)
    print(f"Decision graph saved to {dg_path}")

    # 6. Create and save montage of 100 faces colored by cluster
    montage_path = os.path.join(results_dir, 'Olivetti.png')
    create_montage(images, labels, list(dpc.centers_), cluster_colors, montage_path)
    print(f"Montage image saved to {montage_path}")


if __name__ == '__main__':
    main()
