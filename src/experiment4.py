import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json

# Clustering algorithms
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from DensityPeaksClustering import DensityPeaksClustering

# Metrics
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Datasets
from sklearn import datasets

# Bayesian optimization
from skopt import BayesSearchCV, gp_minimize
from skopt.space import Real, Integer, Categorical

# For visualization and saving results
import os
from sklearn.manifold import TSNE

def plot_decision_graph(rho, delta, center_indices, cluster_colors, output_path=None, ax=None, title=None):
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
    output_path : str, optional
        Path where to save the plot. If None, plot is not saved.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.
    title : str, optional
        Title for the plot
    """
    if ax is None:
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        standalone = True
    else:
        standalone = False
    
    # Plot all points in black with some transparency
    ax.scatter(rho, delta, c='black', marker='o', s=25, alpha=0.5)
    
    # Highlight centers with their corresponding colors
    for i, center_idx in enumerate(center_indices):
        ax.scatter(rho[center_idx], delta[center_idx], c=[cluster_colors[i]], marker='o', s=80)
    
    ax.set_xlabel(r'$\rho$')
    ax.set_ylabel(r'$\delta$')
    
    if title:
        ax.set_title(title)
    
    if standalone:
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

def load_dataset(dataset_name, random_state=42):
    # Create directory for results if it doesn't exist
    os.makedirs("../results/experiment4", exist_ok=True)
    
    if dataset_name == 'iris':
        data = datasets.load_iris()
        X, y = data.data, data.target
        n_clusters = len(np.unique(y))
        preprocessing = Pipeline([
            ('scaler', StandardScaler())
        ])
        X = preprocessing.fit_transform(X)
        
    elif dataset_name == 'wine':
        data = datasets.load_wine()
        X, y = data.data, data.target
        n_clusters = len(np.unique(y))
        preprocessing = Pipeline([
            ('scaler', StandardScaler())
        ])
        X = preprocessing.fit_transform(X)
        
    elif dataset_name == 'digits':
        data = datasets.load_digits()
        X, y = data.data, data.target
        n_clusters = len(np.unique(y))
        preprocessing = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95))
        ])
        X = preprocessing.fit_transform(X)
        
    elif dataset_name == 'moons':
        X, y = datasets.make_moons(n_samples=200, noise=0.05, random_state=random_state)
        n_clusters = len(np.unique(y))
                        
    elif dataset_name == 'blobs':
        X, y = datasets.make_blobs(n_samples=200, centers=3, random_state=random_state)
        n_clusters = len(np.unique(y))
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return X, y, n_clusters

def save_hyperparameters(dataset_name, model_name, hyperparameters):
    # Convert numpy types to Python native types
    def convert_to_native(obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, 
                          np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Convert all values in hyperparameters
    converted_hyperparams = {}
    for key, value in hyperparameters.items():
        converted_hyperparams[key] = convert_to_native(value)

    filepath = f"../results/experiment4/{model_name}_hyperparameters.json"
    
    try:
        # Try to load existing hyperparameters
        with open(filepath, 'r') as f:
            all_hyperparams = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_hyperparams = {}
    
    # Update hyperparameters for this dataset
    all_hyperparams[dataset_name] = converted_hyperparams
    
    # Save updated hyperparameters
    with open(filepath, 'w') as f:
        json.dump(all_hyperparams, f, indent=4)
    
    return filepath

def load_hyperparameters(dataset_name, model_name):
    filepath = f"../results/experiment4/{model_name}_hyperparameters.json"
    try:
        with open(filepath, 'r') as f:
            all_hyperparams = json.load(f)
            return all_hyperparams.get(dataset_name)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def optimize_dpc(X, y, n_clusters, dataset_name, random_state=42):
    # Check if hyperparameters exist
    hyperparams = load_hyperparameters(dataset_name, "dpc")
    if hyperparams is not None:
        print("  Using saved DPC hyperparameters...")
        best_dpc = DensityPeaksClustering(
            n_clusters=n_clusters,
            percent=hyperparams['percent'],
            density_estimator=hyperparams['density_estimator']
        )
        best_dpc.fit(X)
        return best_dpc

    def objective(params):
        percent, density_estimator = params
        dpc = DensityPeaksClustering(
            n_clusters=n_clusters,
            percent=percent,
            density_estimator=density_estimator
        )
        labels = dpc.fit_predict(X)
        return -adjusted_rand_score(y, labels)

    space = [
        Real(0.5, 5.0, prior='uniform', name='percent'),
        Categorical(['cutoff', 'gaussian'], name='density_estimator')
    ]

    res = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=20,
        random_state=random_state
    )

    best_percent, best_density_estimator = res.x
    best_dpc = DensityPeaksClustering(
        n_clusters=n_clusters,
        percent=best_percent,
        density_estimator=best_density_estimator
    )
    best_dpc.fit(X)
    
    # Save hyperparameters
    hyperparams = {
        'percent': best_percent,
        'density_estimator': best_density_estimator
    }
    save_hyperparameters(dataset_name, "dpc", hyperparams)
    
    return best_dpc

def optimize_kmeans(X, y, n_clusters, dataset_name, random_state=42):
    # Check if hyperparameters exist
    hyperparams = load_hyperparameters(dataset_name, "kmeans")
    if hyperparams is not None:
        print("  Using saved KMeans hyperparameters...")
        return KMeans(
            n_clusters=n_clusters,
            n_init=hyperparams['n_init'],
            init=hyperparams['init'],
            random_state=random_state
        )

    param_space = {
        'n_init': Integer(5, 15),
        'init': Categorical(['k-means++', 'random'])
    }
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    
    optimizer = BayesSearchCV(
        kmeans,
        param_space,
        n_iter=20,
        cv=3,
        scoring='adjusted_rand_score',
        random_state=random_state
    )
    
    optimizer.fit(X, y)
    
    # Save hyperparameters
    hyperparams = {
        'n_init': optimizer.best_params_['n_init'],
        'init': optimizer.best_params_['init']
    }
    save_hyperparameters(dataset_name, "kmeans", hyperparams)
    
    return optimizer.best_estimator_

def optimize_dbscan(X, y, dataset_name, random_state=42):
    # Check if hyperparameters exist
    hyperparams = load_hyperparameters(dataset_name, "dbscan")
    if hyperparams is not None:
        print("  Using saved DBSCAN hyperparameters...")
        return DBSCAN(
            eps=hyperparams['eps'],
            min_samples=hyperparams['min_samples']
        )

    def objective(params):
        eps, min_samples = params
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
        return -adjusted_rand_score(y, labels)

    space = [
        Real(0.1, 2.0, prior='log-uniform', name='eps'),
        Integer(2, 10, name='min_samples')
    ]

    res = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=20,
        random_state=random_state
    )

    best_eps, best_min_samples = res.x
    best_db = DBSCAN(eps=best_eps, min_samples=best_min_samples)
    best_db.fit(X)
    
    # Save hyperparameters
    hyperparams = {
        'eps': best_eps,
        'min_samples': best_min_samples
    }
    save_hyperparameters(dataset_name, "dbscan", hyperparams)
    
    return best_db

def optimize_spectral(X, y, n_clusters, dataset_name, random_state=42):
    # Check if hyperparameters exist
    hyperparams = load_hyperparameters(dataset_name, "spectral")
    if hyperparams is not None:
        print("  Using saved SpectralClustering hyperparameters...")
        return SpectralClustering(
            n_clusters=n_clusters,
            gamma=hyperparams['gamma'],
            assign_labels=hyperparams['assign_labels'],
            affinity=hyperparams['affinity'],
            random_state=random_state
        )

    def objective(params):
        gamma, assign_labels, affinity = params
        labels = SpectralClustering(
            n_clusters=n_clusters,
            gamma=gamma,
            assign_labels=assign_labels,
            affinity=affinity,
            random_state=random_state
        ).fit_predict(X)
        return -adjusted_rand_score(y, labels)

    space = [
        Real(0.01, 10.0, prior='log-uniform', name='gamma'),
        Categorical(['kmeans', 'discretize'], name='assign_labels'),
        Categorical(['nearest_neighbors', 'rbf'], name='affinity')
    ]

    res = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=20,
        random_state=random_state
    )

    best_gamma, best_assign, best_affinity = res.x
    best_model = SpectralClustering(
        n_clusters=n_clusters,
        gamma=best_gamma,
        assign_labels=best_assign,
        affinity=best_affinity,
        random_state=random_state
    )
    best_model.fit(X)
    
    # Save hyperparameters
    hyperparams = {
        'gamma': best_gamma,
        'assign_labels': best_assign,
        'affinity': best_affinity
    }
    save_hyperparameters(dataset_name, "spectral", hyperparams)
    
    return best_model

def optimize_gmm(X, y, n_clusters, dataset_name, random_state=42):
    # Check if hyperparameters exist
    hyperparams = load_hyperparameters(dataset_name, "gmm")
    if hyperparams is not None:
        print("  Using saved GaussianMixture hyperparameters...")
        return GaussianMixture(
            n_components=n_clusters,
            covariance_type=hyperparams['covariance_type'],
            init_params=hyperparams['init_params'],
            max_iter=hyperparams['max_iter'],
            random_state=random_state
        )

    param_space = {
        'covariance_type': Categorical(['full', 'tied', 'diag', 'spherical']),
        'init_params': Categorical(['kmeans', 'random']),
        'max_iter': Integer(50, 200)
    }
    
    gmm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    
    optimizer = BayesSearchCV(
        gmm,
        param_space,
        n_iter=20,
        cv=3,
        scoring='adjusted_rand_score',
        random_state=random_state
    )
    
    optimizer.fit(X, y)
    
    # Save hyperparameters
    hyperparams = {
        'covariance_type': optimizer.best_params_['covariance_type'],
        'init_params': optimizer.best_params_['init_params'],
        'max_iter': optimizer.best_params_['max_iter']
    }
    save_hyperparameters(dataset_name, "gmm", hyperparams)
    
    return optimizer.best_estimator_

def calculate_metrics(X, y_true, y_pred):
    # Handle the case where DBSCAN might assign -1 to noise points
    valid_indices = y_pred != -1 if -1 in y_pred else np.ones(len(y_pred), dtype=bool)
    
    if sum(valid_indices) < 2:
        # Not enough samples for proper evaluation
        return {
            'ARI': np.nan,
            'NMI': np.nan,
            'Silhouette': np.nan,
            'DBI': np.nan
        }
    
    X_valid = X[valid_indices]
    y_true_valid = y_true[valid_indices]
    y_pred_valid = y_pred[valid_indices]
    
    # Only compute silhouette score if we have at least 2 clusters
    unique_clusters = np.unique(y_pred_valid)
    if len(unique_clusters) <= 1:
        silhouette = np.nan
        dbi = np.nan
    else:
        try:
            silhouette = silhouette_score(X_valid, y_pred_valid)
            dbi = davies_bouldin_score(X_valid, y_pred_valid)
        except Exception:
            silhouette = np.nan
            dbi = np.nan
    
    # ARI and NMI can be computed regardless
    ari = adjusted_rand_score(y_true_valid, y_pred_valid)
    nmi = normalized_mutual_info_score(y_true_valid, y_pred_valid)
    
    return {
        'ARI': ari,
        'NMI': nmi,
        'Silhouette': silhouette,
        'DBI': dbi
    }

def visualize_clusters(X, y_true, y_dpc, y_kmeans, y_dbscan, y_spectral, y_gmm, dataset_name, random_state=42):
    # Get unique colors for the number of clusters in the true labels
    # Sort labels to ensure consistent color assignment
    unique_labels = np.sort(np.unique(y_true))
    n_clusters = len(unique_labels)
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    
    # If dimensions > 2, use TSNE to reduce to 2D for visualization
    if X.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=random_state)
        X_2d = tsne.fit_transform(X)
    else:
        X_2d = X
    
    # Set up the figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Clustering Results for {dataset_name.capitalize()} Dataset', fontsize=16)
    
    # Ground Truth
    ax = axes[0, 0]
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, cmap='viridis', s=30, alpha=0.8)
    ax.set_title('Ground Truth')
    
    # DPC
    ax = axes[0, 1]
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_dpc, cmap='viridis', s=30, alpha=0.8)
    ax.set_title('DPC')
    
    # Store colors for decision graph - sort labels to ensure consistency
    dpc_unique_labels = np.sort(np.unique(y_dpc))
    dpc_colors = [colors[np.where(unique_labels == label)[0][0]] for label in dpc_unique_labels if label != -1]
    
    # K-means
    ax = axes[0, 2]
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_kmeans, cmap='viridis', s=30, alpha=0.8)
    ax.set_title('K-means')
    
    # DBSCAN
    ax = axes[1, 0]
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_dbscan, cmap='viridis', s=30, alpha=0.8)
    ax.set_title('DBSCAN')
    
    # Spectral Clustering
    ax = axes[1, 1]
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_spectral, cmap='viridis', s=30, alpha=0.8)
    ax.set_title('Spectral Clustering')
    
    # GMM
    ax = axes[1, 2]
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_gmm, cmap='viridis', s=30, alpha=0.8)
    ax.set_title('GMM')
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = f"../results/experiment4/{dataset_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return dpc_colors

def visualize_decision_graphs(all_dpc_models, dataset_names, all_colors):
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Decision Graphs for DPC', fontsize=16)
    
    axes = axes.flatten()
    
    # Plot decision graph for each dataset
    for i, (dpc, dataset_name, colors) in enumerate(zip(all_dpc_models, dataset_names, all_colors)):
        center_indices = dpc.centers_
        
        plot_decision_graph(
            dpc.rho_,
            dpc.delta_,
            center_indices,
            colors,
            ax=axes[i],
            title=dataset_name.capitalize()
        )
    
    # Remove any empty subplots
    for j in range(i+1, 9):
        fig.delaxes(axes[j])
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = "../results/experiment4/decision_graphs.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Decision graphs saved to {save_path}")

def main(random_state=42):
    datasets = ['iris', 'wine', 'digits', 'moons', 'blobs']
    
    all_dpc_models = []
    all_colors = []
    
    for dataset_name in datasets:
        print(f"Processing {dataset_name} dataset...")
        
        # Load and preprocess the dataset
        X, y_true, n_clusters = load_dataset(dataset_name, random_state=random_state)
        
        # Optimize and fit clustering models
        print("  Optimizing DensityPeaksClustering...")
        dpc = optimize_dpc(X, y_true, n_clusters, dataset_name, random_state=random_state)
        y_dpc = dpc.fit_predict(X)
        
        all_dpc_models.append(dpc)
        
        print("  Optimizing KMeans...")
        kmeans = optimize_kmeans(X, y_true, n_clusters, dataset_name, random_state=random_state)
        y_kmeans = kmeans.fit_predict(X)
        
        print("  Optimizing DBSCAN...")
        dbscan = optimize_dbscan(X, y_true, dataset_name, random_state=random_state)
        y_dbscan = dbscan.fit_predict(X)
        
        print("  Optimizing SpectralClustering...")
        spectral = optimize_spectral(X, y_true, n_clusters, dataset_name, random_state=random_state)
        y_spectral = spectral.fit_predict(X)
        
        print("  Optimizing GaussianMixture...")
        gmm = optimize_gmm(X, y_true, n_clusters, dataset_name, random_state=random_state)
        y_gmm = gmm.fit_predict(X)
        
        # Calculate metrics
        metrics_dpc = calculate_metrics(X, y_true, y_dpc)
        metrics_kmeans = calculate_metrics(X, y_true, y_kmeans)
        metrics_dbscan = calculate_metrics(X, y_true, y_dbscan)
        metrics_spectral = calculate_metrics(X, y_true, y_spectral)
        metrics_gmm = calculate_metrics(X, y_true, y_gmm)
        
        # Save all model metrics
        df_metrics = pd.DataFrame([
            metrics_dpc,
            metrics_kmeans,
            metrics_dbscan,
            metrics_spectral,
            metrics_gmm
        ], index=['DPC', 'K-means', 'DBSCAN', 'Spectral Clustering', 'GMM'])

        df_metrics = df_metrics[['ARI', 'NMI', 'Silhouette', 'DBI']].round(2)
        output_path = f"../results/experiment4/{dataset_name}_metrics.csv"
        df_metrics.to_csv(output_path, index_label='Model')
        
        # Create and save visualization, get colors for decision graph
        dpc_colors = visualize_clusters(X, y_true, y_dpc, y_kmeans, y_dbscan, y_spectral, y_gmm, dataset_name, random_state=random_state)
        all_colors.append(dpc_colors)
        
        print(f"  Results saved for all models")
        print(f"  Visualization saved to: ../results/experiment4/{dataset_name}.png")
        print()
    
    visualize_decision_graphs(all_dpc_models, datasets, all_colors)

if __name__ == "__main__":
    random_seed = 42
    np.random.seed(random_seed)
    main(random_state=random_seed)