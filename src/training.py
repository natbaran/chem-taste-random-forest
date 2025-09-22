import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import os


def run_pca_scatter(X, y, save_path=None):
    """Plots 2D PCA scatter plot of molecular fingerprints."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(8,6))
    for taste in np.unique(y):
        idx = y == taste
        plt.scatter(X_pca[idx,0], X_pca[idx,1], label=taste, alpha=0.7)
    plt.legend()
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of molecular embeddings")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"✅ PCA 2D scatter saved at: {save_path}")
    
    plt.show()

def plot_pca_variance(X, n_components=10, save_path=None):
    """Plots explained variance of the first n_components principal components."""
    pca = PCA(n_components=n_components)
    pca.fit(X)
    explained_variance = pca.explained_variance_ratio_

    plt.figure(figsize=(8,5))
    plt.bar(range(1, n_components+1), explained_variance*100, color='skyblue')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance (%)')
    plt.title(f'Variance explained by the first {n_components} principal components')
    plt.xticks(range(1, n_components+1))

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"✅ PCA variance bar plot saved at: {save_path}")
    
    plt.show()

    for i, var in enumerate(explained_variance, 1):
        print(f"PC{i}: {var*100:.2f}%")


def crossval_random_forest(X, y, classes, n_splits=5):
    """Performs cross-validation with Random Forest + plots ROC curves."""
    mean_fpr = np.linspace(0, 1, 100)
    aucs, tprs = {c: [] for c in classes}, {c: [] for c in classes}

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(kf.split(X,y),1):
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])
        # Print accuracy for each fold
        print(f"Fold {fold} accuracy:", accuracy_score(y[test_idx], y_pred))
