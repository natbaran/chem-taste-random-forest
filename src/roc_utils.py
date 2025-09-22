# roc_utils.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report

def plot_roc_cv(X, y, classes, n_splits=5, n_estimators=100):
    """
    Performs Stratified K-Fold CV and plots ROC-AUC curves per class.
    Returns average metrics per class.
    """
    mean_fpr = np.linspace(0, 1, 100)
    tprs, aucs = {cls: [] for cls in classes}, {cls: [] for cls in classes}
    reports = []

    y_bin = label_binarize(y, classes=classes)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Fold {fold} - Accuracy: {acc:.3f}")

        report = classification_report(y_test, y_pred, output_dict=True)
        reports.append(report)

        y_score = rf.predict_proba(X_test)

        for cls_idx, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[test_idx, cls_idx], y_score[:, cls_idx])
            roc_auc = auc(fpr, tpr)
            aucs[cls].append(roc_auc)

            tpr_interp = np.interp(mean_fpr, fpr, tpr)
            tpr_interp[0] = 0.0
            tprs[cls].append(tpr_interp)
            print(f'Fold {fold}, Class {cls} - AUC: {roc_auc:.3f}')

    # graph
    plt.figure(figsize=(7, 6))
    for cls in classes:
        mean_tpr = np.mean(tprs[cls], axis=0)
        std_tpr = np.std(tprs[cls], axis=0)
        mean_auc = np.mean(aucs[cls])
        std_auc = np.std(aucs[cls])

        plt.plot(mean_fpr, mean_tpr, label=f'{cls} (AUC={mean_auc:.2f}±{std_auc:.2f})')
        plt.fill_between(mean_fpr,
                         np.maximum(mean_tpr - std_tpr, 0),
                         np.minimum(mean_tpr + std_tpr, 1),
                         alpha=0.2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC curves with ±1 std over {n_splits} folds')
    plt.legend()
    plt.show()

    # average repports
    avg_report = {}
    for cls in report.keys():
        if cls not in ['accuracy', 'macro avg', 'weighted avg']:
            avg_precision = np.mean([r[cls]['precision'] for r in reports])
            avg_recall = np.mean([r[cls]['recall'] for r in reports])
            avg_f1 = np.mean([r[cls]['f1-score'] for r in reports])
            avg_report[cls] = {'precision': avg_precision, 'recall': avg_recall, 'f1-score': avg_f1}

    return avg_report
