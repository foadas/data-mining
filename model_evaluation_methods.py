from sklearn.metrics import roc_curve, auc, classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np

def calculate_error_rate(conf_matrix):
    total_samples = conf_matrix.sum()
    correct_predictions = conf_matrix.trace()
    error_rate = 1 - (correct_predictions / total_samples)
    return error_rate


def calculate_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    sensitivity = report['weighted avg']['recall']
    precision = report['weighted avg']['precision']
    f_measure = report['weighted avg']['f1-score']
    return sensitivity, precision, f_measure

def plot_roc_curve_multiclass(y_true, y_pred_proba, labels, modelName):
    plt.figure(figsize=(8, 6))
    for i in range(len(labels)):
        fpr, tpr, _ = roc_curve(y_true == labels[i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {labels[i]} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Multiclass - {modelName}')
    plt.legend(loc='lower right')
    plt.show()

def plot_learning_curve(estimator, title, X, y, cv=5, n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='accuracy'
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.grid()
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    return plt