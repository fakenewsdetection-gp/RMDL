import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from RMDL import plot as plt


def print_precision_recall_fscore_support(metrics, average):
    print(f"\nOverall {average} Metrics:\n")
    print(f"\tPrecision: {metrics[0]}\n")
    print(f"\tRecall: {metrics[1]}\n")
    print(f"\tF-measure: {metrics[2]}\n")
    print(f"\tSupport: {metrics[3]}\n")


def report_score(y_test, y_pred, accuracies, sparse_categorical=True, plot=False):
    if not sparse_categorical:
        y_test = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    binary_metrics = precision_recall_fscore_support(y_test, y_pred, average='binary')
    micro_metrics = precision_recall_fscore_support(y_test, y_pred, average='micro')
    macro_metrics = precision_recall_fscore_support(y_test, y_pred, average='macro')
    weighted_metrics = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    if plot:
        classes = list(range(np.max(y_test) + 1))
        plt.plot_confusion_matrix(conf_matrix, classes=classes,
                                    title="Non-Normalized Confusion Matrix")
        plt.plot_confusion_matrix(conf_matrix, classes=classes, normalize=True,
                                    title="Normalized Confusion Matrix")

    print(f"Accuracy of each individual model of the {len(accuracies)} models: {accuracies}\n")
    print(f"Overall Accuracy: {accuracy}\n")
    print_precision_recall_fscore_support(binary_metrics, "Binary")
    print_precision_recall_fscore_support(micro_metrics, "Micro")
    print_precision_recall_fscore_support(macro_metrics, "Macro")
    print_precision_recall_fscore_support(weighted_metrics, "Weighted")
