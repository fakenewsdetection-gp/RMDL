import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from RMDL import plot as plt


def report_score(y_true, y_pred, models_y_pred=None, plot=False):
    if models_y_pred is not None:
        for model_name, model_y_pred in models_y_pred.items():
            print(f"\n\nClassification Report of {model_name}:\n")
            print(f"Accuracy: {accuracy_score(y_true, model_y_pred)}\n")
            classification_report(y_true, model_y_pred)
    print(f"\n\n\nClassification Report of RMDL as a whole:\n")
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    classification_report(y_true, y_pred)

    if plot:
        conf_matrix = confusion_matrix(y_true, y_pred)
        classes = list(range(np.max(y_test) + 1))
        plt.plot_confusion_matrix(conf_matrix, classes=classes,
                                    title="Non-Normalized Confusion Matrix")
        plt.plot_confusion_matrix(conf_matrix, classes=classes, normalize=True,
                                    title="Normalized Confusion Matrix")
