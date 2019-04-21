from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from RMDL import plot as plt

def report_score(y_test, y_pred, scores, sparse_categorical=True, plot=False):
    if not sparse_categorical:
        y_test = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
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

    print(f"Accuracy of each individual model of the {len(scores)} models: {scores}")
    print(f"Overall Accuracy: {accuracy}")
    print(f"Overall Micro Metrics: {micro_metrics}")
    print(f"Overall Macro Metrics: {macro_metrics}")
    print(f"Overall Weighted Metrics: {weighted_metrics}")
