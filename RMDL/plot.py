"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
RMDL: Random Multimodel Deep Learning for Classification

* Copyright (C) 2018  Kamran Kowsari <kk7nc@virginia.edu>
* Last Update: Oct 26, 2018
* This file is part of  RMDL project, University of Virginia.
* Free to use, change, share and distribute source code of RMDL
* Refrenced paper : RMDL: Random Multimodel Deep Learning for Classification
* Link: https://dl.acm.org/citation.cfm?id=3206111
* Refrenced paper : An Improvement of Data Classification using Random Multimodel Deep Learning (RMDL)
* Link :  http://www.ijmlc.org/index.php?m=content&c=index&a=show&catid=79&id=823
* Comments and Error: email: kk7nc@virginia.edu

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from pylab import *
import itertools


def plot_history(history):
    """Plots the training and validation history(accuracy and loss) of RMDL."""
    num_models = len(history)
    caption = []
    for i in range(len(history)):
        caption.append('RDL ' + str(i + 1))

    for i in range(num_models):
        plt.plot(history[i].history['loss'])
        plt.title('RMDL Training Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
    plt.legend(caption, loc='upper right')
    plt.show()

    for i in range(num_models):
        plt.plot(history[i].history['val_loss'])
        plt.title('RMDL Validation Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
    plt.legend(caption, loc='upper right')
    plt.show()

    for i in range(num_models):
        plt.plot(history[i].history['acc'])
        plt.title('RMDL Training Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
    plt.legend(caption, loc='upper right')
    plt.show()

    for i in range(num_models):
        plt.plot(history[i].history['val_acc'])
        plt.title('RMDL Validation Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
    plt.legend(caption, loc='upper right')
    plt.show()

    for i in range(num_models):
        plt.plot(history[i].history['prec'])
        plt.title('RMDL Training Precision')
        plt.ylabel('Precision')
        plt.xlabel('Epoch')
    plt.legend(caption, loc='upper right')
    plt.show()

    for i in range(num_models):
        plt.plot(history[i].history['val_prec'])
        plt.title('RMDL Validation Precision')
        plt.ylabel('Precision')
        plt.xlabel('Epoch')
    plt.legend(caption, loc='upper right')
    plt.show()

    for i in range(num_models):
        plt.plot(history[i].history['rec'])
        plt.title('RMDL Training Recall')
        plt.ylabel('Recall')
        plt.xlabel('Epoch')
    plt.legend(caption, loc='upper right')
    plt.show()

    for i in range(num_models):
        plt.plot(history[i].history['val_rec'])
        plt.title('RMDL Validation Recall')
        plt.ylabel('Recall')
        plt.xlabel('Epoch')
    plt.legend(caption, loc='upper right')
    plt.show()

    for i in range(num_models):
        plt.plot(history[i].history['true_pos'])
        plt.title('RMDL Training True Positives')
        plt.ylabel('True Positives')
        plt.xlabel('Epoch')
    plt.legend(caption, loc='upper right')
    plt.show()

    for i in range(num_models):
        plt.plot(history[i].history['val_true_pos'])
        plt.title('RMDL Validation True Positives')
        plt.ylabel('True Positives')
        plt.xlabel('Epoch')
    plt.legend(caption, loc='upper right')
    plt.show()

    for i in range(num_models):
        plt.plot(history[i].history['true_neg'])
        plt.title('RMDL Training True Negatives')
        plt.ylabel('True Negatives')
        plt.xlabel('Epoch')
    plt.legend(caption, loc='upper right')
    plt.show()

    for i in range(num_models):
        plt.plot(history[i].history['val_true_neg'])
        plt.title('RMDL Validation True Negatives')
        plt.ylabel('True Negatives')
        plt.xlabel('Epoch')
    plt.legend(caption, loc='upper right')
    plt.show()

    for i in range(num_models):
        plt.plot(history[i].history['false_pos'])
        plt.title('RMDL Training False Positives')
        plt.ylabel('False Positives')
        plt.xlabel('Epoch')
    plt.legend(caption, loc='upper right')
    plt.show()

    for i in range(num_models):
        plt.plot(history[i].history['val_false_pos'])
        plt.title('RMDL Validation False Positives')
        plt.ylabel('False Positives')
        plt.xlabel('Epoch')
    plt.legend(caption, loc='upper right')
    plt.show()

    for i in range(num_models):
        plt.plot(history[i].history['false_neg'])
        plt.title('RMDL Training False Negatives')
        plt.ylabel('False Negatives')
        plt.xlabel('Epoch')
    plt.legend(caption, loc='upper right')
    plt.show()

    for i in range(num_models):
        plt.plot(history[i].history['val_false_neg'])
        plt.title('RMDL Validation False Negatives')
        plt.ylabel('False Negatives')
        plt.xlabel('Epoch')
    plt.legend(caption, loc='upper right')
    plt.show()


def plot_confusion_matrix(cm, classes, normalize=False,
                            title='Confusion Matrix',
                            cmap=plt.cm.Blues):
    """
    Plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
