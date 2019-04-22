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

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gc
import os
import numpy as np
import collections

from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from RMDL import BuildModel as BuildModel
from RMDL.Download import Download_Glove as GloVe
from RMDL import text_feature_extraction as txt
from RMDL import util
from RMDL import plot as plt
from RMDL import score


def evaluate_model(x_test, y_test, model_filepath, weights_filepath,
                    batch_size=128, sparse_categorical=True):
    with open(model_filepath, "r") as model_file:
        model = model_from_json(model_file.read())
    model.load_weights(weights_filepath)

    if sparse_categorical:
        y_pred = model.predict_classes(x_test, batch_size=batch_size)
        y_pred = np.array(y_pred)
        acc_score = accuracy_score(y_test, y_pred)
    else:
        y_pred = model.predict(x_test, batch_size=batch_size)
        y_pred = np.argmax(y_pred, axis=1)
        acc_score = accuracy_score(np.argmax(y_test, axis=1), y_pred)
    return y_pred, acc_score


def train(x_train, y_train, x_val,  y_val, batch_size=128,
            embedding_dim=50, max_seq_len=500, max_num_words=75000,
            glove_dir="", glove_file="glove.6B.50d.txt",
            sparse_categorical=True, random_deep=[3, 3, 3], epochs=[500, 500, 500], plot=False,
            min_hidden_layer_dnn=1, max_hidden_layer_dnn=8, min_nodes_dnn=128, max_nodes_dnn=1024,
            min_hidden_layer_rnn=1, max_hidden_layer_rnn=5, min_nodes_rnn=32,  max_nodes_rnn=128,
            min_hidden_layer_cnn=3, max_hidden_layer_cnn=10, min_nodes_cnn=128, max_nodes_cnn=512,
            random_state=42, random_optimizor=True, dropout=0.5, no_of_classes=0):
    """
    train(x_train, y_train, x_val,  y_val, batch_size=128,
            embedding_dim=50, max_seq_len=500, max_num_words=75000,
            glove_dir="", glove_file="glove.6B.50d.txt",
            sparse_categorical=True, random_deep=[3, 3, 3], epochs=[500, 500, 500], plot=False,
            min_hidden_layer_dnn=1, max_hidden_layer_dnn=8, min_nodes_dnn=128, max_nodes_dnn=1024,
            min_hidden_layer_rnn=1, max_hidden_layer_rnn=5, min_nodes_rnn=32,  max_nodes_rnn=128,
            min_hidden_layer_cnn=3, max_hidden_layer_cnn=10, min_nodes_cnn=128, max_nodes_cnn=512,
            random_state=42, random_optimizor=True, dropout=0.5, no_of_classes=0)

        Parameters
        ----------
            batch_size: int, optional
                Number of samples per gradient update. It will default to 128.
            embedding_dim: int, optional
                Dimensionality of the vector representation (word embedding) of each token in the corpus.
                It will default to 50.
            max_seq_len: int, optional
                Maximum number of words in a text to consider. It will default to 500.
            max_num_words: int, optional
                Maximum number of unique words in datasets. It will default to 75000.
            glove_dir: string, optional
                Path to GloVe or any pre-trained word embedding directory. It will default to the current
                directory where glove.6B.zip should be downloaded.
            glove_file: string, optional
                Which version of GloVe or any pre-trained word embedding will be used. It will default to glove.6B.50d.txt.
                NOTE: If you use other version of GloVe embedding_dim must be the same dimensions.
            sparse_categorical: bool
                When target's dataset is (n,1) should be True. It will default to True.
            random_deep: array of int [3], optional
                Number of ensembled models used in RMDL random_deep[0] is number of DNNs,
                random_deep[1] is number of RNNs, random_deep[2] is number of CNNs. It will default to [3, 3, 3].
            epochs: array of int [3], optional
                Number of epochs in each ensembled model used in RMDL epochs[0] is number of epochs used in DNNs,
                epochs[1] is number of epochs used in RNNs, epochs[0] is number of epochs used in CNNs. It will default to [500, 500, 500].
            plot: bool, optional
                Plot accuracies and losses of training and validation.
            min_hidden_layer_dnn: int, optional
                Lower Bounds of hidden layers of DNN used in RMDL. It will default to 1.
            max_hidden_layer_dnn: int, optional
                Upper bounds of hidden layers of DNN used in RMDL. It will default to 8.
            min_nodes_dnn: int, optional
                Lower bounds of nodes in each layer of DNN used in RMDL. It will default to 128.
            max_nodes_dnn: int, optional
                Upper bounds of nodes in each layer of DNN used in RMDL. It will default to 1024.
            min_hidden_layer_rnn: int, optional
                Lower Bounds of hidden layers of RNN used in RMDL. It will default to 1.
            min_hidden_layer_rnn: int, optional
                Upper Bounds of hidden layers of RNN used in RMDL. It will default to 5.
            min_nodes_rnn: int, optional
                Lower bounds of nodes (LSTM or GRU) in each layer of RNN used in RMDL. It will default to 32.
            max_nodes_rnn: int, optional
                Upper bounds of nodes (LSTM or GRU) in each layer of RNN used in RMDL. It will default to 128.
            min_hidden_layer_cnn: int, optional
                Lower Bounds of hidden layers of CNN used in RMDL. It will default to 3.
            max_hidden_layer_cnn: int, optional
                Upper Bounds of hidden layers of CNN used in RMDL. It will default to 10.
            min_nodes_cnn: int, optional
                Lower bounds of nodes (2D convolution layer) in each layer of CNN used in RMDL. It will default to 128.
            min_nodes_cnn: int, optional
                Upper bounds of nodes (2D convolution layer) in each layer of CNN used in RMDL. It will default to 512.
            random_state: int, optional
                RandomState instance or None, optional (default=None)
                If Integer, random_state is the seed used by the random number generator;
            random_optimizor: bool, optional
                If False, all models use adam optimizer. If True, all models use random optimizers. It will default to True
            dropout: float, optional
                between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.
            no_of_classes: int, optional
                If 0, number of classes is induced from the given labels.
                Else number of classes will be set to the given no_of_classes parameter.
    """
    np.random.seed(random_state)

    models_dir = "models"
    weights_dir = "weights"

    util.setup()

    history = []

    if not isinstance(y_train[0], list) and not isinstance(y_train[0], np.ndarray) \
        and not sparse_categorical:
        #checking if labels are one hot or not otherwise dense_layer will give shape error
        print("convert labels into one hot encoded representation")
        y_train = txt.get_one_hot_values(y_train)
        y_val = txt.get_one_hot_values(y_val)

    glove_needed = random_deep[1] != 0 or random_deep[2] != 0
    if glove_needed:
        if glove_dir == "":
            glove_dir = GloVe.download_and_extract()
            glove_filepath = os.path.join(glove_dir, glove_file)
        else:
            glove_filepath = os.path.join(glove_dir, glove_file)

        if not os.path.isfile(glove_filepath):
            print(f"Could not find {GloVe} Set GloVe Directory in Global.py")
            exit()

    all_text = np.concatenate((x_train, x_val))
    if random_deep[0] != 0:
        all_text_tf_idf = txt.get_tf_idf_vectors(all_text, max_num_words=max_num_words)
        x_train_tf_idf = all_text_tf_idf[:len(x_train), ]
        x_val_tf_idf = all_text_tf_idf[len(x_train):, ]
    if random_deep[1] != 0 or random_deep[2] != 0:
        print(glove_filepath)
        all_text_tokenized, word_index = txt.tokenize(all_text,
                                                        max_num_words=max_num_words,
                                                        max_seq_len=max_seq_len)
        x_train_tokenized = all_text_tokenized[:len(x_train), ]
        x_val_tokenized = all_text_tokenized[len(x_train):, ]
        embeddings_index = txt.get_word_embeddings_index(glove_filepath)

    del x_train
    del x_val
    gc.collect()

    if no_of_classes==0:
        #checking no_of_classes
        #np.max(data)+1 will not work for one_hot encoding labels
        if sparse_categorical:
            number_of_classes = np.max(y_train) + 1
        else:
            number_of_classes = len(y_train[0])
    else:
        number_of_classes = no_of_classes

    i = 0
    while i < random_deep[0]:
        try:
            print(f"\nBuilding and Training DNN-{i}")
            model_file = f"DNN_{i}.json"
            weights_file = f"DNN_{i}.hdf5"
            checkpoint = ModelCheckpoint(os.path.join(weights_dir, weights_file),
                                            monitor='val_acc',
                                            verbose=1,
                                            save_best_only=True,
                                            mode='max')
            model_DNN, _ = BuildModel.Build_Model_DNN_Text(x_train_tf_idf.shape[1],
                                                            number_of_classes,
                                                            sparse_categorical,
                                                            min_hidden_layer_dnn,
                                                            max_hidden_layer_dnn,
                                                            min_nodes_dnn,
                                                            max_nodes_dnn,
                                                            random_optimizor,
                                                            dropout)
            model_json = model_DNN.to_json()
            with open(os.path.join(models_dir, model_file), "w") as model_DNN_file:
                model_DNN_file.write(model_json)
            model_history = model_DNN.fit(x_train_tf_idf, y_train,
                                            validation_data=(x_val_tf_idf, y_val),
                                            epochs=epochs[0],
                                            batch_size=batch_size,
                                            callbacks=[checkpoint],
                                            verbose=2)
            history.append(model_history)
            i += 1
            del model_DNN
            gc.collect()
        except Exception as e:
            print(f"Check the Error \n {e}")
            print(f"Error in DNN-{i} model trying to re-generate another model")
            if max_hidden_layer_dnn > 3:
                max_hidden_layer_dnn -= 1
            if max_nodes_dnn > 256:
                max_nodes_dnn -= 8

    del x_train_tf_idf
    del x_val_tf_idf
    gc.collect()

    i = 0
    while i < random_deep[1]:
        try:
            print(f"\nBuilding and Training RNN-{i}")
            model_file = f"RNN_{i}.json"
            weights_file = f"RNN_{i}.hdf5"
            checkpoint = ModelCheckpoint(os.path.join(weights_dir, weights_file),
                                            monitor='val_acc',
                                            verbose=1,
                                            save_best_only=True,
                                            mode='max')
            model_RNN, _ = BuildModel.Build_Model_RNN_Text(word_index,
                                                            embeddings_index,
                                                            number_of_classes,
                                                            max_seq_len,
                                                            embedding_dim,
                                                            sparse_categorical,
                                                            min_hidden_layer_rnn,
                                                            max_hidden_layer_rnn,
                                                            min_nodes_rnn,
                                                            max_nodes_rnn,
                                                            random_optimizor,
                                                            dropout)
            model_json = model_RNN.to_json()
            with open(os.path.join(models_dir, model_file), "w") as model_RNN_file:
                model_RNN_file.write(model_json)
            model_history = model_RNN.fit(x_train_tokenized, y_train,
                                            validation_data=(x_val_tokenized, y_val),
                                            epochs=epochs[1],
                                            batch_size=batch_size,
                                            callbacks=[checkpoint],
                                            verbose=2)
            history.append(model_history)
            i += 1
            del model_RNN
            gc.collect()
        except:
            print(f"Check the Error \n {e}")
            print(f"Error in RNN-{i} model trying to re-generate another model")
            if max_hidden_layer_rnn > 3:
                max_hidden_layer_rnn -= 1
            if max_nodes_rnn > 64:
                max_nodes_rnn -= 2

    gc.collect()

    i = 0
    while i < random_deep[2]:
        try:
            print(f"\nBuilding and Training CNN-{i}")
            model_file = f"CNN_{i}.json"
            weights_file = f"CNN_{i}.hdf5"
            checkpoint = ModelCheckpoint(os.path.join(weights_dir, weights_file),
                                            monitor='val_acc',
                                            verbose=1,
                                            save_best_only=True,
                                            mode='max')
            model_CNN, _ = BuildModel.Build_Model_CNN_Text(word_index,
                                                            embeddings_index,
                                                            number_of_classes,
                                                            max_seq_len,
                                                            embedding_dim,
                                                            sparse_categorical,
                                                            min_hidden_layer_cnn,
                                                            max_hidden_layer_cnn,
                                                            min_nodes_cnn,
                                                            max_nodes_cnn,
                                                            random_optimizor,
                                                            dropout)
            model_json = model_CNN.to_json()
            with open(os.path.join(models_dir, model_file), "w") as model_CNN_file:
                model_CNN_file.write(model_json)
            model_history = model_CNN.fit(x_train_tokenized, y_train,
                                            validation_data=(x_val_tokenized, y_val),
                                            epochs=epochs[2],
                                            batch_size=batch_size,
                                            callbacks=[checkpoint],
                                            verbose=2)
            history.append(model_history)
            i += 1
            del model_CNN
            gc.collect()
        except:
            print(f"Check the Error \n {e}")
            print(f"Error in CNN-{i} model trying to re-generate another model")
            if max_hidden_layer_cnn > 5:
                max_hidden_layer_cnn -= 1
            if max_nodes_cnn > 128:
                max_nodes_cnn -= 2
                min_nodes_cnn -= 1

    if plot:
        plt.plot_history(history)


def predict_evaluate(x_test, y_test, batch_size=128, max_seq_len=500, max_num_words=75000,
                        sparse_categorical=True, random_deep=[3, 3, 3], plot=False, models_dir="models",
                        weights_dir="weights", tf_idf_vectorizer_filepath="tf_idf_vectorizer.pickle",
                        text_tokenizer_filepath="text_tokenizer.pickle"):
    """
    predict_evaluate(x_test, y_test, batch_size=128, max_seq_len=500, max_num_words=75000,
                        sparse_categorical=True, random_deep=[3, 3, 3], plot=False, models_dir="models",
                        weights_dir="weights", tf_idf_vectorizer_filepath="tf_idf_vectorizer.pickle",
                        text_tokenizer_filepath="text_tokenizer.pickle")

    Parameters
    ----------
        batch_size: int, optional
            Number of samples per gradient update. It will default to 128.
        max_seq_len: int, optional
            Maximum number of words in a text to consider. It will default to 500.
        max_num_words: int, optional
            Maximum number of unique words in datasets. It will default to 75000.
        sparse_categorical: bool
            When target's dataset is (n,1) should be True. It will default to True.
        random_deep: array of int [3], optional
            Number of ensembled models used in RMDL random_deep[0] is number of DNNs,
            random_deep[1] is number of RNNs, random_deep[2] is number of CNNs. It will default to [3, 3, 3].
        plot: bool, optional
            Plot confusion matrices(non-normalized and normalized).
        models_dir: string, optional
            Path to the directory where the pre-trained models are saved. It will default to "models".
        weights_dir: string, optional
            Path to the directory where the weights of the pre-trained models are saved. It will default to "weights".
        tf_idf_vectorizer_filepath: string, optional
            Path to tf-idf vectorizer used in preprocessing while training RMDL. It will default to "tf_idf_vectorizer.pickle".
        text_tokenizer_filepath: string, optional
            Path to text tokenizer used in preprocessing while training RMDL. It will default to "text_tokenizer.pickle".
    """

    all_y_pred = []
    accuracies = []

    if not isinstance(y_test[0], list) and not isinstance(y_test[0], np.ndarray) \
        and not sparse_categorical:
        #checking if labels are one hot or not otherwise dense_layer will give shape error
        print("convert labels into one hot encoded representation")
        y_test = txt.get_one_hot_values(y_test)

    if random_deep[0] != 0:
        x_test_tf_idf = txt.get_tf_idf_vectors(x_test,
                                                max_num_words=max_num_words,
                                                fit=False,
                                                vectorizer_filepath=tf_idf_vectorizer_filepath)
    if random_deep[1] != 0 or random_deep[2] != 0:
        x_test_tokenized, _ = txt.tokenize(x_test,
                                            max_num_words=max_num_words,
                                            max_seq_len=max_seq_len,
                                            fit=False,
                                            tokenizer_filepath=text_tokenizer_filepath)

    del x_test
    gc.collect()

    for i in range(len(random_deep)):
        for j in range(random_deep[i]):
            try:
                print(f"\nEvaluating {util.model_type[i]}-{j}")
                model_file = f"{util.model_type[i]}_{j}.json"
                weights_file = f"{util.model_type[i]}_{j}.hdf5"
                model_filepath = os.path.join(models_dir, model_file)
                weights_filepath = os.path.join(weights_dir, weights_file)
                if i == 0:
                    x_test = x_test_tf_idf
                else:
                    x_test = x_test_tokenized
                y_pred, accuracy = evaluate_model(x_test, y_test,
                                                    model_filepath, weights_filepath,
                                                    batch_size=batch_size,
                                                    sparse_categorical=sparse_categorical)
                print(f"Accuracy of {util.model_type[i]}-{j}: {accuracy}")
                all_y_pred.append(y_pred)
                accuracies.append(accuracy)
            except Exception as e:
                print(f"Error in {util.model_type[i]}-{j}\n")
                print(f"Check the Error \n {e}")

    del x_test
    del x_test_tf_idf
    del x_test_tokenized
    gc.collect()

    y_probs = np.array(all_y_pred).transpose()
    y_pred = []
    for i in range(y_probs.shape[0]):
        sample_pred = np.array(y_probs[i, :])
        sample_pred = collections.Counter(sample_pred).most_common()[0][0]
        y_pred.append(sample_pred)

    score.report_score(y_test, y_pred, accuracies, sparse_categorical=sparse_categorical, plot=plot)
    return y_pred