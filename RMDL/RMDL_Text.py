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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from keras.callbacks import ModelCheckpoint
from RMDL import BuildModel as BuildModel
from RMDL.Download import Download_Glove as GloVe
from RMDL import text_feature_extraction as txt
from RMDL import Global as G
from RMDL import Plot as Plot


def train(x_train, y_train, x_val,  y_val, batch_size=128,
            embedding_dim=50, max_sequence_length=500, max_nb_words=75000,
            glove_dir="", glove_file="glove.6B.50d.txt",
            sparse_categorical=True, random_deep=[3, 3, 3], epochs=[500, 500, 500], plot=False,
            min_hidden_layer_dnn=1, max_hidden_layer_dnn=8, min_nodes_dnn=128, max_nodes_dnn=1024,
            min_hidden_layer_rnn=1, max_hidden_layer_rnn=5, min_nodes_rnn=32,  max_nodes_rnn=128,
            min_hidden_layer_cnn=3, max_hidden_layer_cnn=10, min_nodes_cnn=128, max_nodes_cnn=512,
            random_state=42, random_optimizor=True, dropout=0.5, no_of_classes=0):
    """
    train(x_train, y_train, x_val,  y_val, batch_size=128,
            embedding_dim=50, max_sequence_length=500, max_nb_words=75000,
            glove_dir="", glove_file="glove.6B.50d.txt",
            sparse_categorical=True, random_deep=[3, 3, 3], epochs=[500, 500, 500], plot=False,
            min_hidden_layer_dnn=1, max_hidden_layer_dnn=8, min_nodes_dnn=128, max_nodes_dnn=1024,
            min_hidden_layer_rnn=1, max_hidden_layer_rnn=5, min_nodes_rnn=32,  max_nodes_rnn=128,
            min_hidden_layer_cnn=3, max_hidden_layer_cnn=10, min_nodes_cnn=128, max_nodes_cnn=512,
            random_state=42, random_optimizor=True, dropout=0.5, no_of_classes=0):

        Parameters
        ----------
            batch_size: Integer, , optional
                Number of samples per gradient update. If unspecified, it will default to 128
            MAX_NB_WORDS: int, optional
                Maximum number of unique words in datasets, it will default to 75000.
            GloVe_dir: String, optional
                Address of GloVe or any pre-trained directory, it will default to null which glove.6B.zip will be download.
            GloVe_dir: String, optional
                Which version of GloVe or pre-trained word emending will be used, it will default to glove.6B.50d.txt.
                NOTE: if you use other version of GloVe EMBEDDING_DIM must be same dimensions.
            sparse_categorical: bool.
                When target's dataset is (n,1) should be True, it will default to True.
            random_deep: array of int [3], optional
                Number of ensembled model used in RMDL random_deep[0] is number of DNN, random_deep[1] is number of RNN, random_deep[0] is number of CNN, it will default to [3, 3, 3].
            epochs: array of int [3], optional
                Number of epochs in each ensembled model used in RMDL epochs[0] is number of epochs used in DNN, epochs[1] is number of epochs used in RNN, epochs[0] is number of epochs used in CNN, it will default to [500, 500, 500].
            plot: bool, optional
                True: shows confusion matrix and accuracy and loss
            min_hidden_layer_dnn: Integer, optional
                Lower Bounds of hidden layers of DNN used in RMDL, it will default to 1.
            max_hidden_layer_dnn: Integer, optional
                Upper bounds of hidden layers of DNN used in RMDL, it will default to 8.
            min_nodes_dnn: Integer, optional
                Lower bounds of nodes in each layer of DNN used in RMDL, it will default to 128.
            max_nodes_dnn: Integer, optional
                Upper bounds of nodes in each layer of DNN used in RMDL, it will default to 1024.
            min_hidden_layer_rnn: Integer, optional
                Lower Bounds of hidden layers of RNN used in RMDL, it will default to 1.
            min_hidden_layer_rnn: Integer, optional
                Upper Bounds of hidden layers of RNN used in RMDL, it will default to 5.
            min_nodes_rnn: Integer, optional
                Lower bounds of nodes (LSTM or GRU) in each layer of RNN used in RMDL, it will default to 32.
            max_nodes_rnn: Integer, optional
                Upper bounds of nodes (LSTM or GRU) in each layer of RNN used in RMDL, it will default to 128.
            min_hidden_layer_cnn: Integer, optional
                Lower Bounds of hidden layers of CNN used in RMDL, it will default to 3.
            max_hidden_layer_cnn: Integer, optional
                Upper Bounds of hidden layers of CNN used in RMDL, it will default to 10.
            min_nodes_cnn: Integer, optional
                Lower bounds of nodes (2D convolution layer) in each layer of CNN used in RMDL, it will default to 128.
            min_nodes_cnn: Integer, optional
                Upper bounds of nodes (2D convolution layer) in each layer of CNN used in RMDL, it will default to 512.
            random_state: Integer, optional
                RandomState instance or None, optional (default=None)
                If Integer, random_state is the seed used by the random number generator;
            random_optimizor: bool, optional
                If False, all models use adam optimizer. If True, all models use random optimizers. it will default to True
            dropout: Float, optional
                between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.
            no_of_classes: Integer, optional
                If 0, number of classes is induced from the given labels.
                Else number of classes will be set to the given no_of_classes parameter.
    """
    np.random.seed(random_state)

    models_dir = "models"
    weights_dir = "weights"

    history = []

    glove_needed = random_deep[1] != 0 or random_deep[2] != 0

    def one_hot_encoder(value, label_data_):
        label_data_[value] = 1
        return label_data_

    def get_one_hot_values(labels_data):
        encoded = [0] * len(labels_data)
        for index_no, value in enumerate(labels_data):
            max_value = [0] * (np.max(labels_data) + 1)
            encoded[index_no] = one_hot_encoder(value, max_value)
        return np.array(encoded)

    if not isinstance(y_train[0], list) and not isinstance(y_train[0], np.ndarray) and not sparse_categorical:
        #checking if labels are one hot or not otherwise dense_layer will give shape error
        print("converted_into_one_hot")
        y_train = get_one_hot_values(y_train)
        y_val = get_one_hot_values(y_val)

    if glove_needed:
        if glove_dir == "":
            glove_dir = GloVe.download_and_extract()
            glove_path = os.path.join(glove_dir, glove_file)
        else:
            glove_path = os.path.join(glove_dir, glove_file)

        if not os.path.isfile(glove_path):
            print("Could not find %s Set GloVe Directory in Global.py ", GloVe)
            exit()

    G.setup()
    if random_deep[0] != 0:
        x_train_tfidf, x_val_tfidf = txt.loadData(x_train, x_val, MAX_NB_WORDS=max_nb_words)
    if random_deep[1] != 0 or random_deep[2] != 0 :
        print(glove_path)
        x_train_embedded, x_val_embedded, word_index, embeddings_index = txt.loadData_Tokenizer(x_train, x_val,
                                                                                                glove_path, max_nb_words,
                                                                                                max_sequence_length, embedding_dim)
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
            print("Building and Training DNN-%d", i)
            model_filepath = "DNN_" + str(i) + ".json"
            weights_filepath = "DNN_" + str(i) + ".hdf5"
            checkpoint = ModelCheckpoint(os.path.join(weights_dir, weights_filepath),
                                            monitor='val_acc',
                                            verbose=1,
                                            save_best_only=True,
                                            mode='max')
            model_DNN, _ = BuildModel.Build_Model_DNN_Text(x_train_tfidf.shape[1],
                                                                number_of_classes,
                                                                sparse_categorical,
                                                                min_hidden_layer_dnn,
                                                                max_hidden_layer_dnn,
                                                                min_nodes_dnn,
                                                                max_nodes_dnn,
                                                                random_optimizor,
                                                                dropout)
            model_json = model_DNN.to_json()
            with open(os.path.join(models_dir, model_filepath), "w") as model_json_file:
                model_json_file.write(model_json)
            model_history = model_DNN.fit(x_train_tfidf, y_train,
                                            validation_data=(x_val_tfidf, y_val),
                                            epochs=epochs[0],
                                            batch_size=batch_size,
                                            callbacks=[checkpoint],
                                            verbose=2)
            history.append(model_history)
            i += 1
            del model_DNN
        except Exception as e:
            print("Check the Error \n {} ".format(e))
            print("Error in model", i, "try to re-generate another model")
            if max_hidden_layer_dnn > 3:
                max_hidden_layer_dnn -= 1
            if max_nodes_dnn > 256:
                max_nodes_dnn -= 8
    del x_train_tfidf
    del x_val_tfidf
    gc.collect()

    i=0
    while i < random_deep[1]:
        try:
            print("Building and Training RNN-%d", i)
            model_filepath = "RNN_" + str(i) + ".json"
            weights_filepath = "RNN_" + str(i) + ".hdf5"
            checkpoint = ModelCheckpoint(os.path.join(weights_dir, weights_filepath),
                                            monitor='val_acc',
                                            verbose=1,
                                            save_best_only=True,
                                            mode='max')
            model_RNN, _ = BuildModel.Build_Model_RNN_Text(word_index,
                                                                embeddings_index,
                                                                number_of_classes,
                                                                max_sequence_length,
                                                                embedding_dim,
                                                                sparse_categorical,
                                                                min_hidden_layer_rnn,
                                                                max_hidden_layer_rnn,
                                                                min_nodes_rnn,
                                                                max_nodes_rnn,
                                                                random_optimizor,
                                                                dropout)
            model_json = model_RNN.to_json()
            with open(os.path.join(models_dir, model_filepath), "w") as model_json_file:
                model_json_file.write(model_json)
            model_history = model_RNN.fit(x_train_embedded, y_train,
                                            validation_data=(x_val_embedded, y_val),
                                            epochs=epochs[1],
                                            batch_size=batch_size,
                                            callbacks=[checkpoint],
                                            verbose=2)
            history.append(model_history)
            i += 1
            del model_RNN
            gc.collect()
        except:
            print("Error in model", i, "try to re-generate another model")
            if max_hidden_layer_rnn > 3:
                max_hidden_layer_rnn -= 1
            if max_nodes_rnn > 64:
                max_nodes_rnn -= 2
    gc.collect()

    i = 0
    while i < random_deep[2]:
        try:
            print("Building and Training CNN-%d", i)
            model_filepath = "CNN_" + str(i) + ".json"
            weights_filepath = "CNN_" + str(i) + ".hdf5"
            checkpoint = ModelCheckpoint(os.path.join(weights_dir, weights_filepath),
                                            monitor='val_acc',
                                            verbose=1,
                                            save_best_only=True,
                                            mode='max')
            model_CNN, _ = BuildModel.Build_Model_CNN_Text(word_index,
                                                                embeddings_index,
                                                                number_of_classes,
                                                                max_sequence_length,
                                                                embedding_dim,
                                                                sparse_categorical,
                                                                min_hidden_layer_cnn,
                                                                max_hidden_layer_cnn,
                                                                min_nodes_cnn,
                                                                max_nodes_cnn,
                                                                random_optimizor,
                                                                dropout)
            model_json = model_CNN.to_json()
            with open(os.path.join(models_dir, model_filepath), "w") as model_json_file:
                model_json_file.write(model_json)
            model_history = model_CNN.fit(x_train_embedded, y_train,
                                            validation_data=(x_val_embedded, y_val),
                                            epochs=epochs[2],
                                            batch_size=batch_size,
                                            callbacks=[checkpoint],
                                            verbose=2)
            history.append(model_history)
            i += 1
            del model_CNN
            gc.collect()
        except:
            print("Error in model", i, "try to re-generate an other model")
            if max_hidden_layer_cnn > 5:
                max_hidden_layer_cnn -= 1
            if max_nodes_cnn > 128:
                max_nodes_cnn -= 2
                min_nodes_cnn -= 1
