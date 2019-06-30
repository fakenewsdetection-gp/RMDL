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

from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pickle


def transliterate(line):
    cedilla2latin = [
        [u'Á', u'A'],
        [u'á', u'a'],
        [u'Č', u'C'],
        [u'č', u'c'],
        [u'Š', u'S'],
        [u'š', u's']
    ]

    tr = dict([(a[0], a[1]) for (a) in cedilla2latin])
    new_line = ""
    for letter in line:
        if letter in tr:
            new_line += tr[letter]
        else:
            new_line += letter
    return new_line


def text_cleaner(text, deep_clean=False, stem=True, stop_words=True, translite_rate=True):
    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        # newline after </p> and </div> and <h1/>...
        {r'</(div)\s*>\s*': u'\n'},
        # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        # show links instead of texts
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning
    ]

    if deep_clean:
        text = text.replace(".", "")
        text = text.replace("[", " ")
        text = text.replace(",", " ")
        text = text.replace("]", " ")
        text = text.replace("(", " ")
        text = text.replace(")", " ")
        text = text.replace("\"", "")
        text = text.replace("-", " ")
        text = text.replace("=", " ")
        text = text.replace("?", " ")
        text = text.replace("!", " ")

        for rule in rules:
            for (k, v) in rule.items():
                regex = re.compile(k)
                text = regex.sub(v, text)
            text = text.rstrip()
            text = text.strip()
        text = text.replace('+', ' ').replace('.',
                                              ' ').replace(',', ' ').replace(':', ' ')
        text = re.sub("(^|\W)\d+($|\W)", " ", text)
        if translite_rate:
            text = transliterate(text)
        if stem:
            text = PorterStemmer().stem(text)
        text = WordNetLemmatizer().lemmatize(text)
        if stop_words:
            stop_words = set(stopwords.words('english'))
            word_tokens = word_tokenize(text)
            text = [w for w in word_tokens if not w in stop_words]
            text = ' '.join(str(e) for e in text)
    else:
        for rule in rules:
            for (k, v) in rule.items():
                regex = re.compile(k)
                text = regex.sub(v, text)
            text = text.rstrip()
            text = text.strip()
    return text.lower()


def get_one_hot_values(labels):
    encoded = [0] * len(labels)
    for index_no, value in enumerate(labels):
        max_value = [0] * (np.max(labels) + 1)
        max_value[value] = 1
        encoded[index_no] = max_value
    return np.array(encoded)


def tokenize(text, max_num_words=75000, max_seq_len=500, fit=True, tokenizer=None, tokenizer_filepath=None):
    np.random.seed(7)
    if fit:
        tokenizer = Tokenizer(num_words=max_num_words)
        tokenizer.fit_on_texts(text)
        with open("text_tokenizer.pickle", "wb") as text_tokenizer_file:
            pickle.dump(tokenizer, text_tokenizer_file)
    else:
        if tokenizer is None and tokenizer_filepath is not None:
            with open(tokenizer_filepath, "rb") as text_tokenizer_file:
                tokenizer = pickle.load(text_tokenizer_file)
        elif tokenizer is None:
            raise Exception("Pickle file for text tokenizer is not specified.")
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(text)
    text_tokenized = pad_sequences(sequences, maxlen=max_seq_len)
    return text_tokenized, word_index


def get_word_embedding_index(glove_filepath, word_index):
    embedding_index = {}
    with open(glove_filepath, encoding="utf8") as glove_file:
        for line in glove_file:
            values = line.split()
            word = values[0]
            try:
                float(values[1])
            except:
                continue
            if word == 'UNK' or word in word_index:
                embedding_index[word] = np.array([float(n) for n in values[1:]], dtype=float)
    if 'UNK' not in embedding_index:
        embedding_vecs = np.array(list(embedding_index.values()), dtype=float)
        embedding_index['UNK'] = np.mean(embedding_vecs, axis=0)
    return embedding_index


def get_tf_idf_vectors(text, max_num_words=75000, fit=True, vectorizer=None, vectorizer_filepath=None):
    if fit:
        vectorizer = TfidfVectorizer(max_features=max_num_words)
        text_tf_idf = vectorizer.fit_transform(text).toarray()
        with open("tf_idf_vectorizer.pickle", "wb") as tf_idf_vectorizer_file:
            pickle.dump(vectorizer, tf_idf_vectorizer_file)
    else:            
        if vectorizer is None and vectorizer_filepath is not None:
            vectorizer = None
            with open(vectorizer_filepath, "rb") as tf_idf_vectorizer_file:
                vectorizer = pickle.load(tf_idf_vectorizer_file)
        elif vectorizer is None:
            raise Exception("Pickle file for tf-idf vectorizer is not specified.")
        text_tf_idf = vectorizer.transform(text).toarray()
    return text_tf_idf
