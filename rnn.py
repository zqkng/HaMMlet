###############################################################################
# rnn.py
# ------
# Build and train LSTM (long short-term memory) recurrent neural network (RNN).
#
###############################################################################

import os
os.environ["KERAS_BACKEND"] = "torch"
import pickle

import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Lambda, Dropout
from keras.utils import to_categorical

import tokenizer


MODEL_DIRNAME = "models/rnn"
MAX_SEQUENCE_LENGTH = 40
STEP_SIZE = 1


def load_rnn_data():
    data = tokenizer.load_data()
    sonnets = tokenizer.sequence_full_sonnet(tokenizer.tokenize_lpunc, data)
    sequences = _process_sequences(sonnets)
    char2vec = _map_characters(sonnets)
    
    char_sequences = []
    for seq in sequences:
        char_sequences.append([char2vec[c] for c in seq])

    return char_sequences, char2vec
    

def _process_sequences(data, max_len=MAX_SEQUENCE_LENGTH, step=STEP_SIZE):
    sequences = []
    for sonnet in data:
        for i in range(0, len(sonnet) - max_len,  step):
            sequences.append(sonnet[i:i + max_len + 1])
    return sequences


def _map_characters(data):
    unique_chars = sorted(list(set(''.join(data))))   
    char2vec = dict((c, i) for i, c in enumerate(unique_chars))
    return char2vec


def generate_training_data(char_sequences, char2vec):
    char_sequences = np.array(char_sequences)
    x_sequence = char_sequences[:, :-1]
    y_sequence = char_sequences[:, -1]

    X = np.array([to_categorical(s, num_classes=len(char2vec)) for s in x_sequence])
    Y = to_categorical(y_sequence, num_classes=len(char2vec))
    
    return X, Y


def train_rnn_model(X, Y, char2vec):
    # Simple model: 1 LSTM layer of 200 units, 1 Dense layer with `softmax` activation
    model = Sequential([
        LSTM(200, input_shape(X.shape[1], X.shape[2])),
        Dense(len(char2vec), activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    
    model.fit(X, Y, batch_size=128, epochs=60)
    save_rnn_model(model, 1, 200, 60)


def save_rnn_model(model, layers, units, epochs):
    model.save(f"{MODEL_DIRNAME}/LSTM-L{layers}-U{units}-E{epochs}.h5")


def load_rnn_model(model_name):
    return load_model(f"{MODEL_DIRNAME}/{model_name}.h5")


def build_lambda_model(model, temperature, X, char2vec):
    model_weights = [layer.get_weights() for layer in model.layers]
    # Add Lambda layer between LSTM and Dense
    lambda_model = Sequential([
        LSTM(200, input_shape=(X.shape[1], X.shape[2])),
        Lambda(lambda x: x / temperature),
        Dense(len(char2vec), activation='softmax')
    ])
    lambda_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Assign trained weights to new model
    lambda_model.layers[0].set_weights(model_weights[0])
    lambda_model.layers[2].set_weights(model_weights[1])

    return lambda_model


