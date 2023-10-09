###############################################################################
# rnn.py
# ------
# Build and train LSTM (long short-term memory) recurrent neural network (RNN).
#
###############################################################################

import os
import pickle

import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Lambda, Dropout
from keras.utils import to_categorical

import tokenizer

MODEL_DIRNAME = "models/rnn"
DATA_DIRNAME = "data"

# Parameters for processing training sequences:
MAX_SEQUENCE_LENGTH = 40    # Avg. number of chars per sonnet line is ~40
STEP_SIZE = 1
# Parameters for building and training model:
HIDDEN_SIZE = 600
DROPOUT_PERCENTAGE = 0.2
BATCH_SIZE = 128
EPOCHS = 40


def load_rnn_data():
    """Wrapper function to load and process raw text data to sequences for RNN training.
    
    Returns
    -------
    char_sequences : list
        List of 40-character sequences.
    char2vec : dict
        Mapping of characters to indices.
    """
    data = tokenizer.load_data()
    sonnets = tokenizer.sequence_full_sonnet(tokenizer.tokenize_lpunc, data)
    sequences = _process_sequences(sonnets)
    char2vec = _map_characters(sonnets)
    
    char_sequences = []
    for seq in sequences:
        char_sequences.append([char2vec[c] for c in seq])

    with open(f"{DATA_DIRNAME}/character_sequences.p", "wb") as f1:
        pickle.dump(char_sequences, f1)
    with open(f"{DATA_DIRNAME}/char2vec.p", "wb") as f2:
        pickle.dump(char2vec, f2)

    return char_sequences, char2vec
    

def _process_sequences(data, max_len=MAX_SEQUENCE_LENGTH, step=STEP_SIZE):
    """Create 40-character training sequences from each sonnet.

    MAX_SEQUENCE_LENGTH defaults to 40, since that is the average number of
    characters per sonnet line (based on analysis of the distribution of
    number of characters per line in `shakespeare.txt`).

    """
    sequences = []
    for sonnet in data:
        for i in range(0, len(sonnet) - max_len,  step):
            sequences.append(sonnet[i:i + max_len + 1])
    return sequences


def _map_characters(data):
    """Map characters to indices to create sequence vectors for training."""
    unique_chars = sorted(list(set(''.join(data))))   
    char2vec = dict((c, i) for i, c in enumerate(unique_chars))
    return char2vec


def generate_training_data(char_sequences, char2vec):
    """Generate training data (X, Y) as inputs for RNN model.
    
    Parameters
    ----------
    char_sequences : list
        List of 40-character sequences.
    char2vec : dict
        Mapping of characters to indices.

    Returns
    -------
    X : np.array
        Training data (test)
    Y : np.array
        Training data (prediction)

    """
    char_sequences = np.array(char_sequences)
    x_sequence = char_sequences[:, :-1]
    y_sequence = char_sequences[:, -1]

    X = np.array([to_categorical(s, num_classes=len(char2vec)) for s in x_sequence])
    Y = to_categorical(y_sequence, num_classes=len(char2vec))
    
    return X, Y


def add_lambda_layer(model, char2vec, X, temperature):
    """Add Lambda layer (temperature hyperparameter) to model to introduce randomness in predictions.

    Temperature is a hyperparameter of RNNs used to control the randomness of
    predictions by scaling the logits before applying `softmax`.
    When temperature=1, the softmax is computed directly on the logits
    (the unscaled output of earlier layers).
    When temperature=0.5, the model computes the softmax on (logits / 0.5)
    (resulting in a larger value).

    Performing softmax on larger values makes the LSTM more confident
    (less input is needed to activate the output layer)
    but also more conservative in its samples
    (it is less likely to sample from unlikely candidates).

    Using a higher temperature produces a softer probability distribution over
    the classes, and makes the RNN more “easily excited” by samples,
    resulting in more diversity and also more mistakes.

    Temperature therefore increases the sensitivity to low probability candidates.

    Parameters
    ----------
    model : Sequential Model (Keras)
        Trained RNN model.
    char2vec : dict
        Mapping of characters to indices.
    X : np.array
        Training data vector (test)
    temperature : float
        Hyperparameter that represents the randomness of reweighting probabilities
        (higher temperature -> more randomn; lower temperature -> more deterministic).

    Returns
    -------
    lambda_model : Sequential Model (Keras)

    """
    # Save weight values from base trained model
    model_weights = [layer.get_weights() for layer in model.layers]
    # Add Lambda layer between LSTM and Dense
    lambda_model = Sequential([
        LSTM(HIDDEN_SIZE, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
        LSTM(HIDDEN_SIZE, return_sequences=True),
        LSTM(HIDDEN_SIZE),
        Lambda(lambda x: x / temperature),
        Dense(len(char2vec), activation='softmax')
    ])
    lambda_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Re-assign trained weights to new temperature model
    for i in range(3):
        lambda_model.layers[i].set_weights(model_weights[i*2])
    lambda_model.layers[-1].set_weights(model_weights[-1])

    return lambda_model


def train_rnn_model(X, Y, char2vec):
    """Wrapper function to define, build, and train the RNN model.

    Specifies the following information about model:
    architecture, weights, optimizer, loss, and metrics.
    
    Keras Sequential Model Specifications:
        - 3 LSTM layers of 600 units (each with a 20% Dropout layer)
        - 1 Dense layer with `softmax` activation
        - Loss: `categorical_crossentropy`
        - Optimizer: `adam`
        - metrics: `accuracy`

    Parameters
    ----------
    X : np.array 
    Y : np.array
    char2vex : dict

    Returns
    -------
    model : Sequential Model (Keras)       
    """
    # Build model
    model = Sequential([
        LSTM(HIDDEN_SIZE, input_shape(X.shape[1], X.shape[2]), return_sequences=True),
        Dropout(DROPOUT_PERCENTAGE),
        LSTM(HIDDEN_SIZE, return_sequences=True),
        Dropout(DROPOUT_PERCENTAGE),
        LSTM(HIDDEN_SIZE),
        Dropout(DROPOUT_PERCENTAGE),
        Dense(len(char2vec), activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Train model
    model.fit(X, Y, batch_size=BATCH_SIZE, epochs=EPOCHS)
    save_rnn_model(model, 3, hidden_size, epochs)    
    
    return model


def save_rnn_model(model, layers, units, epochs):
    """Save RNN model (Keras) as HDF5 state file.

    Parameters
    ----------
    model : Sequential Model (Keras)
        RNN model to save
    layers : int
        Number of LSTM layers in model.
    units : int
        Number of LSTM units in model.
    epochs : int
        Number of epochs to train model.
    """
    model.save(f"{MODEL_DIRNAME}/LSTM-L{layers}-U{units}-E{epochs}.h5")


def load_rnn_model(model_name):
    """Load RNN model (Keras) with the given model name.

    Parameters
    ----------
    model_name : str
        Name of RNN model to load from HDF5 state file.

    Returns
    -------
    model: Sequential Model (Keras)
        Deserialized RNN model 
        (contains information about: architecture, weights, optimizer, losses, metrics)
    """
    return load_model(f"{MODEL_DIRNAME}/{model_name}.h5")

