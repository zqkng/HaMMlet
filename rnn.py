###############################################################################
# rnn.py
# ------
# Build and train LSTM (long short-term memory) recurrent neural network (RNN).
#
###############################################################################

import tokenizer

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


