##############################################################################
# hammlet.py
# ----------
# Generate sonnets using the given Hidden Markov Model.
#   - Supports iambic pentameter or rhyming (though not both, as of yet).
##############################################################################

import re
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

import hmm
import rnn
import sonnet_helper as sh
import tokenizer

# Number of lines per sonnet
SONNET_LINES = 14
# Maximum character count per sonnet
MAX_CHARACTERS = 800
# Seed line for RNN generation
DEFAULT_SEED = "my mistress' eyes are nothing like the sun,"

def generate_rnn_sonnet(model_name, seed=DEFAULT_SEED, temperature=None):
    """Generate a Shakespearean-like sonnet using the given RNN model.

    Parameters
    ----------
    model_name : str
        Name of model.
    seed : str
        Initial line for sonnet generation.
    temperature : float
        Hyperparamter to control randomness of predictions.
 
    Returns
    -------
    poem : str
        Sonnet text.
    """
    char_sequences, char2vec = rnn.load_rnn_data()
    X, Y = rnn.generate_training_data(char_sequences, char2vec)
    model = rnn.load_rnn_model(model_name)
    if temperature:
        model = rnn.add_lambda_layer(model, char2vec, X, temperature)

    return _gen_rnn_poem(model, char2vec, X, seed)


def _gen_rnn_poem(model, char2vec, X, seed, max_char=MAX_CHARACTERS):
    # Generate `max_char` number of characters from seed
    for _ in range(max_char):
        X_test = [char2vec[c] for c in seed]
        X_test = pad_sequences([X_test], maxlen=X.shape[1])
        X_test = to_categorical(X_test, num_classes=len(char2vec))
        # Select class with highest probability
        Y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

        # Map vec2char
        for c, i in char2vec.items():
            if i == Y_pred[0]:
                seed += c
                break

    # Generate sonnet lines
    poem = []
    seed_tokens = re.split('(?<=[!?,.;:]) +', seed)
    
    for i, s in enumerate(seed_tokens):
        if i == 0:
            poem.append((f"{s}\n").capitalize())
        else:
            if len(poem[-1]) > 20:
                if len(s) > 20:
                    poem.append((f"{s}\n").capitalize())
                else:
                    poem.append((f"{s} ").capitalize())
            else:
                poem[-1] += (f"{s}\n")
    
    return ''.join(poem[:14])
 

def generate_hmm_sonnet(model_name, rhyme=False):
    """Generate a Shakespearean-like sonnet using the given hidden Markov model.
    
    If `rhyme := True`, then only reversed models trained on all sonnet lines
    are supported.
    Reversed models (trained on observation sequences where the word order of
    each sonnet line is reversed) is required since rhyming lines are
    generated from a seeded last word back to the first word.

    Otherwise, if `rhyme := False`, then only models trained on entire
    quatrains and couplets are supported.

    Parameters
    ----------
    model_name : str
        Name of model.
    rhyme : bool
        Flag to enable or disable rhyming scheme.
 
    Returns
    -------
    poem : str
        Sonnet text (with either iambic pentameter or rhyming).

    """
    if rhyme:
        A, B, PI, token_to_index = hmm.load_hmm_model(f"{model_name}")
        index_to_token = {index: token for token, index in token_to_index.items()}
        sonnet = _gen_sonnet_rhyme(A, B, token_to_index, index_to_token)
    else:
        q_A, q_B, q_PI, q_tokens = hmm.load_hmm_model(f"q-{model_name}")
        c_A, c_B, c_PI, c_tokens = hmm.load_hmm_model(f"c-{model_name}")
        sonnet = _gen_sonnet(q_A, q_B, q_PI, q_tokens, c_A, c_B, c_PI, c_tokens)

    return sonnet


def _gen_sonnet(q_transition, q_emission, q_init, q_tokens,
              c_transition, c_emission, c_init, c_tokens):
    """Generate an un-rhymed sonnet."""
    poem = ""
    syllable_dict = sh.load_syllable_dict()
    stress_dict = sh.load_stress_dict()

    # Generate quatrains (first 12 lines).
    for q in range(tokenizer.NUM_QUATRAINS):
        q_state = None
        for line in range(tokenizer.QUATRAIN_LINES):
            q_line, q_state = _gen_line(q_transition, q_emission, q_init,
                                        q_tokens, syllable_dict, stress_dict,
                                        current_state=q_state)
            poem += sh.fix_sentence_mechanics(q_line) + '\n'

    # Generate couplet (last 2 lines).
    c_state = None
    for c in range(tokenizer.COUPLET_LINES):
        c_line, c_state = _gen_line(c_transition, c_emission, c_init,
                                    c_tokens, syllable_dict, stress_dict,
                                    current_state=c_state)
        poem += sh.fix_sentence_mechanics(c_line) + '\n'

    return poem


def _gen_line(A, B, PI, token_dict, syllable_dict, stress_dict,
              length=10, delim=' ', current_state=None):
    """Generate an un-rhymed sonnet line.

    The start state is chosen using an initial vector and the length
    specified by the number of syllables.

    """
    # Verify model is functional.
    num_states = len(A)
    num_tokens = len(B[0])
    assert(num_states == len(A[0])), "Transition matrix is not square."
    assert(num_states == len(B)), "Emission matrix has wrong dimensions."

    line = ""
    start_state = current_state
    while (sh.syllable_count(line, syllable_dict) != 10 or
            sh.has_stress(line, stress_dict, syllable_dict) is not True):
        line = ""
        if start_state is None:
            current_state = np.random.choice(num_states, p=PI)
        else:
            current_state = start_state

        # Build word sequence (sonnet line).
        while sh.syllable_count(line, syllable_dict) < 10:
            # Select random token from current state based on emission matrix.
            token = int(np.random.choice(num_tokens, p=B[int(current_state)]))
            line += token_dict[token] + delim
            # Go to next state based on transition matrix.
            current_state = np.random.choice(num_states, p=A[int(current_state)])

    return line, current_state


def _gen_sonnet_rhyme(transition, emission, token_to_index, index_to_token):
    """Generate a sonnet with rhyming.

    Seed words for generating each line in the poem are randomly chosen
    from list of rhyming pairs extraced from sonnets. The intial state is
    determined by this seed word.

    """
    poem = ""
    data = tokenizer.load_data()
    q_rhymes, c_rhymes = tokenizer.process_rhymes(data)
    syllable_dict = sh.load_syllable_dict()

    # Generate quatrains (first 12 lines).
    for q in range(tokenizer.NUM_QUATRAINS):
        i, j = tuple(np.random.choice(len(q_rhymes), 2))
        q_seeds = [q_rhymes[i][0], q_rhymes[j][0],
                   q_rhymes[i][1], q_rhymes[j][1]]
        for seed in q_seeds:
            q_line = _gen_line_rhyme(transition, emission, token_to_index,
                                     index_to_token, syllable_dict, seed)
            poem += sh.fix_sentence_mechanics(q_line) + '\n'

    # Generate couplet (last 2 lines).
    k = np.random.choice(len(c_rhymes))
    c_seeds = [c_rhymes[k][0], c_rhymes[k][1]]
    for seed in c_seeds:
        c_line = _gen_line_rhyme(transition, emission, token_to_index,
                                 index_to_token, syllable_dict, seed)
        poem += sh.fix_sentence_mechanics(c_line) + '\n'

    return poem


def _gen_line_rhyme(A, B, token_to_index, index_to_token,
                    syllable_dict, seed, length=10):
    """Generate a sonnet line with rhyming.

    For rhyming to work, lines are generated from a seeded last word
    (rhyming words) back to the first word, and stress and iambic
    pentameter considerations were omitted.
    Also, a few assumptions are necessary:
        - Model provided must be trained on observation sequences where the
            word order of each sonnet line is reversed.
        - Mappings of both token to index and index to token are required.
    The start state is chosen using the emission matrix to find the state
    most likely to emit the seeded rhyming word.

    """
    # Verify model is functional.
    num_states = len(A)
    num_tokens = len(B[0])
    assert(num_states == len(A[0])), "Transition matrix is not square."
    assert(num_states == len(B)), "Emission matrix has incorrect dimensions."

    # Choose start state to be the state most likely to emit seed.
    index = token_to_index[seed]
    start_state = np.argmax(B[:, index])
    line = seed

    while sh.syllable_count(line, syllable_dict) != 10:
        current_state = start_state
        line = seed

        while sh.syllable_count(line, syllable_dict) < 10:
            # Go to next state based on transition matrix.
            current_state = np.random.choice(num_states, p=A[int(current_state)])
            # Select random token from current state based on emission matrix.
            index = int(np.random.choice(num_tokens, p=B[int(current_state)]))
            # Add tokens to sonnet line starting from end (in reverse).
            line = index_to_token[index] + ' ' + line

    return line
