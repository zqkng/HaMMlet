##############################################################################
# sonnet_helper.py
# ----------------
# Helper methods for generating sonnets.
#
##############################################################################

import pickle
import tokenizer

SYLLABLE_DICT = "data/syllable_dict.p"
STRESS_DICT = "data/stress_dict.p"
PUNCTUATION = [',', ':', '.', ';', '?', '!', '(', ')', "'", '"']


def fix_sentence_mechanics(line):
    """Parses line and fixes sentence mechanics (mostly capitalization)."""
    # Capitalize the first word.
    words = line.split(' ')
    words[0] = words[0].capitalize()

    # Capitalize the word 'I'.
    for i in range(len(words)):
        if words[i] == 'i':
            words[i] = 'I'

    # Capitalize the first word of sentences.
    for i in range(len(words) - 1, 0, -1):
        prev = words[i - 1]
        if prev[-1] in ['!', '.', '?']:
            words[i] = words[i].capitalize()

    return ' '.join(words)


def syllable_count(line, syllable_dict):
    """Returns the number of syllables in a given line."""
    count = 0
    words = tokenizer.tokenize_nopunc(line)

    for word in words:
        if word != '':
            count += syllable_dict[word]

    return count


def has_stress(line, stress_dict, syllable_dict):
    """Checks whether a given line ends in a stress or unstressed syllable."""
    stress = True
    current_stress = 0
    words = tokenizer.tokenize_nopunc(line)

    for word in words:
        if word != '':
            if stress_dict[word] != current_stress:
                stress = False
            current_stress = (current_stress + syllable_dict[word]) % 2

    return stress


def load_syllable_dict():
    return _load_lingual_dict(SYLLABLE_DICT)


def load_stress_dict():
    return _load_lingual_dict(STRESS_DICT)


def _load_lingual_dict(filename):
    with open(filename, 'rb') as fp:
        d = pickle.load(fp)
    return d
