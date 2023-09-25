##############################################################################
# tokenizer.py
# -------------
# Parse and pre-process sonnet text data into tokens for model training.
#
##############################################################################

import collections


# SONNET FORMAT:
SONNET_LINES = 14
NUM_QUATRAINS = 3
QUATRAIN_LINES = 4
COUPLET_LINES = 2

# Shakespeare sonnets 99, 126, 145 removed due to format deviations
EXCLUDED_SONNETS = ['99', '126', '145']

PUNCTUATION = [',', ':', '.', ';', '?', '!', '(', ')', "'", '"']
RAW_DATA_FILES = ["data/shakespeare.txt", "data/spenser.txt"]


def load_data(data_files=RAW_DATA_FILES):
    """Read in raw text of sonnets as training data.

    Parameters
    ----------
    data__files : list
        List of file paths to sonnets training data files.

    Returns
    -------
    data : list
        Sonnet data (lines) read from file.

    """
    data = []
    for filename in data_files:
        with open(filename, 'r') as f:
            for line in f:
                if line is not None and line != '\n':
                    data.append(line)
    return data


def tokenize_lpunc(line):
    """Parse sonnet lines and tokenize on words.

    Tokenization rules:
        - For standardization, all lines are in lowercase.
        - Newline characters are removed.
        - Hyphenated words are kept as single words
            (to prevent random words to be hyphenated together).
        - Ending punctuation is attached to the word to its left
            (this can help the model learn the positioning
             of words with punctuation).

    Parameters
    ----------
    line : str
        A single sonnet line.

    Returns
    -------
    line : str
        Formatted sonnet line with punctuation attached to word on the left
        and newline characters removed.

    """
    line = line.lower().lstrip().rstrip()
    line = line.split(' ')
    return line


def tokenize_nopunc(line):
    """Parse sonnet lines and tokenize on words.

    Tokenization rules:
        - For standardization, all lines are in lowercase.
        - Newline characters are removed.
        - Hyphenated words are kept as single words
            (to prevent random words to be hyphenated together).
        - All punctuation is removed.

    Parameters
    ----------
    line : str
        A single sonnet line.

    Returns
    -------
    line : str
        Formatted sonnet line with punctuation attached to word on the left
        and newline characters removed.

    """
    line = line.lower().lstrip().rstrip()
    for punc in PUNCTUATION:
        line = line.replace(punc, '')
    line = line.split(' ')
    return line


# TRAINING SEQUENCES
def sequence_each_line(tokenize, data):
    """Parse and format sonnets into training sequences on a per-line basis.

    Parameters
    ----------
    tokenizer : function
        Function that parses sonnet lines into tokens.
    data : list
        Sonnet data (lines) read from file.

    Returns
    -------
    sequences : list
        List of training sequences, where each sequence is
        a tokenized sonnet line.

    """
    sequences = []

    for line in data:
        parsed_line = tokenize(line)
        # Skip first line (which is just the sonnet number)
        if len(parsed_line) > 1:
            sequences.append(parsed_line)

    return sequences


def sequence_quatrains_couplets(tokenize, data):
    """Parse and format sonnets into training sequences as sets of quatrains and couplets.

    Sonnets are split into quatrains and couplets, and then quatrains and
    couplets are split on a per-line basis.

    Parameters
    ----------
    tokenizer : function
        Function that parses sonnet lines into tokens.
    data : list
        Sonnet data (lines) read from file.

    Returns
    -------
    quatrains : list
        Training seqeunces of all lines from quatrains.
    couplets : list
        Training sequences of all lines from couplets.

    """
    quatrains = []
    couplets = []
    lines = iter(data)
    line = next(lines, None)
    
    while line is not None:
        sonnet_line = tokenize(line)    # First line is the sonnet number
        if len(sonnet_line) == 1 and sonnet_line not in EXCLUDED_SONNETS:
            for i in range(NUM_QUATRAINS * QUATRAIN_LINES):
                line = next(lines, None)
                if line: quatrains.append(tokenize(line))
            for i in range(COUPLET_LINES):
                line = next(lines, None)
                if line: couplets.append(tokenize(line))
        line = next(lines, None)

    return quatrains, couplets


def sequence_full_sonnet(tokenize, data):
    """Parse and format each sonnet (full text) into a training sequence.

    Parameters
    ----------
    tokenizer : function
        Function that parses sonnet lines into tokens.
    data : list
        Sonnet data (lines) read from file.

    Returns
    -------
    sequences : list
        List of training sequences, where each sequence is
        a pre-processed sonnet string.
    """
    sonnets = collections.defaultdict(list)
    count = 0

    for line in data:
        sequence = tokenize(line)
        if len(sequence) == 1:
            count += 1
        else:
            sonnets[count].append(sequence)

    for i in sonnets:
        sonnets[i] = [' '.join(line) for line in sonnets[i]]

    return [' '.join(sonnets[line]) for line in sonnets]


# MISCELLANEOUS SONNET PROCESSING
def process_rhymes(data):
    """Compile lists of rhyming pairs from the sonnets text.

    Parameters
    ----------
    data : list
        Sonnet data (lines) read from file.

    Returns
    -------
    quatrain_rhymes : list
        Pairs of rhyming lines (in tuples) from all the quatrains.
    couplet_rhymes : list
        Pairs of rhyming lines (in tuples) from all the couplets.

    """
    quatrain_rhymes = []
    couplet_rhymes = []
    lines = iter(data)
    line = next(lines, None)
    
    while line is not None:
        sonnet_line = tokenize_nopunc(line) # First line is the sonnet number
        if len(sonnet_line) == 1 and sonnet_line not in EXCLUDED_SONNETS:
            for i in range(NUM_QUATRAINS):
                quatrain = []
                for j in range(QUATRAIN_LINES):
                    line = next(lines, None)
                    if line: quatrain.append(tokenize_nopunc(line))
                quatrain_rhymes.append((quatrain[0][-1], quatrain[2][-1]))
                quatrain_rhymes.append((quatrain[1][-1], quatrain[3][-1]))

            couplet = []
            for i in range(COUPLET_LINES):
                line = next(lines, None)
                if line: couplet.append(tokenize_nopunc(line))
            couplet_rhymes.append((couplet[-1], couplet[-1]))
        line = next(lines, None)

    return quatrain_rhymes, couplet_rhymes


def process_word_frequency(data):
    """Count frequency of words in all sonnets.

    Parameters
    ----------
    data : list
        Sonnet data (lines) read from file.

    Returns
    -------
    word_count : dict
        Mapping of words to frequency count.

    """
    word_count = collections.Counter()

    for line in data:
        words = tokenize_nopunc(line)
        if len(words) > 1:
            word_count.update(words)

    return dict(word_count)


