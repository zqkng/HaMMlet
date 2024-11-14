# HaMMlet

**Shakespearean sonnet generator using [*hidden Markov models* (HMM)](https://en.wikipedia.org/wiki/Hidden_Markov_model)
and a [*long short-term memory* (LSTM)](https://en.wikipedia.org/wiki/Long_short-term_memory) recurrent neural network.**

Models were trained on a corpus that includes: all 154 of William Shakespeare's sonnets (`data/shakespeare.txt`)
and 89 sonnets from Edmund Spenser's *Amoretti* (`data/spenser.txt`).

  - Spenserian sonnets are a variation of the Shakespearean sonnet but with a more
    challenging rhyme scheme. 

In general, the sonnets follow a very specific structure 
(which makes it ideal for generative models):
  
  - Contain 14 lines, with 3 quatrains (each with 4 lines) followed
    by a couplet (2 lines)
  - Written in iambic pentameter (unstressed syllable followed by a stressed syllable;
    approximately 10 syllables per line)
  - Follow a strict rhyme scheme:
    - Shakespearean Sonnet: `ABAB CDCD EFEF GG`
    - Spenserian Sonnet: `ABAB BCBC CDCD EE`

Training data uses several different tokenization and sequencing methodologies:

- *character-based*, *line-based*, *quatrain/couplet-based*, and *sonnet-based*

### HMM (Hidden Markov Model)

Hidden Markov models are trained using the
[Baum-Welch (EM) algorithm](https://en.wikipedia.org/wiki/Baum–Welch_algorithm)
(unsupervised learning). Sonnets are generated using the
[Viterbi algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm).

- To re-train a HMM: see `HiddenMarkovModel.ipynb` for an example.
- Each trained model (HMM) includes the following:
	- Initial State Distribution Vector: `*-INIT.p`
	- Transition Probability Matrix: `*-TRANSITION.p`
	- Emission Probability Matrix: `*-EMISSION.p`
	- Mapping of word tokens to observation indices: `*-TOKENS.p`


### RNN (Recurrent Neural Network)
  
The RNN model consists of: 

- 3 LSTM layers of 600 units (each accompanied by 20% Dropout), 
- a Lambda layer is added (between LSTM and Dense layers) to control randomness via the `temperature` hyperparameter, and
- a fully connected Dense layer with `softmax` activation.

The model is trained to minimize *categorical cross-entropy* loss with
the `adam` optimizer. 

- To re-train the RNN: see `RecurrentNeuralNetwork.ipynb` for an example.
- Each trained model (RNN) includes the following:
  - Keras Sequential Model (HDF5): `*.h5`
  - List of 40-Character Training Sequences: `character_sequences.p`
  - Mapping of characters to indices: `char2vec.p`


## Dependencies:
- Python 3.9+
- [numpy](https://numpy.org/install/)
- [Keras](https://keras.io/getting_started/)
- [Tensorflow](https://www.tensorflow.org/install/pip)
- [Matplotlib](https://matplotlib.org/stable/users/getting_started/)
- [Jupyter Notebook](https://jupyter.org/install)

`pip install -r requirements.txt`

## How to Run:

To generate sonnets from a pre-existing model:

<pre>
import hammlet

hammlet.generate_hmm_sonnet("{model-name}", rhyme=True)
hammlet.generate_rnn_sonnet("{model-name}", seed="{sonnet-line}", temperature={float})
</pre>

Examples of trained models (for testing) are located in: `models/hmm` and `models/rnn`.
