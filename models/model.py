from __future__ import print_function

import random
import numpy as np
import tensorflow as tf

from utils.preprocess import basic

from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq

# To import:
# attention_rnn_cell.py
# beam_decoder.py