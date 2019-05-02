from __future__ import print_function

import random
import numpy as np
import tensorflow as tf

from utils.preprocess import basic      # for preprocessing data
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq

# To import:
# attention_rnn_cell.py
# beam_decoder.py

class Seq2SeqModel(object):
    def __init__(self, vocab_size, embedding_size, buckets_or_sequence_length, size_nodes, num_layers,
                 max_gradient_norm, batch_size, learning_rate, learning_rate_decay_factor, model_type,
                 use_lstm=True,
                 num_samples=512,
                 forward_only=False,
                 beam_search=True,
                 beam_size=10):
        """
        Create the models.  This constructor can be used to create embedded or embedded-attention,
            bucketed or non-bucketed models made of single or multi-layer RNN cells.

        Args:
          vocab_size: Size of the vocabulary.
          buckets_or_sentence_length:
            If using buckets:
              A list of pairs (I, O), where I specifies maximum input length
              that will be processed in that bucket, and O specifies maximum output
              length. Training instances that have inputs longer than I or outputs
              longer than O will be pushed to the next bucket and padded accordingly.
              We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
            Else:
              Number of the maximum number of words per sentence.
          size_nodes: Number of units (nodes) in each layer of the models.
          num_layers: Number of layers in the models.
          max_gradient_norm: Gradients will be clipped to maximally this norm.
          batch_size: The size of the batches used during training;
            the models construction is independent of batch_size, so it can be
            changed after initialization if this is convenient, e.g., for decoding.
          learning_rate: Learning rate to start with.
          learning_rate_decay_factor: Decay learning rate by this much when needed.
          num_samples: Number of samples for sampled softmax.
          forward_only: If set, we do not construct the backward pass in the models. (test phase)
        """
        # Determine if bucket is used or not
        self.buckets = None
        if type(buckets_or_sequence_length) == list:
            self.buckets = buckets_or_sequence_length
        else:
            self.max_sentence_length = buckets_or_sequence_length

        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)

        # If we use sampled softmax, we need an output projection
        """
        output_projection: 
            output_projection: None or a pair (W, B) of output projection weights and biases; 
            W has shape [output_size x num_decoder_symbols] and B has shape [num_decoder_symbols];
            # output_size = HIDDEN_SIZE
            - if provided and feed_previous=True, each fed previous output will first be multiplied by W and added B.
            
            If we use an output projection, we need to project outputs for decoding
        """
        output_projection = None
        softmax_loss_function = None

        # Sampled softmax only makes sense if we sample less than vocabulary size
        if 0 < num_samples < self.vocab_size:
            with tf.device("/cpu:0"):
                w = tf.get_variable("proj_w", [size_nodes, self.vocab_size])
                w_T = tf.transpose(w)
                b = tf.get_variable("proj_b", [self.vocab_size])
            output_projection = (w, b)

            def sampled_loss(inputs, labels):
                labels = tf.reshape(labels, [-1, 1])
                return tf.nn.sampled_softmax_loss(w_T, b. inputs, labels, num_samples, vocab_size)

            softmax_loss_function = sampled_loss

        # Create the internal multi-layer cell for our RNN
        single_cell = tf.contrib.rnn.GRUCell(size_nodes)
        if use_lstm:
            single_cell = tf.contrib.rnn.BasicLSTMCell(size_nodes, state_is_tuple=True)

        # Dropout
        # TODO: Make Dropout placeholder
        single_cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_probe=.8)

        cell = single_cell
        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([single_cell] * num_layers, state_is_tuple=True)

        # The seq2seq function: we use embedding for the input and attention if applicable
        if model_type is "embedding_attention":
            def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
                return seq2seq.embedding_attention_seq2seq(encoder_inputs=encoder_inputs,
                                                           decoder_inputs=decoder_inputs,
                                                           cell=cell,
                                                           num_encoder_symbols=self.vocab_size,
                                                           num_decoder_symbols=self.vocab_size,
                                                           embedding_size=embedding_size,
                                                           output_projection=output_projection,
                                                           feed_previous=do_decode,
                                                           beam_search=beam_search,
                                                           beam_size=beam_size)