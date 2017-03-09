# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from math import log
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.ops import control_flow_ops
from lib import data_utils as data_utils
from lib import seq2seq as tf_seq2seq

class Seq2SeqModel(object):
  """Sequence-to-sequence model with attention and for multiple buckets.

  This class implements a multi-layer recurrent neural network as encoder,
  and an attention-based decoder. This is the same as the model described in
  this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
  or into the seq2seq library for complete model implementation.
  This class also allows to use GRU cells in addition to LSTM cells, and
  sampled softmax to handle large output vocabulary size. A single-layer
  version of this model, but with bi-directional encoder, was presented in
    http://arxiv.org/abs/1409.0473
  and sampled softmax is described in Section 3 of the following paper.
    http://arxiv.org/abs/1412.2007
  """

  def __init__(self,
               source_vocab_size,
               target_vocab_size,
               buckets,
               size,
               num_layers,
               max_gradient_norm,
               batch_size,
               learning_rate,
               learning_rate_decay_factor,
               use_lstm=False,
               num_samples=512,
               forward_only=False,
               scope_name='seq2seq',
               dtype=tf.float32):
    """Create the model.

    Args:
      source_vocab_size: size of the source vocabulary.
      target_vocab_size: size of the target vocabulary.
      buckets: a list of pairs (I, O), where I specifies maximum input length
        that will be processed in that bucket, and O specifies maximum output
        length. Training instances that have inputs longer than I or outputs
        longer than O will be pushed to the next bucket and padded accordingly.
        We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
      size: number of units in each layer of the model.
      num_layers: number of layers in the model.
      max_gradient_norm: gradients will be clipped to maximally this norm.
      batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
      learning_rate: learning rate to start with.
      learning_rate_decay_factor: decay learning rate by this much when needed.
      use_lstm: if true, we use LSTM cells instead of GRU cells.
      num_samples: number of samples for sampled softmax.
      forward_only: if set, we do not construct the backward pass in the model.
      dtype: the data type to use to store internal variables.
    """
    self.scope_name = scope_name
    with tf.variable_scope(self.scope_name):
      self.source_vocab_size = source_vocab_size
      self.target_vocab_size = target_vocab_size
      self.buckets = buckets
      self.batch_size = batch_size
      self.learning_rate = tf.Variable(
          float(learning_rate), trainable=False, dtype=dtype)
      self.learning_rate_decay_op = self.learning_rate.assign(
          self.learning_rate * learning_rate_decay_factor)
      self.global_step = tf.Variable(0, trainable=False)
      self.dummy_dialogs = [] # [TODO] load dummy sentences 

      # If we use sampled softmax, we need an output projection.
      output_projection = None
      softmax_loss_function = None
      # Sampled softmax only makes sense if we sample less than vocabulary size.
      if num_samples > 0 and num_samples < self.target_vocab_size:
        w_t = tf.get_variable("proj_w", [self.target_vocab_size, size], dtype=dtype)
        w = tf.transpose(w_t)
        b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=dtype)
        output_projection = (w, b)

        def sampled_loss(labels, inputs):
          labels = tf.reshape(labels, [-1, 1])
          # We need to compute the sampled_softmax_loss using 32bit floats to
          # avoid numerical instabilities.
          local_w_t = tf.cast(w_t, tf.float32)
          local_b = tf.cast(b, tf.float32)
          local_inputs = tf.cast(inputs, tf.float32)
          return tf.cast(
              tf.nn.sampled_softmax_loss(
                  weights=local_w_t,
                  biases=local_b,
                  labels=labels,
                  inputs=local_inputs,
                  num_sampled=num_samples,
                  num_classes=self.target_vocab_size),
              dtype)
        softmax_loss_function = sampled_loss

      # Create the internal multi-layer cell for our RNN.
      def single_cell():
        return tf.contrib.rnn.GRUCell(size)
      if use_lstm:
        def single_cell():
          return tf.contrib.rnn.BasicLSTMCell(size)
      cell = single_cell()
      if num_layers > 1:
        cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])

      # The seq2seq function: we use embedding for the input and attention.
      def seq2seq_f(encoder_inputs, decoder_inputs, feed_previous):
        return tf_seq2seq.embedding_attention_seq2seq(
            encoder_inputs, 
            decoder_inputs, 
            cell,
            num_encoder_symbols=source_vocab_size,
            num_decoder_symbols=target_vocab_size,
            embedding_size=size,
            output_projection=output_projection,
            feed_previous=feed_previous, #do_decode,
            dtype=dtype)

      # Feeds for inputs.
      self.encoder_inputs = []
      self.decoder_inputs = []
      self.target_weights = []
      for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
        self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                  name="encoder{0}".format(i)))
      for i in xrange(buckets[-1][1] + 1):
        self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                  name="decoder{0}".format(i)))
        self.target_weights.append(tf.placeholder(dtype, shape=[None],
                                                  name="weight{0}".format(i)))

      # Our targets are decoder inputs shifted by one.
      targets = [self.decoder_inputs[i + 1]
                 for i in xrange(len(self.decoder_inputs) - 1)]

      # for reinforcement learning
      # self.force_dec_input = tf.placeholder(tf.bool, name="force_dec_input")
      # self.en_output_proj = tf.placeholder(tf.bool, name="en_output_proj")

      # Training outputs and losses.
      if forward_only:
        self.outputs, self.losses, self.encoder_state = tf_seq2seq.model_with_buckets(
            self.encoder_inputs, self.decoder_inputs, targets,
            self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
            softmax_loss_function=softmax_loss_function)
        # If we use output projection, we need to project outputs for decoding.
        if output_projection is not None:
          for b in xrange(len(buckets)):
            self.outputs[b] = [
                tf.matmul(output, output_projection[0]) + output_projection[1]
                for output in self.outputs[b]
            ]
      else:
        self.outputs, self.losses, self.encoder_state = tf_seq2seq.model_with_buckets(
            self.encoder_inputs, self.decoder_inputs, targets,
            self.target_weights, buckets,
            lambda x, y: seq2seq_f(x, y, False),
            softmax_loss_function=softmax_loss_function)

      # # Training outputs and losses.
      # self.outputs, self.losses, self.encoder_state = tf_seq2seq.model_with_buckets(
      #     self.encoder_inputs, self.decoder_inputs, targets,
      #     self.target_weights, buckets, 
      #     lambda x, y: seq2seq_f(x, y, tf.where(self.force_dec_input, False, True)),
      #     softmax_loss_function=softmax_loss_function
      #   )
      #   # If we use output projection, we need to project outputs for decoding.
      #   # if output_projection is not None:
      # for b in xrange(len(buckets)):
      #   self.outputs[b] = [
      #       control_flow_ops.cond(
      #         self.en_output_proj,
      #         lambda: tf.matmul(output, output_projection[0]) + output_projection[1],
      #         lambda: output
      #       )
      #       for output in self.outputs[b]
      #   ]
        
      # Gradients and SGD update operation for training the model.
      params = tf.trainable_variables()
      # if not forward_only:
      self.gradient_norms = []
      self.updates = []
      self.advantage = [tf.placeholder(tf.float32, name="advantage_%i" % i) for i in xrange(len(buckets))]
      opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      for b in xrange(len(buckets)):
        # self.losses[b] = tf.subtract(self.losses[b], self.advantage[b])
        gradients = tf.gradients(self.losses[b], params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                         max_gradient_norm)
        self.gradient_norms.append(norm)
        self.updates.append(opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step))

      all_variables = tf.global_variables()
      all_variables = [k for k in tf.global_variables() if k.name.startswith(self.scope_name)]
      self.saver = tf.train.Saver(all_variables)


  def step(self, session, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, forward_only, force_dec_input=False, advantage=None):

    """Run a step of the model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      encoder_inputs: list of numpy int vectors to feed as encoder inputs.
      decoder_inputs: list of numpy int vectors to feed as decoder inputs.
      target_weights: list of numpy float vectors to feed as target weights.
      bucket_id: which bucket of the model to use.
      forward_only: whether to do the backward step or only forward.

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, and the outputs.

    Raises:
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    # Check if the sizes match.
    encoder_size, decoder_size = self.buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(target_weights) != decoder_size:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {
      # self.force_dec_input.name:  force_dec_input,
      # self.en_output_proj.name:   forward_only,
    }
    for l in xrange(len(self.buckets)):
      input_feed[self.advantage[l].name] = advantage[l] if advantage else 0
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]
    
    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)


    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms[bucket_id],  # Gradient norm.
                     self.losses[bucket_id]]  # Loss for this batch.
    else:
      output_feed = [self.encoder_state[bucket_id], 
                     self.losses[bucket_id]]  # Loss for this batch.
      for l in xrange(decoder_size):  # Output logits.
        output_feed.append(self.outputs[bucket_id][l])

    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    else:
      return outputs[0], outputs[1], outputs[2:]  # encoder_state, loss, outputs.


    # # Output feed: depends on whether we do a backward step or not.
    # if training:  # normal training
    #   output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
    #                  self.gradient_norms[bucket_id],  # Gradient norm.
    #                  self.losses[bucket_id]]  # Loss for this batch.
    # else:  # testing or reinforcement learning
    #   output_feed = [self.encoder_state[bucket_id], self.losses[bucket_id]]  # Loss for this batch.
    #   for l in xrange(decoder_size):  # Output logits.
    #     output_feed.append(self.outputs[bucket_id][l])

    # outputs = session.run(output_feed, input_feed)
    # if training:
    #   return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    # else:
    #   return outputs[0], outputs[1], outputs[2:]  # encoder_state, loss, outputs.



  def step_rf(self, args, session, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, rev_vocab=None, debug=True):
    
    # initialize 
    init_inputs = [encoder_inputs, decoder_inputs, target_weights, bucket_id]
    sent_max_length = args.buckets[-1][0]
    resp_tokens, resp_txt = self.logits2tokens(encoder_inputs, rev_vocab, sent_max_length, reverse=True)
    if debug: print("[INPUT]:", resp_txt)
    
    # Initialize
    ep_rewards, ep_step_loss, enc_states = [], [], []
    ep_encoder_inputs, ep_target_weights, ep_bucket_id = [], [], []

    # [Episode] per episode = n steps, until break
    while True:
      #----[Step]----------------------------------------
      encoder_state, step_loss, output_logits = self.step(session, encoder_inputs, decoder_inputs, target_weights,
                          bucket_id, training=False, force_dec_input=False)
      
      # memorize inputs for reproducing curriculum with adjusted losses
      ep_encoder_inputs.append(encoder_inputs)
      ep_target_weights.append(target_weights)
      ep_bucket_id.append(bucket_id)
      ep_step_loss.append(step_loss)
      enc_states_vec = np.reshape(np.squeeze(encoder_state, axis=1), (-1))
      enc_states.append(enc_states_vec)
      
      # process response
      resp_tokens, resp_txt = self.logits2tokens(output_logits, rev_vocab, sent_max_length)
      if debug: print("[RESP]: (%.4f) %s" % (step_loss, resp_txt))

      # prepare for next dialogue
      bucket_id = min([b for b in range(len(args.buckets)) if args.buckets[b][0] > len(resp_tokens)])
      feed_data = {bucket_id: [(resp_tokens, [])]}
      encoder_inputs, decoder_inputs, target_weights = self.get_batch(feed_data, bucket_id)
      
      #----[Reward]----------------------------------------
      # r1: Ease of answering
      r1 = [self.logProb(session, args.buckets, resp_tokens, d) for d in self.dummy_dialogs]
      r1 = -np.mean(r1) if r1 else 0
      
      # r2: Information Flow
      if len(enc_states) < 2:
        r2 = 0
      else:
        vec_a, vec_b = enc_states[-2], enc_states[-1]
        r2 = sum(vec_a*vec_b) / sum(abs(vec_a)*abs(vec_b))
        r2 = -log(r2)
      
      # r3: Semantic Coherence
      r3 = -self.logProb(session, args.buckets, resp_tokens, ep_encoder_inputs[-1])

      # Episode total reward
      R = 0.25*r1 + 0.25*r2 + 0.5*r3
      rewards.append(R)
      #----------------------------------------------------
      if (resp_txt in self.dummy_dialogs) or (len(resp_tokens) <= 3) or (encoder_inputs in ep_encoder_inputs): 
        break # check if dialog ended
      
    # gradient decent according to batch rewards
    rto = (max(ep_step_loss) - min(ep_step_loss)) / (max(ep_rewards) - min(ep_rewards))
    advantage = [mp.mean(ep_rewards)*rto] * len(args.buckets)
    _, step_loss, _ = self.step(session, init_inputs[0], init_inputs[1], init_inputs[2], init_inputs[3],
              training=True, force_dec_input=False, advantage=advantage)
    
    return None, step_loss, None



  # log(P(b|a)), the conditional likelyhood
  def logProb(self, session, buckets, tokens_a, tokens_b):
    def softmax(x):
      return np.exp(x) / np.sum(np.exp(x), axis=0)

    # prepare for next dialogue
    bucket_id = min([b for b in range(len(buckets)) if buckets[b][0] > len(tokens_a)])
    feed_data = {bucket_id: [(tokens_a, tokens_b)]}
    encoder_inputs, decoder_inputs, target_weights = self.get_batch(feed_data, bucket_id)

    # step
    _, _, output_logits = self.step(session, encoder_inputs, decoder_inputs, target_weights,
                        bucket_id, training=False, force_dec_input=True)
    
    # p = log(P(b|a)) / N
    p = 1
    for t, logit in zip(tokens_b, output_logits):
      p *= softmax(logit[0])[t]
    p = log(p) / len(tokens_b)
    return p


  def logits2tokens(self, logits, rev_vocab, sent_max_length=None, reverse=False):
    if reverse:
      tokens = [t[0] for t in reversed(logits)]
    else:
      tokens = [int(np.argmax(t, axis=1)) for t in logits]
    if data_utils.EOS_ID in tokens:
      eos = tokens.index(data_utils.EOS_ID)
      tokens = tokens[:eos]
    txt = [rev_vocab[t] for t in tokens]
    if sent_max_length:
      tokens, txt = tokens[:sent_max_length], txt[:sent_max_length]
    return tokens, txt


  def discount_rewards(self, r, gamma=0.99):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


  def get_batch(self, data, bucket_id):
    """Get a random batch of data from the specified bucket, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
      data: a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
      bucket_id: integer, which bucket to get the batch for.

    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for _ in xrange(self.batch_size):
      encoder_input, decoder_input = random.choice(data[bucket_id])

      # Encoder inputs are padded and then reversed.
      encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
      encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_pad_size = decoder_size - len(decoder_input) - 1
      decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                            [data_utils.PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(self.batch_size, dtype=np.float32)
      for batch_idx in xrange(self.batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights
