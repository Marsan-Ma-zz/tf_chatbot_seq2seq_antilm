from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from random import random
from datetime import datetime
from tensorflow.python.platform import gfile

from lib import data_utils
from lib import seq2seq_model


import heapq

def create_model(session, args, forward_only, force_dec_input=False):
  """Create translation model and initialize or load parameters in session."""
  model = seq2seq_model.Seq2SeqModel(
      source_vocab_size=args.vocab_size,
      target_vocab_size=args.vocab_size,
      buckets=args.buckets,
      size=args.size,
      num_layers=args.num_layers,
      max_gradient_norm=args.max_gradient_norm,
      batch_size=args.batch_size,
      learning_rate=args.learning_rate,
      learning_rate_decay_factor=args.learning_rate_decay_factor,
      use_lstm=False,
      forward_only=forward_only,
      force_dec_input=force_dec_input,
      # scope_name=args.scope_name,
  )

  # for tensorboard
  summary_writer = tf.train.SummaryWriter(args.tf_board_dir, session.graph)

  ckpt = tf.train.get_checkpoint_state(args.model_dir)
  if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model


def dict_lookup(rev_vocab, out):
    if (out < len(rev_vocab)):
      return rev_vocab[out]  
    else: 
      return data_utils._UNK


def softmax(x):
    prob = np.exp(x) / np.sum(np.exp(x), axis=0)
    return prob



def cal_bleu(cands, ref, stopwords=['的', '嗎']):
    cands = [s['dec_inp'].split() for s in cands]
    cands = [[w for w in sent if w[0] != '_'] for sent in cands]
    refs  = [w for w in ref.split() if w not in stopwords]
    bleus = []
    for cand in cands:
        if len(cand) < 4: cand += [''] * (4 - len(cand))
        bleu = sentence_bleu(refs, cand)
        bleus.append(bleu)
        print(refs, cand, bleu)
    return np.average(bleus)
    


def get_predicted_sentence(args, input_sentence, vocab, rev_vocab, model, sess, debug=False, antilm=0, n_bonus=0, beam_size=5):
    def model_step(enc_inp, dec_inp, dptr, target_weights, bucket_id):
      _, _, logits = model.step(sess, enc_inp, dec_inp, target_weights, bucket_id, forward_only=True, debug=debug)
      prob = softmax(logits[dptr][0])
      # print("model_step @ %s" % (datetime.now()))
      return prob

    input_token_ids = data_utils.sentence_to_token_ids(input_sentence, vocab)

    # Which bucket does it belong to?
    bucket_id = min([b for b in range(len(args.buckets)) if args.buckets[b][0] > len(input_token_ids)])
    outputs = []

    feed_data = {bucket_id: [(input_token_ids, outputs)]}
    # Get a 1-element batch to feed the sentence to the model.
    encoder_inputs, decoder_inputs, target_weights = model.get_batch(feed_data, bucket_id)
    if debug: print("\n[get_batch]\n", encoder_inputs, decoder_inputs, target_weights)

    # Get output logits for the sentence.
    beams, new_beams, results = [(1, {'eos': 0, 'dec_inp': decoder_inputs, 'prob': 1, 'prob_ts': 1, 'prob_t': 1})], [], [] # initialize beams as (log_prob, empty_string, eos)
    dummy_encoder_inputs = [np.array([data_utils.PAD_ID]) for _ in range(len(encoder_inputs))]
    
    for dptr in range(len(decoder_inputs)-1):
      if dptr > 0: 
        target_weights[dptr] = [1.]
        beams, new_beams = new_beams[:beam_size], []
      if debug: print("=====[beams]=====", beams)
      heapq.heapify(beams)  # since we will remove something
      for prob, cand in beams:
        if cand['eos']: 
          results += [(prob, cand)]
          continue

        # normal seq2seq
        if debug: print(cand['prob'], " ".join([dict_lookup(rev_vocab, w) for w in cand['dec_inp']]))

        all_prob_ts = model_step(encoder_inputs, cand['dec_inp'], dptr, target_weights, bucket_id)
        # anti-lm
        all_prob_t  = model_step(dummy_encoder_inputs, cand['dec_inp'], dptr, target_weights, bucket_id)
        # adjusted probability
        all_prob    = all_prob_ts - antilm * all_prob_t + n_bonus * dptr + random() * 1e-50

        # beam search
        for c in np.argsort(all_prob)[::-1][:beam_size]:
          new_cand = {
            'eos'     : (c == data_utils.EOS_ID),
            'dec_inp' : [(np.array([c]) if i == (dptr+1) else k) for i, k in enumerate(cand['dec_inp'])],
            'prob_ts' : cand['prob_ts'] * all_prob_ts[c],
            'prob_t'  : cand['prob_t'] * all_prob_t[c],
            'prob'    : cand['prob'] * all_prob[c],
          }
          new_cand = (new_cand['prob'], new_cand) # for heapq can only sort according to list[0]
          
          if (len(new_beams) < beam_size):
            heapq.heappush(new_beams, new_cand)
            # print("new_beam push", new_cand)
          elif (new_cand[0] > new_beams[0][0]):
            heapq.heapreplace(new_beams, new_cand)
            # print("new_beam replace", new_cand)
    results += new_beams  # last cands

    # post-process results
    res_cands = []
    for prob, cand in sorted(results, reverse=True):
      cand['dec_inp'] = " ".join([dict_lookup(rev_vocab, w) for w in cand['dec_inp']])
      res_cands.append(cand)
    return res_cands



# def get_predicted_sentence(args, input_sentence, vocab, rev_vocab, model, sess):
#     input_token_ids = data_utils.sentence_to_token_ids(input_sentence, vocab)

#     # Which bucket does it belong to?
#     bucket_id = min([b for b in range(len(args.buckets)) if args.buckets[b][0] > len(input_token_ids)])
#     outputs = []

#     feed_data = {bucket_id: [(input_token_ids, outputs)]}
#     # Get a 1-element batch to feed the sentence to the model.
#     encoder_inputs, decoder_inputs, target_weights = model.get_batch(feed_data, bucket_id)

#     # Get output logits for the sentence.
#     _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only=True)

#     outputs = []
#     # This is a greedy decoder - outputs are just argmaxes of output_logits.
#     for logit in output_logits:
#         selected_token_id = int(np.argmax(logit, axis=1))

#         if selected_token_id == data_utils.EOS_ID:
#             break
#         else:
#             outputs.append(selected_token_id)

#     # Forming output sentence on natural language
#     output_sentence = ' '.join([rev_vocab[output] for output in outputs])

#     return output_sentence
