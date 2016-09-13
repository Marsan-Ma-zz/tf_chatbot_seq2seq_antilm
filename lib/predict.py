import os

import tensorflow as tf
from datetime import datetime

from tf_seq2seq_chatbot.lib import data_utils
from tf_seq2seq_chatbot.lib.seq2seq_model_utils import create_model, get_predicted_sentence


def predict(args, debug=False):
    def _get_test_dataset():
        with open(args.test_dataset_path) as test_fh:
            test_sentences = [s.strip() for s in test_fh.readlines()]
        return test_sentences

    results_filename = '_'.join(['results', str(args.num_layers), str(args.size), str(args.vocab_size)])
    results_path = os.path.join(args.results_dir, results_filename+'.txt')

    with tf.Session() as sess, open(results_path, 'w') as results_fh:
        # Create model and load parameters.
        args.batch_size = 1  # We decode one sentence at a time.
        model = create_model(sess, args, forward_only=True, force_dec_input=False)

        # Load vocabularies.
        vocab_path = os.path.join(args.data_dir, "vocab%d.in" % args.vocab_size)
        vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

        test_dataset = _get_test_dataset()

        for sentence in test_dataset:
            # Get token-ids for the input sentence.
            predicted_sentence = get_predicted_sentence(args, sentence, vocab, rev_vocab, model, sess)
            print(sentence, ' -> ', predicted_sentence)

            results_fh.write(predicted_sentence + '\n')
    results_fh.close()
    print("results written in %s" % results_path)
