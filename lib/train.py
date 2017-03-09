import sys, os, math, time, argparse, shutil, gzip
import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from six.moves import xrange  # pylint: disable=redefined-builtin
from datetime import datetime
from lib import seq2seq_model_utils, data_utils


def setup_workpath(workspace):
  for p in ['data', 'nn_models', 'results']:
    wp = "%s/%s" % (workspace, p)
    if not os.path.exists(wp): os.mkdir(wp)

  data_dir = "%s/data" % (workspace)
  # training data
  if not os.path.exists("%s/chat.in" % data_dir):
    n = 0
    f_zip   = gzip.open("%s/train/chat.txt.gz" % data_dir, 'rt')
    f_train = open("%s/chat.in" % data_dir, 'w')
    f_dev   = open("%s/chat_test.in" % data_dir, 'w')
    for line in f_zip:
      f_train.write(line)
      if n < 10000: 
        f_dev.write(line)
        n += 1


def train(args):
    print("[%s] Preparing dialog data in %s" % (args.model_name, args.data_dir))
    setup_workpath(workspace=args.workspace)
    train_data, dev_data, _ = data_utils.prepare_dialog_data(args.data_dir, args.vocab_size)

    if args.reinforce_learn:
      args.batch_size = 1  # We decode one sentence at a time.

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_usage)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        # Create model.
        print("Creating %d layers of %d units." % (args.num_layers, args.size))
        model = seq2seq_model_utils.create_model(sess, args, forward_only=False)

        # Read data into buckets and compute their sizes.
        print("Reading development and training data (limit: %d)." % args.max_train_data_size)
        dev_set = data_utils.read_data(dev_data, args.buckets, reversed=args.rev_model)
        train_set = data_utils.read_data(train_data, args.buckets, args.max_train_data_size, reversed=args.rev_model)
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(args.buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []

        # Load vocabularies.
        vocab_path = os.path.join(args.data_dir, "vocab%d.in" % args.vocab_size)
        vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

        while True:
          # Choose a bucket according to data distribution. We pick a random number
          # in [0, 1] and use the corresponding interval in train_buckets_scale.
          random_number_01 = np.random.random_sample()
          bucket_id = min([i for i in xrange(len(train_buckets_scale))
                           if train_buckets_scale[i] > random_number_01])

          # Get a batch and make a step.
          start_time = time.time()
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              train_set, bucket_id)

          # print("[shape]", np.shape(encoder_inputs), np.shape(decoder_inputs), np.shape(target_weights))
          if args.reinforce_learn:
            _, step_loss, _ = model.step_rf(args, sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, rev_vocab=rev_vocab)
          else:
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, forward_only=False, force_dec_input=True)

          step_time += (time.time() - start_time) / args.steps_per_checkpoint
          loss += step_loss / args.steps_per_checkpoint
          current_step += 1

          # Once in a while, we save checkpoint, print statistics, and run evals.
          if (current_step % args.steps_per_checkpoint == 0) and (not args.reinforce_learn):
            # Print statistics for the previous epoch.
            perplexity = math.exp(loss) if loss < 300 else float('inf')
            print ("global step %d learning rate %.4f step-time %.2f perplexity %.2f @ %s" %
                   (model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity, datetime.now()))

            # Decrease learning rate if no improvement was seen over last 3 times.
            if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
              sess.run(model.learning_rate_decay_op)

            previous_losses.append(loss)

            # # Save checkpoint and zero timer and loss.
            checkpoint_path = os.path.join(args.model_dir, "model.ckpt")
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)
            step_time, loss = 0.0, 0.0

            # Run evals on development set and print their perplexity.
            for bucket_id in xrange(len(args.buckets)):
              encoder_inputs, decoder_inputs, target_weights = model.get_batch(dev_set, bucket_id)
              _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, 
                                          target_weights, bucket_id, forward_only=True, force_dec_input=False)

              eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
              print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))

            sys.stdout.flush()
