import argparse

def params_setup(cmdline=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', type=str, required=True, help='work mode: train/test/chat')
  
  # path ctrl
  parser.add_argument('--model_name', type=str, default='movie_subtitles_en', help='model name, affects data, model, result save path')
  parser.add_argument('--scope_name', type=str, help='separate namespace, for multi-models working together')
  parser.add_argument('--work_root', type=str, default='works', help='root dir for data, model, result save path')

  # training params
  parser.add_argument('--learning_rate', type=float, default=0.5, help='Learning rate.')
  parser.add_argument('--learning_rate_decay_factor', type=float, default=0.99, help='Learning rate decays by this much.')
  parser.add_argument('--max_gradient_norm', type=float, default=5.0, help='Clip gradients to this norm.')
  parser.add_argument('--batch_size', type=int, default=64, help='Batch size to use during training.')

  parser.add_argument('--vocab_size', type=int, default=100000, help='Dialog vocabulary size.')
  parser.add_argument('--size', type=int, default=256, help='Size of each model layer.')
  parser.add_argument('--num_layers', type=int, default=4, help='Number of layers in the model.')

  parser.add_argument('--max_train_data_size', type=int, default=0, help='Limit on the size of training data (0: no limit)')
  parser.add_argument('--steps_per_checkpoint', type=int, default=500, help='How many training steps to do per checkpoint')

  # predicting params
  parser.add_argument('--beam_size', type=int, default=1, help='beam search size')
  parser.add_argument('--antilm', type=float, default=0, help='anti-language model weight')
  parser.add_argument('--n_bonus', type=int, default=0, help='bonus with sentence length')

  # environment params
  parser.add_argument('--gpu_usage', type=float, default=1.0, help='tensorflow gpu memory fraction used')
  parser.add_argument('--rev_model', type=int, default=0, help='reverse Q-A pair, for bi-direction model')
  parser.add_argument('--reinforce_learn', type=int, default=0, help='1 to enable reinforcement learning mode')
  parser.add_argument('--en_tfboard', type=int, default=0, help='Enable writing out tensorboard meta data')
  

  if cmdline:
    args = parser.parse_args(cmdline)
  else:
    args = parser.parse_args()
  
  if not args.scope_name: args.scope_name = args.model_name
  if args.rev_model: args.model_name += '_bidi' # bi-direction model
  
  # We use a number of buckets and pad to the closest one for efficiency.
  # See seq2seq_model.Seq2SeqModel for details of how they work.
  args.buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

  # post-process
  args.workspace = '%s/%s' % (args.work_root, args.model_name)
  args.test_dataset_path = '%s/data/test/test_set.txt' % (args.workspace)
  args.mert_dataset_path = '%s/data/test/mert_set.txt' % (args.workspace)
  args.data_dir = '%s/data' % args.workspace
  args.model_dir = '%s/nn_models' % args.workspace
  args.results_dir = '%s/results' % args.workspace
  args.tf_board_dir = '%s/tf_board' % args.workspace
  return args

