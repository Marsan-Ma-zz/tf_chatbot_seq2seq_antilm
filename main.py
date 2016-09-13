import os, sys, argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf

from lib.config import params_setup
from lib.train import train
from lib.predict import predict
from lib.chat import chat
# from lib.mert import mert


def main(_):
    args = params_setup()
    print("[args]: ", args)
    if args.mode == 'train':
      train(args)
    elif args.mode == 'test':
      predict(args)
    elif args.mode == 'chat':
      chat(args)
    # elif args.mode == 'mert':
    #   mert(args)


if __name__ == "__main__":
    tf.app.run()