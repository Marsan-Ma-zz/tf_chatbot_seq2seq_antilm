# tensorflow chatbot
### (with seq2seq + attention + beam search + anti-LM)


## Briefing
This is a [seq2seq model](http://arxiv.org/abs/1406.1078) modified from [tensorflow example](https://www.tensorflow.org/versions/r0.10/tutorials/seq2seq/index.html), while:  


1. The original tensorflow seq2seq has [attention mechanism](http://arxiv.org/abs/1412.7449) out of box.
2. This work add option to do [beam search](https://en.wikipedia.org/wiki/Beam_search) in decoding procedure, which supposed to find better results.
3. This work also add [anti-language model](https://arxiv.org/abs/1510.03055) to suppress the generic response problem of intrinsic seq2seq model.
4. A simple [Flask]() server (app.py) is included, which is used to be a Facebook Messenger App backend.


## (tl;dr) Just tell me how it works

Download this repository

    git clone github.com/Marsan-Ma/tf_chatbot_seq2seq_antilm.git
    
Training the model

    cd tf_chatbot_antilm
    python3 main.py --mode train --model_name movie_lines_selected
    
Run some test example and see the bot response

    python3 main.py --mode test --model_name movie_lines_selected

Start your Facebook Messenger backend server

    python3 app.py

You may see my standalone [fb_messenger]() repository, it will explain more about details such as SSL, webhook, work-around of known bug.

## Introduction

Seq2seq is a great model released by Google in 2014, by [ et al](). At first it's used to do the machine translation, and easily out-performed the traditional statistical + rule based MT model.

Here is the classic intro picture of seq2seq model from [blogpost of gmail auto-reply feature](http://googleresearch.blogspot.ru/2015/11/computer-respond-to-this-email.html).

[![seq2seq](https://4.bp.blogspot.com/-aArS0l1pjHQ/Vjj71pKAaEI/AAAAAAAAAxE/Nvy1FSbD_Vs/s640/2TFstaticgraphic_alt-01.png)](http://4.bp.blogspot.com/-aArS0l1pjHQ/Vjj71pKAaEI/AAAAAAAAAxE/Nvy1FSbD_Vs/s1600/2TFstaticgraphic_alt-01.png)


Soon people find that anything about "mapping something to another thing" could be magically achieved by seq2seq model. Chatbot is just one of these amazing application, which we consider consecutive dialog as such kind of "mapping" relationship.

There is a known problem about seq2seq chatbot: since we train 
the model with MLE (maximum likelyhood estimation) as our object function, which make sense for machine translation, but not good for chatbot.

While we talk to each other, we are not expecting a generic reponse like "me too", "I think so", "I love you" since these are not informative. Here we reproduce the work of [Li. et al ](http://arxiv.org/pdf/1510.03055v3.pdf) from stanford and microsoft research which try to solve this problem.

The main idea is that: using the same seq2seq model as a language model to get the words with high probability as a anti-model, then we penalize these words with generic high probability by this anti-model to get more special, informative response.

The original work use [MERT]() with [BLEU]() as metrics to find the best weight of the probability of vanilla seq2seq model and the proposed anti-language model. But here I find the bleu score often being zero, thus we can't has meaningful result here. I am not sure this part make sense, if anyone has idea about this, please mail me thanks!


## Requirements

1. For training, GPU is recommended. Seq2seq is a large model, you might need certain computing power to do the training and predicting efficiently, especially when you set the beam-search size large.

2. DRAM requirement is not strict as CPU/GPU, since we are doing stochastic gradient decent.

3. If you are new to deep-learning, setting-up things like GPU, python environment is annoying to you, here are dockers of my machine learning environment:  
  [(non-gpu version docker)](https://github.com/Marsan-Ma/docker_mldm)  
  [(gpu version docker)](https://github.com/Marsan-Ma/docker_mldm_gpu)  


## References

1. To understand seq2seq and language model, we need to understand LSTM first. This work [sherjilozair/char-rnn-tensorflow](https://github.com/sherjilozair/char-rnn-tensorflow) helps me learns a lot, if you are new to language model, I suggest you try this work first.

2. Here is the seq2seq+attention only work: [nicolas-ivanov/tf_seq2seq_chatbot](https://github.com/nicolas-ivanov/tf_seq2seq_chatbot). This will help you figure out the main flow of vanilla seq2seq model.

