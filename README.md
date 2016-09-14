# tensorflow chatbot
### (with seq2seq + attention + beam search + anti-LM)


## Briefing
This is a [seq2seq model](http://arxiv.org/abs/1406.1078) modified from [tensorflow example](https://www.tensorflow.org/versions/r0.10/tutorials/seq2seq/index.html), while:  


1. The original tensorflow seq2seq has [attention mechanism](http://arxiv.org/abs/1412.7449) out-of-box.
2. This work add option to do [beam search](https://en.wikipedia.org/wiki/Beam_search) in decoding procedure, which supposed to find better results.
3. This work add [anti-language model](https://arxiv.org/abs/1510.03055) to suppress the generic response problem of intrinsic seq2seq model.
4. A simple [Flask](http://flask.pocoo.org/) server `app.py` is included, which used to be a Facebook Messenger App backend.


## Just tell me how it works

#### Clone the repository

    git clone github.com/Marsan-Ma/tf_chatbot_seq2seq_antilm.git
    
#### Train the model

    cd tf_chatbot_antilm
    python3 main.py --mode train --model_name lyrics_ptt
    
#### Run some test example and see the bot response

    python3 main.py --mode test --model_name lyrics_ptt

#### Start your Facebook Messenger backend server

    python3 app.py --model_name lyrics_ptt

You may see the standalone [fb_messenger](https://github.com/Marsan-Ma/fb_messenger) repository for more details such as SSL, webhook, work-around of known bug.


### Want different corpus for different model?
You may find other corpus such as open movie subtitle, or forums from [this repository](https://github.com/Marsan-Ma/chat_corpus). You need to put it under path like:  

    works/<YOUR_MODEL_NAME>/data/train/chat.txt

And hand craft some testing sentences (each sentence per line) in:

    works/<YOUR_MODEL_NAME>/data/test/test_set.txt
    
    
## Introduction

Seq2seq is a great model released by [Cho et al., 2014](http://arxiv.org/abs/1406.1078). At first it's used to do machine translation, and soon people find that anything about **mapping something to another thing** could be also achieved by seq2seq model. Chatbot is one of these miracles, where we consider consecutive dialog as such kind of "mapping" relationship.

Here is the classic intro picture show the seq2seq model architecture, from this [blogpost about gmail auto-reply feature](http://googleresearch.blogspot.ru/2015/11/computer-respond-to-this-email.html).

[![seq2seq](https://4.bp.blogspot.com/-aArS0l1pjHQ/Vjj71pKAaEI/AAAAAAAAAxE/Nvy1FSbD_Vs/s640/2TFstaticgraphic_alt-01.png)](http://4.bp.blogspot.com/-aArS0l1pjHQ/Vjj71pKAaEI/AAAAAAAAAxE/Nvy1FSbD_Vs/s1600/2TFstaticgraphic_alt-01.png)


But the problem is, so far we haven't find a better objective function for charbot. We are still using **MLE** (maximum likelyhood estimation), which is doing good for machine translation, but always generate lousy response like "me too", "I think so", "I love you" while doing chat. 

These responses are not informative, but they do have large probability --- since they tend to appear many times in training corpus. We don't won't our chatbot always replying these noncense, so we need to find some way to make our bot more "interesting", technically speaking, to increase the "perplexity" of reponse.

Here we reproduce the work of [Li. et al., 2016](http://arxiv.org/pdf/1510.03055v3.pdf) try to solve this problem. The main idea is using the same seq2seq model as a language model, to get the candidate words with high probability in each decoding timestamp as a anti-model, then we penalize these words always being high probability for any input. By this anti-model, we could get more special, non-generic, informative response.

The original work of [Li. et al](http://arxiv.org/pdf/1510.03055v3.pdf) use [MERT (Och, 2003)](http://delivery.acm.org/10.1145/1080000/1075117/p160-och.pdf) with [BLEU](https://en.wikipedia.org/wiki/BLEU) as metrics to find the best probability weighting (the **λ** and **γ** in
**Score(T) = p(T|S) − λU(T) + γNt**) of the corresponding anti-language model. But I find that BLEU score in chat corpus tend to always being zero, thus can't get meaningful result here. If anyone has any idea about this, drop me a message, thanks!


## Parameters

There are some options to for model training and predicting in lib/config.py. Basically they are self-explained and could work with default value for most of cases. Here we only list something you  need to config:

**About environment**

name | type | Description
---- | ---- | -----------
mode | string | work mode: train/test/chat
model_name | string | model name, affects your working path (storing the data, nn_model, result folders)
scope_name | string | In tensorflow if you need to load two graph at the same time, you need to save/load them in different namespace. (If you need only one seq2seq model, leave it as default)
vocab_size | integer | depends on your corpus language: for english, 60000 is good enough. For chinese you need at least 100000 or 200000.

**About decoding**

name | type | default | Description
---- | ---- | ------- | -------
beam_size | int | 10 | beam search size, setting 1 equals to greedy search 
antilm | float | 0 (disabled) | punish weight of [anti-language model](http://arxiv.org/pdf/1510.03055v3.pdf) 
n_bonus | float | 0 (disabled) | reward weight of sentence length 


The anti-LM functin is disabled in default, you may start from setting antilm=0.5~0.7 and n_bonus=0.05 to see if you like the difference.


## Requirements

1. For training, GPU is recommended since seq2seq is a large model, you need certain computing power to do the training and predicting efficiently, especially when you set a large beam-search size.

2. DRAM requirement is not strict as CPU/GPU, since we are doing stochastic gradient decent.

3. If you are new to deep-learning, setting-up things like GPU, python environment is annoying to you, here are dockers of my machine learning environment:  
  [(non-gpu version docker)](https://github.com/Marsan-Ma/docker_mldm)  /  [(gpu version docker)](https://github.com/Marsan-Ma/docker_mldm_gpu)  



## References

Seq2seq is a model with many preliminaries, I've been spend quite some time surveying and here are some best materials which benefit me a lot:

1. The best blogpost explaining RNN, LSTM, GRU and seq2seq model: [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) by Christopher Olah.

2. This work [sherjilozair/char-rnn-tensorflow](https://github.com/sherjilozair/char-rnn-tensorflow) helps me understand the implementation detail (modeling, training, how graph works).

3. If you are interested in more magic about RNN, here is a MUST-READ blogpost: [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) by Andrej Karpathy.

4. The vanilla version seq2seq+attention: [nicolas-ivanov/tf_seq2seq_chatbot](https://github.com/nicolas-ivanov/tf_seq2seq_chatbot). This will help you figure out the main flow of vanilla seq2seq model, and I build this repository based on this work.


## TODOs
1. Currently I build beam-search out of graph, which means --- it's very slow. There are discussions about build it in-graph [here](https://github.com/tensorflow/tensorflow/issues/654#issuecomment-196168030) and [there](https://github.com/tensorflow/tensorflow/pull/3756). But unfortunately if you want add something more than beam-search, like this anti-LM work, you need much more than beam search to be in-graph.

2. I haven't figure out how the MERT with BLEU can optimize weight of anti-LM model, since currently the BLEU is often being zero.

