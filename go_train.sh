#!/bin/bash

NOW=$(date +"%Y%m%d_%H%M")
PY3='stdbuf -o0 nohup python3 -u'

# -----[open_subtitles]-----
# python3 main.py --mode train --model_name open_subtitles --vocab_size 80000
# python3 main.py --mode test --model_name open_subtitles --vocab_size 80000
# $PY3 main.py --mode train --model_name open_subtitles --vocab_size 80000 > "./logs/seq2seq_open_subtitles_$NOW.log" &
# $PY3 main.py --mode train --model_name open_subtitles_rev --vocab_size 80000 > "./logs/seq2seq_open_subtitles_rev_$NOW.log" &

# python3 main.py --mode test --model_name open_subtitles_short --vocab_size 20000
# $PY3 main.py --mode train --model_name open_subtitles_short --vocab_size 20000 > "./logs/seq2seq_open_subtitles_short_$NOW.log" &
# python3 main.py --mode train --model_name open_subtitles_short --vocab_size 10000 --batch_size 64 --size 128

# -----[movie_lines_selected]-----
# python3 main.py --mode train --model_name movie_lines_selected --vocab_size 80000
$PY3 main.py --mode train --model_name movie_lines_selected --vocab_size 80000 > "./logs/seq2seq_movie_lines_selected_$NOW.log" &

# -----[lyrics_zh]-----
# python3 main.py --mode train --model_name lyrics_zh
# python3 main.py --mode test --model_name lyrics_zh
# $PY3 main.py --mode train --model_name lyrics_zh > "./logs/seq2seq_lyrics_zh_$NOW.log" &

# -----[17live_comments]-----
# python3 main.py --mode train --model_name 17live_comments --vocab_size 200000
# python3 main.py --mode mert --model_name 17live_comments --vocab_size 200000
# python3 main.py --mode test --model_name 17live_comments --vocab_size 200000 #> "./logs/seq2seq_17live_comments_test_$NOW.log" &
# $PY3 main.py --mode train --model_name 17live_comments --vocab_size 200000 > "./logs/seq2seq_17live_comments_$NOW.log" &

# -----[17live_comments_rev]-----
# python3 main.py --mode test --model_name 17live_comments
# $PY3 main.py --mode train --model_name 17live_comments_rev --vocab_size 200000 > "./logs/seq2seq_17live_comments_rev_$NOW.log" &
