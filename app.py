# coding: utf-8
import os, json, yaml, requests, jieba
from datetime import datetime
from flask import Flask, request, render_template
from OpenSSL import SSL
from random import random, choice

app = Flask(__name__)

#---------------------------
#   Load Model
#---------------------------
import tensorflow as tf
from lib import data_utils
from lib.config import params_setup
from lib.seq2seq_model_utils import create_model, get_predicted_sentence


class ChatBot(object):

    def __init__(self, args, debug=False):
        start_time = datetime.now()

        # flow ctrl
        self.args = args
        self.debug = debug
        self.fbm_processed = []
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_usage)
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
        
        # Create model and load parameters.
        self.args.batch_size = 1  # We decode one sentence at a time.
        self.model = create_model(self.sess, self.args)

        # Load vocabularies.
        self.vocab_path = os.path.join(self.args.data_dir, "vocab%d.in" % self.args.vocab_size)
        self.vocab, self.rev_vocab = data_utils.initialize_vocabulary(self.vocab_path)
        print("[ChatBot] model initialize, cost %i secs" % (datetime.now() - start_time).seconds)

        # load yaml setup
        self.FBM_API = "https://graph.facebook.com/v2.6/me/messages"
        with open("config.yaml", 'rt') as stream:
            try:
                cfg = yaml.load(stream)
                self.FACEBOOK_TOKEN = cfg.get('FACEBOOK_TOKEN')
                self.VERIFY_TOKEN = cfg.get('VERIFY_TOKEN')
            except yaml.YAMLError as exc:
                print(exc)


    def process_fbm(self, payload):
        for sender, msg in self.fbm_events(payload):
            self.fbm_api({"recipient": {"id": sender}, "sender_action": 'typing_on'})
            resp = self.gen_response(msg)
            self.fbm_api({"recipient": {"id": sender}, "message": {"text": resp}})
            if self.debug: print("%s: %s => resp: %s" % (sender, msg, resp))
            

    def gen_response(self, sent, max_cand=100):
        sent = " ".join([w.lower() for w in jieba.cut(sent) if w not in [' ']])
        # if self.debug: return sent
        raw = get_predicted_sentence(self.args, sent, self.vocab, self.rev_vocab, self.model, self.sess, debug=False)
        # find bests candidates
        cands = sorted(raw, key=lambda v: v['prob'], reverse=True)[:max_cand]
        
        if max_cand == -1:  # return all cands for debug
            cands = [(r['prob'], ' '.join([w for w in r['dec_inp'].split() if w[0] != '_'])) for r in cands]
            return cands
        else:
            cands = [[w for w in r['dec_inp'].split() if w[0] != '_'] for r in cands]
            return ' '.join(choice(cands)) or 'No comment'


    def gen_response_debug(self, sent, args=None):
        sent = " ".join([w.lower() for w in jieba.cut(sent) if w not in [' ']])
        raw = get_predicted_sentence(args, sent, self.vocab, self.rev_vocab, self.model, self.sess, debug=False, return_raw=True)
        return raw


    #------------------------------
    #   FB Messenger API
    #------------------------------
    def fbm_events(self, payload):
        data = json.loads(payload.decode('utf8'))
        if self.debug: print("[fbm_payload]", data)
        for event in data["entry"][0]["messaging"]:
            if "message" in event and "text" in event["message"]:
                q = (event["sender"]["id"], event["message"]["seq"])
                if q in self.fbm_processed:
                    continue
                else:
                    self.fbm_processed.append(q)
                    yield event["sender"]["id"], event["message"]["text"]


    def fbm_api(self, data):
        r = requests.post(self.FBM_API,
            params={"access_token": self.FACEBOOK_TOKEN},
            data=json.dumps(data),
            headers={'Content-type': 'application/json'})
        if r.status_code != requests.codes.ok:
            print("fb error:", r.text)
        if self.debug: print("fbm_send", r.status_code, r.text)
        

#---------------------------
#   Server
#---------------------------
@app.route('/chat', methods=['GET'])
def verify():
    if request.args.get('hub.verify_token', '') == chatbot.VERIFY_TOKEN:
        return request.args.get('hub.challenge', '')
    else:
        return 'Error, wrong validation token'

@app.route('/chat', methods=['POST'])
def chat():
    payload = request.get_data()
    chatbot.process_fbm(payload)
    return "ok"


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')    


@app.route('/privacy', methods=['GET'])
def privacy():
    return render_template('privacy.html')    


#---------------------------
#   Start Server
#---------------------------
if __name__ == '__main__':
    # check ssl files
    if not os.path.exists('ssl/server.crt'):
        print("SSL certificate not found! (should placed in ./ssl/server.crt)")
    elif not os.path.exists('ssl/server.key'):
        print("SSL key not found! (should placed in ./ssl/server.key)")
    else:
        # initialize model
        args = params_setup()
        chatbot = ChatBot(args, debug=False)
        # start server
        context = ('ssl/server.crt', 'ssl/server.key')
        app.run(host='0.0.0.0', port=443, debug=False, ssl_context=context)

