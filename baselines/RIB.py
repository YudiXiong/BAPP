import argparse
import datetime
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from perturbation import perturb_sequences
from evaluation import evaluate_rib
import random

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def parse_args():
    parser = argparse.ArgumentParser(description='Run RIB with optional sequence perturbation.')

    parser.add_argument('--epoch', type=int, default=300,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='datasets/JD/data2',
                        help='data directory')


    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--emb_size', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--early_stop_epoch', default=20, type=int)
    parser.add_argument('--alpha', type=float, default=0.8, help='Position-based sampling parameter for inter-subsequence distortion.')
    parser.add_argument('--gamma', type=float, default=0.4, help='Subsequence selection ratio for intra-subsequence deletion.')
    parser.add_argument('--beta', type=float, default=0.4, help='Subsequence selection ratio for intra-subsequence reordering.')
    parser.add_argument('--lamda', type=float, default=0.4, help='Legacy argument retained for compatibility; unused by the current perturbation operations.')
    parser.add_argument('--tag', type=int, default=0, help='0: unchanged; 1: intra-subsequence deletion; 2: inter-subsequence distortion; 3: intra-subsequence reordering.')
    return parser.parse_args()


class RIB:
    def __init__(self, emb_size,learning_rate,item_num,state_size):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.behavior_num = 2
        self.item_num=int(item_num)

        self.all_embeddings=self.initialize_embeddings()

        self.item_seq = tf.placeholder(tf.int32, [None, state_size],name='item_seq')
        self.len_seq=tf.placeholder(tf.int32, [None],name='len_seq')
        self.target= tf.placeholder(tf.int32, [None],name='target')
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.behavior_seq = tf.placeholder(tf.int32, [None, state_size])

        self.behavior_emb = tf.nn.embedding_lookup(self.all_embeddings['behavior_embeddings'], self.behavior_seq)
        self.input_emb=tf.nn.embedding_lookup(self.all_embeddings['item_embeddings'],self.item_seq)
        self.new_input_emb = tf.concat([self.input_emb,self.behavior_emb],axis=2)
        a = tf.Print(self.new_input_emb, ["OUTPUT", self.new_input_emb])

        self.gru_out, self.states_hidden= tf.nn.dynamic_rnn(
            tf.contrib.rnn.GRUCell(self.emb_size),
            self.new_input_emb,
            dtype=tf.float32,
            sequence_length=self.len_seq,
        )
        a = tf.Print(self.gru_out, ["OUTPUT", self.gru_out])

        a = tf.Print(self.states_hidden, ["OUTPUT", self.states_hidden])

        self.att_net = tf.contrib.layers.fully_connected(self.gru_out, self.hidden_size,
                                                         activation_fn=tf.nn.tanh, scope="att_net1")
        a = tf.Print(self.att_net, ["OUTPUT", self.att_net])

        self.att_net = tf.contrib.layers.fully_connected(self.att_net, 1,
                                                         activation_fn=None, scope="att_net2")
        mask = tf.expand_dims(tf.not_equal(self.item_seq, item_num), -1)
        a = tf.Print(mask, ["OUTPUT", mask])

        paddings = tf.ones_like(self.att_net) * (-2 ** 32 + 1)
        self.att_net = tf.where(mask, self.att_net, paddings)
        self.att_net = tf.nn.softmax(self.att_net,axis=1)
        a = tf.Print(self.gru_out * self.att_net, ["OUTPUT", self.gru_out * self.att_net])


        self.final_state = tf.reduce_sum(self.gru_out * self.att_net,axis=1)
        with tf.name_scope("dropout"):
            self.final_state = tf.layers.dropout(self.final_state,
                                     rate=args.dropout_rate,
                                   seed=args.random_seed,
                                   training=tf.convert_to_tensor(self.is_training))

        self.output = tf.contrib.layers.fully_connected(self.final_state,self.item_num,activation_fn=tf.nn.softmax,scope='fc')
        self.loss = tf.keras.losses.sparse_categorical_crossentropy(self.target,self.output)
        self.loss = tf.reduce_mean(self.loss)
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def initialize_embeddings(self):
        all_embeddings = dict()
        item_embeddings= tf.Variable(tf.random_normal([self.item_num, self.hidden_size], 0.0, 0.01),
            name='item_embeddings')
        padding = tf.zeros([1,self.hidden_size],dtype= tf.float32)
        item_embeddings = tf.concat([item_embeddings,padding],axis=0)
        behavior_embeddings = tf.Variable(tf.random_normal([self.behavior_num, self.hidden_size], 0.0, 0.01),
                                          name='behavior_embeddings')
        padding = tf.zeros([1,self.hidden_size],dtype= tf.float32)
        behavior_embeddings = tf.concat([behavior_embeddings,padding],axis=0)
        all_embeddings['item_embeddings']=item_embeddings
        all_embeddings['behavior_embeddings'] = behavior_embeddings
        return all_embeddings

if __name__ == '__main__':

    args = parse_args()
    tag, alpha, beta, gamma, lamda = args.tag, args.alpha, args.beta, args.gamma, args.lamda
    data_directory = args.data
    data_statis = pd.read_pickle(os.path.join(data_directory, 'data_statis.df'))
    state_size = data_statis['state_size'][0]
    item_num = data_statis['item_num'][0]
    topk = [5, 10, 20, 50]

    tf.reset_default_graph()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)


    rib_model = RIB(emb_size=args.emb_size, learning_rate=args.lr,item_num=item_num,state_size=state_size)

    saver = tf.train.Saver(max_to_keep=10000)


    nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    if tag == 0:
        label = 'origin'
    elif tag == 1 or tag == 11:
        label = gamma
    elif tag == 2:
        label = alpha
    elif tag == 3 or tag == 33:
        label = beta
    elif tag == 4 or tag == 5:
        label = lamda

    save_dir = './model_save/RIB/{}_emb_{}_lr_{}_dropout_{}_{}'.format(args.data, args.emb_size, args.lr,
                                                                        args.dropout_rate, nowTime)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    isExists = os.path.exists(save_dir)
    best_model_dir = os.path.join(save_dir, "best_model")

    data_loader = pd.read_pickle(os.path.join(data_directory, 'train2.df'))
    print("data number of click :{} , data number of purchase :{}".format(
        data_loader[data_loader['is_buy'] == 0].shape[0],
        data_loader[data_loader['is_buy'] == 1].shape[0],
    ))

    total_step=0

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        num_rows=data_loader.shape[0]
        num_batches=int(num_rows/args.batch_size)
        print(num_rows,num_batches)
        best_ndcg_10 = -1
        count = 0
        for i in range(args.epoch):
            print(i)
            start_time_i = datetime.datetime.now()

            for j in range(num_batches):
                batch = data_loader.sample(n=args.batch_size).to_dict()
                item_seq = list(batch['item_seq'].values())
                behavior_seq = list(batch['behavior_seq'].values())
                len_seq = list(batch['len_seq'].values())
                target=list(batch['target'].values())

                len_seq = [np.sum(seq!=item_num) for seq in item_seq]
                len_seq = [ss if ss > 0 else 1 for ss in len_seq]
                item_seq = [list(item_seq[r][:l1]) for r,l1 in enumerate(len_seq)]
                behavior_seq = [list(behavior_seq[r][:l1]) for r,l1 in enumerate(len_seq)]


                item_seq, behavior_seq, len_seq = perturb_sequences(item_seq, behavior_seq, len_seq, item_num, state_size, tag, alpha, beta, gamma, lamda)

                loss, _ = sess.run([rib_model.loss, rib_model.opt],
                                   feed_dict={rib_model.item_seq: item_seq,
                                              rib_model.len_seq: len_seq,
                                              rib_model.behavior_seq : behavior_seq,
                                              rib_model.target: target,
                                              rib_model.is_training:True
                })

                total_step+=1
                if total_step % 200 == 0:
                    print("the loss in %dth batch is: %f" % (total_step, loss))

            over_time_i = datetime.datetime.now()
            total_time_i = (over_time_i - start_time_i).total_seconds()
            print('total times: %s' % total_time_i)

            hit5, ndcg5, mrr5, hit10, ndcg10, mrr10, hit20, ndcg20, mrr20, hit50, ndcg50, mrr50 = evaluate_rib(sess,rib_model,data_directory,topk,have_dropout=True,have_user_emb=False,is_test=True)

    print("\nTraining completed. Loading best model for final evaluation...")
