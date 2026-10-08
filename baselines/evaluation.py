import pandas as pd
import numpy as np
import math
from utility import calculate_hit_ndcg
from utility import calculate_mrr
import os
import datetime


def evaluate_ranking(ser_rec, ser_real, num_k, rank_limited=True):


    precision = [0] * (num_k + 1)
    recall = [0] * (num_k + 1)
    F1 = [0] * (num_k + 1)
    NDCG = [0] * (num_k + 1)
    one_call = [0] * (num_k + 1)
    mrr = 0


    for u, real_list in ser_real.iteritems():
        rec_list = ser_rec[u]


        precision_u = [0] * (num_k + 1)
        recall_u = [0] * (num_k + 1)
        F1_u = [0] * (num_k + 1)
        DCG_u = [0] * (num_k + 1)
        NDCG_u = [0] * (num_k + 1)
        one_call_u = [0] * (num_k + 1)
        rr_u = 0

        DCG_best = [0] * (num_k + 1)
        for j in range(1, num_k + 1):
            DCG_best[j] = DCG_best[j - 1]
            if len(real_list) >= j:
                DCG_best[j] += 1 / math.log(j + 1)


        hit_sum = 0
        p = 0
        for j in range(1, num_k + 1):
            DCG_u[j] = DCG_u[j - 1]

            if rec_list[j - 1] in real_list:
                hit_sum += 1
                DCG_u[j] += 1 / math.log(j + 1)
                if p == 0:
                    p = j


            precision_u[j] = hit_sum / j
            recall_u[j] = hit_sum / len(real_list)
            if (precision_u[j] + recall_u[j]) != 0:
                F1_u[j] = 2 * (precision_u[j] * recall_u[j] / (precision_u[j] + recall_u[j]))
            NDCG_u[j] = DCG_u[j] / DCG_best[j]
            one_call_u[j] = 1 if hit_sum > 0 else 0


            precision[j] += precision_u[j]
            recall[j] += recall_u[j]
            F1[j] += F1_u[j]
            NDCG[j] += NDCG_u[j]
            one_call[j] += one_call_u[j]


        if not rank_limited:
            if p == 0:
                for j in range(num_k + 1, len(rec_list)):
                    if rec_list[j - 1] in real_list:
                        p = j
                        break


        if p != 0:
            rr_i = 1 / p
            mrr += rr_i


    print("k | Precision@k | Recall@k | F1@k | NDCG@k | one_call@k | MRR")
    for j in range(1, num_k + 1):
        if (j % 5 == 0):
            precision[j] = precision[j] / len(ser_real)
            recall[j] = recall[j] / len(ser_real)
            F1[j] = F1[j] / len(ser_real)
            NDCG[j] = NDCG[j] / len(ser_real)
            one_call[j] = one_call[j] / len(ser_real)

            if (j < num_k):
                print("%d %.8f %.8f %.8f %.8f %.8f" % (j, precision[j], recall[j], F1[j], NDCG[j], one_call[j]))
            else:
                mrr = mrr / len(ser_real)
                print("%d %.8f %.8f %.8f %.8f %.8f %.8f \n" % (j, precision[j], recall[j], F1[j], NDCG[j], one_call[j], mrr))

    return precision[num_k], recall[num_k], F1[num_k], NDCG[num_k], one_call[num_k], mrr


def evaluate_NDCG(ser_rec, ser_real, num_k):


    NDCG = [0] * (num_k + 1)


    for u, real_list in ser_real.iteritems():
        rec_list = ser_rec[u]


        DCG_u = [0] * (num_k + 1)
        NDCG_u = [0] * (num_k + 1)

        DCG_best = [0] * (num_k + 1)
        for j in range(1, num_k + 1):
            DCG_best[j] = DCG_best[j - 1]
            if len(real_list) >= j:
                DCG_best[j] += 1 / math.log2(j + 1)


        for j in range(1, num_k + 1):
            DCG_u[j] = DCG_u[j - 1]

            if rec_list[j - 1] in real_list:
                DCG_u[j] += 1 / math.log2(j + 1)


            NDCG_u[j] = DCG_u[j] / DCG_best[j]


            NDCG[j] += NDCG_u[j]


    return NDCG[num_k] / len(ser_real)


def evaluate_one_call(ser_rec, ser_real, num_k):


    one_call = [0] * (num_k + 1)


    for u, real_list in ser_real.iteritems():
        rec_list = ser_rec[u]


        one_call_u = [0] * (num_k + 1)


        hit_sum = 0
        for j in range(1, num_k + 1):
            if rec_list[j - 1] in real_list:
                hit_sum += 1


            one_call_u[j] = 1 if hit_sum > 0 else 0


            one_call[j] += one_call_u[j]


    return one_call[num_k] / len(ser_real)


def evaluate_origin(sess, model, data_directory, topk, have_dropout, have_user_emb, is_test, type='all'):
    start_time = datetime.datetime.now()
    if is_test == False:
        eval_sessions = pd.read_pickle(os.path.join(data_directory, 'val.df'))
    else:
        if type == 'all':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test.df'))
        elif type == 'clicked':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test_click.df'))
        elif type == 'unclicked':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test_unclick.df'))
    batch = 100
    hit_purchase = [0, 0, 0]
    ndcg_purchase = [0, 0, 0]
    mrr_purchase = [0, 0, 0]
    user_num = eval_sessions.shape[0]
    for start in range(0, user_num, batch):
        end = start + batch if start + batch < user_num else user_num
        if start % 5000 == 0:
            print("test progress:{}/{}".format(start, user_num))
        batch_df = eval_sessions.iloc[start:end]
        item_seq = batch_df['init_item_seq'].values.tolist()
        len_seq = batch_df['len_seq'].values.tolist()
        uid_seq = batch_df['uid'].values.tolist()
        feed_dict = {model.item_seq: item_seq, model.len_seq: len_seq}
        if have_dropout == True: feed_dict[model.is_training] = False
        if have_user_emb == True: feed_dict[model.uid_seq] = uid_seq
        prediction = sess.run(model.output, feed_dict=feed_dict)
        predict_list = np.argsort(prediction)
        true_items = batch_df['target'].values.tolist()
        calculate_hit_ndcg(predict_list, topk, true_items, hit_purchase, ndcg_purchase)
    for i in range(len(topk)):
        hit = hit_purchase[i] / user_num
        ndcg = ndcg_purchase[i] / user_num
        print("k | hit@k | ndcg@k")
        print("%d %.8f %.8f" % (topk[i], hit, ndcg))
    print('#############################################################')
    over_time = datetime.datetime.now()
    total_time = (over_time - start_time).total_seconds()
    print('total times: %s' % total_time)
    return hit_purchase[0] / user_num, ndcg_purchase[0] / user_num, \
           hit_purchase[1] / user_num, ndcg_purchase[1] / user_num, \
           hit_purchase[2] / user_num, ndcg_purchase[2] / user_num,


def evaluate_bt(sess, model, data_directory, topk, have_dropout, have_user_emb, is_test, type='all'):
    start_time = datetime.datetime.now()
    if is_test == False:
        eval_sessions = pd.read_pickle(os.path.join(data_directory, 'val.df'))
    else:
        if type == 'all':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test.df'))
        elif type == 'clicked':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test_click.df'))
        elif type == 'unclicked':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test_unclick.df'))
    batch = 100
    hit_purchase = [0, 0, 0]
    ndcg_purchase = [0, 0, 0]
    mrr_purchase = [0, 0, 0]
    user_num = eval_sessions.shape[0]
    for start in range(0, user_num, batch):
        end = start + batch if start + batch < user_num else user_num
        if start % 5000 == 0:
            print("test progress:{}/{}".format(start, user_num))
        batch_df = eval_sessions.iloc[start:end]
        item_seq = batch_df['init_item_seq'].values.tolist()
        len_seq = batch_df['len_seq'].values.tolist()
        uid_seq = batch_df['uid'].values.tolist()
        next_behavior = np.ones_like(len_seq)

        feed_dict = {model.item_seq: item_seq, model.len_seq: len_seq, model.next_behaviors_input: next_behavior}
        if have_dropout == True: feed_dict[model.is_training] = False
        if have_user_emb == True: feed_dict[model.uid_seq] = uid_seq
        prediction = sess.run(model.output, feed_dict=feed_dict)
        predict_list = np.argsort(prediction)
        true_items = batch_df['target'].values.tolist()
        calculate_hit_ndcg(predict_list, topk, true_items, hit_purchase, ndcg_purchase)


    for i in range(len(topk)):
        hit = hit_purchase[i] / user_num
        ndcg = ndcg_purchase[i] / user_num
        print("k | hit@k | ndcg@k")
        print("%d %.8f %.8f  " % (topk[i], hit, ndcg))
    print('#############################################################')
    over_time = datetime.datetime.now()
    total_time = (over_time - start_time).total_seconds()
    print('total times: %s' % total_time)
    return hit_purchase[0] / user_num, ndcg_purchase[0] / user_num, \
           hit_purchase[1] / user_num, ndcg_purchase[1] / user_num, \
           hit_purchase[2] / user_num, ndcg_purchase[2] / user_num,


def evaluate_att(sess, model, data_directory, topk, have_dropout, have_user_emb, is_test, type='all'):
    start_time = datetime.datetime.now()
    if is_test == False:
        eval_sessions = pd.read_pickle(os.path.join(data_directory, 'val.df'))
    else:
        if type == 'all':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test.df'))
        elif type == 'clicked':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test_click.df'))
        elif type == 'unclicked':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test_unclick.df'))

    batch = 100
    hit_purchase = [0, 0, 0]
    ndcg_purchase = [0, 0, 0]
    mrr_purchase = [0, 0, 0]
    user_num = eval_sessions.shape[0]
    for start in range(0, user_num, batch):
        end = start + batch if start + batch < user_num else user_num
        if start % 5000 == 0:
            print("test progress:{}/{}".format(start, user_num))
        batch_df = eval_sessions.iloc[start:end]
        item_seq = batch_df['init_item_seq'].values.tolist()
        behavior_seq = batch_df['init_behavior_seq'].values.tolist()
        len_seq = batch_df['len_seq'].values.tolist()
        uid_seq = batch_df['uid'].values.tolist()
        next_behavior = np.ones_like(len_seq)
        position_info = np.expand_dims(np.arange(0, len(item_seq[0])), axis=0).repeat(len(len_seq), axis=0)
        feed_dict = {
            model.item_seq: item_seq,
            model.len_seq: len_seq,
            model.behavior_seq: behavior_seq,
            model.next_behaviors_input: next_behavior,
            model.position: position_info
        }
        if have_dropout == True: feed_dict[model.is_training] = False
        if have_user_emb == True: feed_dict[model.uid_seq] = uid_seq
        prediction = sess.run(model.output, feed_dict=feed_dict)
        predict_list = np.argsort(prediction)
        true_items = batch_df['target'].values.tolist()
        calculate_hit_ndcg(predict_list, topk, true_items, hit_purchase, ndcg_purchase)
    for i in range(len(topk)):
        hit = hit_purchase[i] / user_num
        ndcg = ndcg_purchase[i] / user_num
        print("k | hit@k | ndcg@k")
        print("%d %.8f %.8f " % (topk[i], hit, ndcg))
    print('#############################################################')
    over_time = datetime.datetime.now()
    total_time = (over_time - start_time).total_seconds()
    print('total times: %s' % total_time)
    return hit_purchase[0] / user_num, ndcg_purchase[0] / user_num, \
           hit_purchase[1] / user_num, ndcg_purchase[1] / user_num, \
           hit_purchase[2] / user_num, ndcg_purchase[2] / user_num,


def evaluate_att_with_bt(sess, model, data_directory, topk, have_dropout, have_user_emb, is_test, type='all'):
    start_time = datetime.datetime.now()
    if is_test == False:
        eval_sessions = pd.read_pickle(os.path.join(data_directory, 'val.df'))
    else:
        if type == 'all':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test.df'))
        elif type == 'clicked':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test_click.df'))
        elif type == 'unclicked':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test_unclick.df'))
    batch = 100
    hit_purchase = [0, 0, 0]
    ndcg_purchase = [0, 0, 0]
    mrr_purchase = [0, 0, 0]
    user_num = eval_sessions.shape[0]
    for start in range(0, user_num, batch):
        end = start + batch if start + batch < user_num else user_num
        if start % 5000 == 0:
            print("test progress:{}/{}".format(start, user_num))
        batch_df = eval_sessions.iloc[start:end]
        item_seq = batch_df['init_item_seq'].values.tolist()
        behavior_seq = batch_df['init_behavior_seq'].values.tolist()
        len_seq = batch_df['len_seq'].values.tolist()
        uid_seq = batch_df['uid'].values.tolist()
        next_behavior = np.ones_like(len_seq)
        position_info = np.expand_dims(np.arange(0, len(item_seq[0])), axis=0).repeat(len(len_seq), axis=0)

        feed_dict = {
            model.item_seq: item_seq,
            model.len_seq: len_seq,
            model.behavior_seq: behavior_seq,
            model.next_behaviors_input: next_behavior,
            model.position: position_info
        }
        if have_dropout == True: feed_dict[model.is_training] = False
        if have_user_emb == True: feed_dict[model.uid_seq] = uid_seq
        prediction = sess.run(model.output, feed_dict=feed_dict)
        predict_list = np.argsort(prediction)
        true_items = batch_df['target'].values.tolist()


        calculate_hit_ndcg(predict_list, topk, true_items, hit_purchase, ndcg_purchase)
    for i in range(len(topk)):
        hit = hit_purchase[i] / user_num
        ndcg = ndcg_purchase[i] / user_num
        print("k | hit@k | ndcg@k")
        print("%d %.8f %.8f" % (topk[i], hit, ndcg))
    print('#############################################################')
    over_time = datetime.datetime.now()
    total_time = (over_time - start_time).total_seconds()
    print('total times: %s' % total_time)
    return hit_purchase[0] / user_num, ndcg_purchase[0] / user_num, \
           hit_purchase[1] / user_num, ndcg_purchase[1] / user_num, \
           hit_purchase[2] / user_num, ndcg_purchase[2] / user_num,


def evaluate_att_with_bt_reverse(sess, model, data_directory, topk, have_dropout, have_user_emb, is_test, type='all'):
    start_time = datetime.datetime.now()
    if is_test == False:
        eval_sessions = pd.read_pickle(os.path.join(data_directory, 'val.df'))
    else:
        if type == 'all':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test.df'))
        elif type == 'clicked':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test_click.df'))
        elif type == 'unclicked':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test_unclick.df'))
    batch = 100
    hit_purchase = [0, 0, 0]
    ndcg_purchase = [0, 0, 0]
    mrr_purchase = [0, 0, 0]
    user_num = eval_sessions.shape[0]
    for start in range(0, user_num, batch):
        end = start + batch if start + batch < user_num else user_num
        if start % 5000 == 0:
            print("test progress:{}/{}".format(start, user_num))
        batch_df = eval_sessions.iloc[start:end]
        item_seq = batch_df['init_item_seq'].values.tolist()
        behavior_seq = batch_df['init_behavior_seq'].values.tolist()
        len_seq = batch_df['len_seq'].values.tolist()
        uid_seq = batch_df['uid'].values.tolist()
        next_behavior = np.ones_like(len_seq)
        position_info = np.zeros((len(len_seq), len(item_seq[0])))
        for idx, len_ in enumerate(len_seq):
            position_info[idx][:len_] = range(len_, 0, -1)

        feed_dict = {
            model.item_seq: item_seq,
            model.len_seq: len_seq,
            model.behavior_seq: behavior_seq,
            model.next_behaviors_input: next_behavior,
            model.position: position_info
        }
        if have_dropout == True: feed_dict[model.is_training] = False
        if have_user_emb == True: feed_dict[model.uid_seq] = uid_seq
        prediction = sess.run(model.output, feed_dict=feed_dict)

        predict_list = np.argsort(prediction)
        true_items = batch_df['target'].values.tolist()


        calculate_hit_ndcg(predict_list, topk, true_items, hit_purchase, ndcg_purchase)
    for i in range(len(topk)):
        hit = hit_purchase[i] / user_num
        ndcg = ndcg_purchase[i] / user_num
        print("k | hit@k | ndcg@k")
        print("%d %.8f %.8f" % (topk[i], hit, ndcg))
    print('#############################################################')
    over_time = datetime.datetime.now()
    total_time = (over_time - start_time).total_seconds()
    print('total times: %s' % total_time)
    return hit_purchase[0] / user_num, ndcg_purchase[0] / user_num, \
           hit_purchase[1] / user_num, ndcg_purchase[1] / user_num, \
           hit_purchase[2] / user_num, ndcg_purchase[2] / user_num,


def evaluate_att_with_bt_no_p(sess, model, data_directory, topk, have_dropout, have_user_emb, is_test, type='all'):
    start_time = datetime.datetime.now()
    if is_test == False:
        eval_sessions = pd.read_pickle(os.path.join(data_directory, 'val.df'))
    else:
        if type == 'all':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test.df'))
        elif type == 'clicked':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test_click.df'))
        elif type == 'unclicked':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test_unclick.df'))
    batch = 100
    hit_purchase = [0, 0, 0]
    ndcg_purchase = [0, 0, 0]
    mrr_purchase = [0, 0, 0]
    user_num = eval_sessions.shape[0]
    for start in range(0, user_num, batch):
        end = start + batch if start + batch < user_num else user_num
        if start % 5000 == 0:
            print("test progress:{}/{}".format(start, user_num))
        batch_df = eval_sessions.iloc[start:end]
        item_seq = batch_df['init_item_seq'].values.tolist()
        behavior_seq = batch_df['init_behavior_seq'].values.tolist()
        len_seq = batch_df['len_seq'].values.tolist()
        uid_seq = batch_df['uid'].values.tolist()
        next_behavior = np.ones_like(len_seq)

        feed_dict = {
            model.item_seq: item_seq,
            model.len_seq: len_seq,
            model.behavior_seq: behavior_seq,
            model.next_behaviors_input: next_behavior
        }
        if have_dropout == True: feed_dict[model.is_training] = False
        if have_user_emb == True: feed_dict[model.uid_seq] = uid_seq
        prediction = sess.run(model.output, feed_dict=feed_dict)

        predict_list = np.argsort(prediction)
        true_items = batch_df['target'].values.tolist()


        calculate_hit_ndcg(predict_list, topk, true_items, hit_purchase, ndcg_purchase)
    for i in range(len(topk)):
        hit = hit_purchase[i] / user_num
        ndcg = ndcg_purchase[i] / user_num
        print("k | hit@k | ndcg@k")
        print("%d %.8f %.8f" % (topk[i], hit, ndcg))
    print('#############################################################')
    over_time = datetime.datetime.now()
    total_time = (over_time - start_time).total_seconds()
    print('total times: %s' % total_time)
    return hit_purchase[0] / user_num, ndcg_purchase[0] / user_num, \
           hit_purchase[1] / user_num, ndcg_purchase[1] / user_num, \
           hit_purchase[2] / user_num, ndcg_purchase[2] / user_num,


def evaluate_BAR(sess,model,data_directory,topk,have_dropout,is_test,type='all'):
    start_time = datetime.datetime.now()
    if is_test == False:
        eval_sessions=pd.read_pickle(os.path.join(data_directory, 'val.df'))
    else:
        if type == 'all':
            eval_sessions=pd.read_pickle(os.path.join(data_directory, 'test.df'))
        elif type == 'clicked':
            eval_sessions=pd.read_pickle(os.path.join(data_directory, 'test_click.df'))
        elif type =='unclicked':
            eval_sessions=pd.read_pickle(os.path.join(data_directory, 'test_unclick.df'))
    batch = 100
    hit_purchase=[0, 0, 0, 0]
    ndcg_purchase=[0, 0, 0, 0]
    mrr_purchase = [0, 0, 0, 0]
    user_num = eval_sessions.shape[0]
    for start in range(0,user_num,batch):
        end = start + batch if start + batch < user_num else user_num
        if start % 5000 == 0:
            print("test progress:{}/{}".format(start,user_num))
        batch_df = eval_sessions.iloc[start:end]
        item_seq = batch_df['init_item_seq'].values.tolist()
        behavior_seq = batch_df['init_behavior_seq'].values.tolist()
        len_seq = batch_df['len_seq'].values.tolist()
        uid_seq = batch_df['uid'].values.tolist()
        next_behavior = np.ones_like(len_seq)
        position_info = np.zeros((len(len_seq), len(item_seq[0])))
        for idx, len_ in enumerate(len_seq):
            position_info[idx][:len_] = range(len_, 0, -1)

        feed_dict = {model.item_seq: item_seq,
                     model.len_seq:len_seq,
                     model.behavior_seq : behavior_seq,
                     model.next_behaviors_input : next_behavior,
                     model.position : position_info}
        if have_dropout == True: feed_dict[model.is_training] = False
        prediction=sess.run(model.output, feed_dict=feed_dict)

        predict_list = np.argsort(prediction)
        true_items = batch_df['target'].values.tolist()

        calculate_hit_ndcg(predict_list, topk, true_items, hit_purchase, ndcg_purchase)
        calculate_mrr(predict_list, topk, true_items, mrr_purchase)
    for i in range(len(topk)):
        hit=hit_purchase[i]/ user_num
        ndcg=ndcg_purchase[i]/user_num
        mrr = mrr_purchase[i] / user_num


        print("k | hit@k | ndcg@k | mrr@k")
        print("%d %.8f %.8f %.8f\n" % (topk[i], hit, ndcg, mrr))
    print('#############################################################')
    over_time = datetime.datetime.now()
    total_time = (over_time - start_time).total_seconds()
    print('total times: %s' % total_time)
    return hit_purchase[0] / user_num, ndcg_purchase[0] / user_num, mrr_purchase[0] / user_num, \
           hit_purchase[1] / user_num, ndcg_purchase[1] / user_num, mrr_purchase[1] / user_num, \
           hit_purchase[2] / user_num, ndcg_purchase[2] / user_num, mrr_purchase[2] / user_num, \
           hit_purchase[3] / user_num, ndcg_purchase[3] / user_num, mrr_purchase[3] / user_num


def evaluate_rib(sess, model, data_directory, topk, have_dropout, have_user_emb, is_test, type='all'):
    start_time = datetime.datetime.now()
    if is_test == False:
        eval_sessions = pd.read_pickle(os.path.join(data_directory, 'val.df'))
    else:
        if type == 'all':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test.df'))
        elif type == 'clicked':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test_click.df'))
        elif type == 'unclicked':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test_unclick.df'))

    batch = 100
    hit_purchase = [0, 0, 0, 0]
    ndcg_purchase = [0, 0, 0, 0]
    mrr_purchase = [0, 0, 0, 0]
    user_num = eval_sessions.shape[0]
    for start in range(0, user_num, batch):
        end = start + batch if start + batch < user_num else user_num
        if start % 5000 == 0:
            print("test progress:{}/{}".format(start, user_num))
        batch_df = eval_sessions.iloc[start:end]
        item_seq = batch_df['init_item_seq'].values.tolist()
        behavior_seq = batch_df['init_behavior_seq'].values.tolist()
        len_seq = batch_df['len_seq'].values.tolist()
        uid_seq = batch_df['uid'].values.tolist()
        feed_dict = {model.item_seq: item_seq, model.len_seq: len_seq, model.behavior_seq: behavior_seq}
        if have_dropout == True: feed_dict[model.is_training] = False
        if have_user_emb == True: feed_dict[model.uid_seq] = uid_seq
        prediction = sess.run(model.output, feed_dict=feed_dict)
        predict_list = np.argsort(prediction)
        true_items = batch_df['target'].values.tolist()
        calculate_hit_ndcg(predict_list, topk, true_items, hit_purchase, ndcg_purchase)
        calculate_mrr(predict_list, topk, true_items, mrr_purchase)
    for i in range(len(topk)):
        hit = hit_purchase[i] / user_num
        ndcg = ndcg_purchase[i] / user_num
        mrr = mrr_purchase[i] / user_num
        print("k | hit@k | ndcg@k | mrr@k")
        print("%d %.8f %.8f %.8f\n" % (topk[i], hit, ndcg, mrr))


    print('#############################################################')
    over_time = datetime.datetime.now()
    total_time = (over_time - start_time).total_seconds()
    print('total times: %s' % total_time)
    return hit_purchase[0] / user_num, ndcg_purchase[0] / user_num, mrr_purchase[0] / user_num, \
           hit_purchase[1] / user_num, ndcg_purchase[1] / user_num, mrr_purchase[1] / user_num, \
           hit_purchase[2] / user_num, ndcg_purchase[2] / user_num, mrr_purchase[2] / user_num, \
           hit_purchase[3] / user_num, ndcg_purchase[3] / user_num, mrr_purchase[3] / user_num

def evaluate_nextip(sess, model, data_directory, topk, have_dropout, have_user_emb, is_test, type='all'):
    start_time = datetime.datetime.now()
    if is_test == False:
        eval_sessions = pd.read_pickle(os.path.join(data_directory, 'val.df'))
    else:
        if type == 'all':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test.df'))
        elif type == 'clicked':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test_click.df'))
        elif type == 'unclicked':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test_unclick.df'))

    batch = 100
    hit_purchase = [0, 0, 0]
    ndcg_purchase = [0, 0, 0]
    mrr_purchase = [0, 0, 0]
    user_num = eval_sessions.shape[0]
    for start in range(0, user_num, batch):
        end = start + batch if start + batch < user_num else user_num
        if start % 5000 == 0:
            print("test progress:{}/{}".format(start, user_num))
        batch_df = eval_sessions.iloc[start:end]
        item_seq = batch_df['init_item_seq'].values.tolist()
        behavior_seq = batch_df['init_behavior_seq'].values.tolist()
        len_seq = batch_df['len_seq'].values.tolist()
        uid_seq = batch_df['uid'].values.tolist()
        next_behavior = np.ones_like(len_seq)
        next_behavior = [[x] for x in next_behavior]

        feed_dict = {model.input_id: item_seq, model.target_behavior: next_behavior, model.behavior_id: behavior_seq}
        if have_dropout == True: feed_dict[model.is_training] = False
        if have_user_emb == True: feed_dict[model.user_id] = uid_seq
        prediction = sess.run(model.output, feed_dict=feed_dict)
        predict_list = np.argsort(prediction)
        true_items = batch_df['target'].values.tolist()
        calculate_hit_ndcg(predict_list, topk, true_items, hit_purchase, ndcg_purchase)
    for i in range(len(topk)):
        hit = hit_purchase[i] / user_num
        ndcg = ndcg_purchase[i] / user_num
        print("k | hit@k | ndcg@k")
        print("%d %.8f %.8f " % (topk[i], hit, ndcg))
    print('#############################################################')
    over_time = datetime.datetime.now()
    total_time = (over_time - start_time).total_seconds()
    print('total times: %s' % total_time)
    return hit_purchase[0] / user_num, ndcg_purchase[0] / user_num, \
           hit_purchase[1] / user_num, ndcg_purchase[1] / user_num, \
           hit_purchase[2] / user_num, ndcg_purchase[2] / user_num,


def evaluate_rib_dup_nondup(sess, model, data_directory, topk, have_dropout, have_user_emb, is_test, type='all'):
    start_time = datetime.datetime.now()

    if is_test == False:
        eval_sessions = pd.read_pickle(os.path.join(data_directory, 'val.df'))
    else:
        if type == 'all':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test.df'))
        elif type == 'clicked':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test_click.df'))
        elif type == 'unclicked':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test_unclick.df'))

    batch = 100
    hit_purchase = [0] * len(topk)
    ndcg_purchase = [0] * len(topk)
    mrr_purchase = [0] * len(topk)
    dup_hit_purchase = [0] * len(topk)
    dup_ndcg_purchase = [0] * len(topk)
    nondup_hit_purchase = [0] * len(topk)
    nondup_ndcg_purchase = [0] * len(topk)
    user_num = eval_sessions.shape[0]
    dup_num = 0
    nondup_num = 0

    for start in range(0, user_num, batch):
        end = start + batch if start + batch < user_num else user_num
        if start % 5000 == 0:
            print("test progress:{}/{}".format(start, user_num))
        batch_df = eval_sessions.iloc[start:end]
        item_seq = batch_df['init_item_seq'].values.tolist()
        behavior_seq = batch_df['init_behavior_seq'].values.tolist()
        len_seq = batch_df['len_seq'].values.tolist()
        uid_seq = batch_df['uid'].values.tolist()
        target = batch_df['target'].values.tolist()

        feed_dict = {model.item_seq: item_seq, model.len_seq: len_seq, model.behavior_seq: behavior_seq}
        if have_dropout == True: feed_dict[model.is_training] = False
        if have_user_emb == True: feed_dict[model.uid_seq] = uid_seq
        prediction = sess.run(model.output, feed_dict=feed_dict)


        batch_prediction_dup_scores = []
        batch_prediction_nondup_scores = []
        batch_target_dup = []
        batch_target_nondup = []

        for i in range(len(item_seq)):

            if target[i] in item_seq[i]:
                batch_prediction_dup_scores.append(prediction[i])
                batch_target_dup.append(target[i])
                dup_num += 1
            else:
                batch_prediction_nondup_scores.append(prediction[i])
                batch_target_nondup.append(target[i])
                nondup_num += 1


        predict_list = np.argsort(prediction)


        if len(batch_prediction_dup_scores) > 0:


            sorted_prediction_dup = np.argsort(-np.array(batch_prediction_dup_scores), axis=1)
            calculate_hit_ndcg(sorted_prediction_dup, topk, batch_target_dup, dup_hit_purchase, dup_ndcg_purchase)


        if len(batch_prediction_nondup_scores) > 0:

            sorted_prediction_nondup = np.argsort(-np.array(batch_prediction_nondup_scores), axis=1)
            calculate_hit_ndcg(sorted_prediction_nondup, topk, batch_target_nondup, nondup_hit_purchase,
                               nondup_ndcg_purchase)


        calculate_hit_ndcg(predict_list, topk, target, hit_purchase, ndcg_purchase)
        calculate_mrr(predict_list, topk, target, mrr_purchase)


    for i in range(len(topk)):
        hit = hit_purchase[i] / user_num
        ndcg = ndcg_purchase[i] / user_num
        mrr = mrr_purchase[i] / user_num
        print("k | hit@k | ndcg@k | mrr@k")
        print("%d %.8f %.8f %.8f\n" % (topk[i], hit, ndcg, mrr))

    for i in range(len(topk)):

        dup_hit = dup_hit_purchase[i] / (dup_num if dup_num > 0 else 1)
        dup_ndcg = dup_ndcg_purchase[i] / (dup_num if dup_num > 0 else 1)
        print("duplicate num present \n")
        print("k | hit@k | ndcg@k")
        print("%d %.8f %.8f \n" % (topk[i], dup_hit, dup_ndcg))

    for i in range(len(topk)):

        nondup_hit = nondup_hit_purchase[i] / (nondup_num if nondup_num > 0 else 1)
        nondup_ndcg = nondup_ndcg_purchase[i] / (nondup_num if nondup_num > 0 else 1)
        print("non duplicate the epoch :\n ")
        print("%d %.8f %.8f \n" % (topk[i], nondup_hit, nondup_ndcg))
    print('#############################################################')
    over_time = datetime.datetime.now()
    total_time = (over_time - start_time).total_seconds()
    print('total times: %s' % total_time)
    return hit_purchase[0] / user_num, ndcg_purchase[0] / user_num, mrr_purchase[0] / user_num, \
           hit_purchase[1] / user_num, ndcg_purchase[1] / user_num, mrr_purchase[1] / user_num, \
           hit_purchase[2] / user_num, ndcg_purchase[2] / user_num, mrr_purchase[2] / user_num, \
           hit_purchase[3] / user_num, ndcg_purchase[3] / user_num, mrr_purchase[3] / user_num


def evaluate_RLBL(sess, model, data_directory, topk, have_dropout, have_user_emb, is_test, type='all'):
    start_time = datetime.datetime.now()
    if is_test == False:
        eval_sessions = pd.read_pickle(os.path.join(data_directory, 'val.df'))
    else:
        if type == 'all':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test.df'))
        elif type == 'clicked':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test_click.df'))
        elif type == 'unclicked':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test_unclick.df'))

    batch = 100
    hit_purchase = [0, 0, 0, 0]
    ndcg_purchase = [0, 0, 0, 0]
    mrr_purchase = [0, 0, 0, 0]
    user_num = eval_sessions.shape[0]
    for start in range(0, user_num, batch):
        end = start + batch if start + batch < user_num else user_num
        if start % 5000 == 0:
            print("test progress:{}/{}".format(start, user_num))
        batch_df = eval_sessions.iloc[start:end]
        item_seq = batch_df['init_item_seq'].values.tolist()
        behavior_seq = batch_df['init_behavior_seq'].values.tolist()
        len_seq = batch_df['len_seq'].values.tolist()
        uid_seq = batch_df['uid'].values.tolist()
        next_behavior = np.ones_like(len_seq)
        position_info = np.zeros((len(len_seq), len(item_seq[0])))
        for idx, len_i in enumerate(len_seq):
            for s in range(0, len(item_seq[0]), model.window_size):
                position_info[idx][s:s + model.window_size] = range(0, model.window_size)
        feed_dict = {
            model.item_seq: item_seq,
            model.len_seq: len_seq,
            model.position: position_info,
            model.target_behavior: next_behavior,
            model.behavior_seq: behavior_seq
        }
        if have_dropout == True: feed_dict[model.is_training] = False
        if have_user_emb == True: feed_dict[model.uid_seq] = uid_seq
        prediction = sess.run(model.output, feed_dict=feed_dict)
        predict_list = np.argsort(prediction)
        true_items = batch_df['target'].values.tolist()
        calculate_hit_ndcg(predict_list, topk, true_items, hit_purchase, ndcg_purchase)
        calculate_mrr(predict_list, topk, true_items, mrr_purchase)
    for i in range(len(topk)):
        hit = hit_purchase[i] / user_num
        ndcg = ndcg_purchase[i] / user_num
        mrr = mrr_purchase[i] / user_num
        print("k | hit@k | ndcg@k | mrr@k")
        print("%d %.8f %.8f %.8f\n" % (topk[i], hit, ndcg, mrr))
    print('#############################################################')
    over_time = datetime.datetime.now()
    total_time = (over_time - start_time).total_seconds()
    print('total times: %s' % total_time)
    return hit_purchase[0] / user_num, ndcg_purchase[0] / user_num, mrr_purchase[0] / user_num, \
           hit_purchase[1] / user_num, ndcg_purchase[1] / user_num, mrr_purchase[1] / user_num, \
           hit_purchase[2] / user_num, ndcg_purchase[2] / user_num, mrr_purchase[2] / user_num, \
           hit_purchase[3] / user_num, ndcg_purchase[3] / user_num, mrr_purchase[3] / user_num


def evaluate_RLBL_book(sess, model, data_directory, topk, have_dropout, have_user_emb, is_test, type='all'):
    start_time = datetime.datetime.now()
    if is_test == False:
        eval_sessions = pd.read_pickle(os.path.join(data_directory, 'val.df'))
    else:
        if type == 'all':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test.df'))
        elif type == 'clicked':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test_click.df'))
        elif type == 'unclicked':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test_unclick.df'))

    batch = 100
    hit_purchase = [0, 0, 0]
    ndcg_purchase = [0, 0, 0]
    mrr_purchase = [0, 0, 0]
    user_num = eval_sessions.shape[0]
    for start in range(0, user_num, batch):
        end = start + batch if start + batch < user_num else user_num
        if start % 5000 == 0:
            print("test progress:{}/{}".format(start, user_num))
        batch_df = eval_sessions.iloc[start:end]
        item_seq = batch_df['init_item_seq'].values.tolist()
        behavior_seq = batch_df['init_behavior_seq'].values.tolist()
        len_seq = batch_df['len_seq'].values.tolist()
        uid_seq = batch_df['uid'].values.tolist()
        next_behavior = np.ones_like(len_seq)
        position_info = np.zeros((len(len_seq), len(item_seq[0])))
        for idx, len_i in enumerate(len_seq):
            for s in range(0, len(item_seq[0]), model.window_size):
                position_info[idx][s:s + model.window_size] = range(0, model.window_size)
        feed_dict = {
            model.item_seq: item_seq,
            model.len_seq: len_seq,
            model.position: position_info,
            model.target_behavior: next_behavior,
            model.behavior_seq: behavior_seq
        }
        if have_dropout == True: feed_dict[model.is_training] = False
        if have_user_emb == True: feed_dict[model.uid_seq] = uid_seq
        prediction = sess.run(model.output, feed_dict=feed_dict)
        predict_list = np.argsort(prediction)
        true_items = batch_df['target'].values.tolist()
        calculate_hit_ndcg(predict_list, topk, true_items, hit_purchase, ndcg_purchase)
    for i in range(len(topk)):
        hit = hit_purchase[i] / user_num
        ndcg = ndcg_purchase[i] / user_num
        print("k | hit@k | ndcg@k")
        print("%d %.8f %.8f " % (topk[i], hit, ndcg))
    print('#############################################################')
    over_time = datetime.datetime.now()
    total_time = (over_time - start_time).total_seconds()
    print('total times: %s' % total_time)
    return hit_purchase[0] / user_num, ndcg_purchase[0] / user_num, \
           hit_purchase[1] / user_num, ndcg_purchase[1] / user_num, \
           hit_purchase[2] / user_num, ndcg_purchase[2] / user_num,


def evaluate_RLBL_3(sess, model, data_directory, topk, have_dropout, have_user_emb, is_test, type='all'):
    start_time = datetime.datetime.now()
    if is_test == False:
        eval_sessions = pd.read_pickle(os.path.join(data_directory, 'val.df'))
    else:
        if type == 'all':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test.df'))
        elif type == 'clicked':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test_click.df'))
        elif type == 'unclicked':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test_unclick.df'))

    batch = 100
    hit_purchase = [0, 0, 0]
    ndcg_purchase = [0, 0, 0]
    mrr_purchase = [0, 0, 0]
    user_num = eval_sessions.shape[0]
    for start in range(0, user_num, batch):
        end = start + batch if start + batch < user_num else user_num
        if start % 5000 == 0:
            print("test progress:{}/{}".format(start, user_num))
        batch_df = eval_sessions.iloc[start:end]
        item_seq = batch_df['init_item_seq'].values.tolist()
        behavior_seq = batch_df['init_behavior_seq'].values.tolist()
        len_seq = batch_df['len_seq'].values.tolist()
        uid_seq = batch_df['uid'].values.tolist()
        position_info = np.zeros((len(len_seq), len(item_seq[0])))
        for idx, len_i in enumerate(len_seq):
            for s in range(0, len(item_seq[0]), model.window_size):
                position_info[idx][s:s + model.window_size] = range(0, model.window_size)
        feed_dict = {model.item_seq: item_seq, model.len_seq: len_seq, model.position: position_info, model.behavior_seq: behavior_seq}
        if have_dropout == True: feed_dict[model.is_training] = False
        if have_user_emb == True: feed_dict[model.uid_seq] = uid_seq
        prediction = sess.run(model.output, feed_dict=feed_dict)
        predict_list = np.argsort(prediction)

        true_items = batch_df['target'].values.tolist()
        calculate_hit_ndcg(predict_list, topk, true_items, hit_purchase, ndcg_purchase)
    for i in range(len(topk)):
        hit = hit_purchase[i] / user_num
        ndcg = ndcg_purchase[i] / user_num
        print("k | hit@k | ndcg@k")
        print("%d %.8f %.8f " % (topk[i], hit, ndcg))
    print('#############################################################')
    over_time = datetime.datetime.now()
    total_time = (over_time - start_time).total_seconds()
    print('total times: %s' % total_time)
    return hit_purchase[0] / user_num, ndcg_purchase[0] / user_num, \
           hit_purchase[1] / user_num, ndcg_purchase[1] / user_num, \
           hit_purchase[2] / user_num, ndcg_purchase[2] / user_num,


def evaluate_RLBL_6(sess, model, data_directory, topk, have_dropout, have_user_emb, is_test, type='all'):
    start_time = datetime.datetime.now()
    if is_test == False:
        eval_sessions = pd.read_pickle(os.path.join(data_directory, 'val.df'))
    else:
        if type == 'all':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test.df'))
        elif type == 'clicked':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test_click.df'))
        elif type == 'unclicked':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test_unclick.df'))

    batch = 100
    hit_purchase = [0, 0, 0]
    ndcg_purchase = [0, 0, 0]
    mrr_purchase = [0, 0, 0]
    user_num = eval_sessions.shape[0]
    for start in range(0, user_num, batch):
        end = start + batch if start + batch < user_num else user_num
        if start % 5000 == 0:
            print("test progress:{}/{}".format(start, user_num))
        batch_df = eval_sessions.iloc[start:end]
        item_seq = batch_df['init_item_seq'].values.tolist()
        behavior_seq = batch_df['init_behavior_seq'].values.tolist()
        len_seq = batch_df['len_seq'].values.tolist()
        uid_seq = batch_df['uid'].values.tolist()
        next_behavior = np.ones_like(len_seq)
        position_info = np.zeros((len(len_seq), len(item_seq[0])))
        for idx, len_i in enumerate(len_seq):
            for s in range(0, len(item_seq[0]), model.window_size):
                position_info[idx][s:s + model.window_size] = range(0, model.window_size)
        feed_dict = {
            model.item_seq: item_seq,
            model.len_seq: len_seq,
            model.position: position_info,
            model.target_behavior: next_behavior,
            model.behavior_seq: behavior_seq
        }
        if have_dropout == True: feed_dict[model.is_training] = False
        if have_user_emb == True: feed_dict[model.uid_seq] = uid_seq
        prediction = sess.run(model.output, feed_dict=feed_dict)
        predict_list = np.argsort(prediction)
        true_items = batch_df['target'].values.tolist()
        calculate_hit_ndcg(predict_list, topk, true_items, hit_purchase, ndcg_purchase)
    for i in range(len(topk)):
        hit = hit_purchase[i] / user_num
        ndcg = ndcg_purchase[i] / user_num
        print("k | hit@k | ndcg@k")
        print("%d %.8f %.8f " % (topk[i], hit, ndcg))
    print('#############################################################')
    over_time = datetime.datetime.now()
    total_time = (over_time - start_time).total_seconds()
    print('total times: %s' % total_time)
    return hit_purchase[0] / user_num, ndcg_purchase[0] / user_num, \
           hit_purchase[1] / user_num, ndcg_purchase[1] / user_num, \
           hit_purchase[2] / user_num, ndcg_purchase[2] / user_num,


def get_slice(inputs, padding):
    items, n_node, A_in, A_out, alias_inputs = [], [], [], [], []
    for u_input in inputs:

        n_node.append(len(np.unique(u_input)))

    max_n_node = np.max(n_node)
    for u_input in inputs:

        node = np.unique(u_input)


        items.append(node.tolist() + (max_n_node - len(node)) * [padding])

        u_A = np.zeros((max_n_node, max_n_node))
        for i in np.arange(len(u_input) - 1):
            if u_input[i + 1] == padding:
                break

            u = np.where(node == u_input[i])[0][0]
            v = np.where(node == u_input[i + 1])[0][0]
            u_A[u][v] += 1
        u_sum_in = np.sum(u_A, 0)
        u_sum_in[np.where(u_sum_in == 0)] = 1
        u_A_in = np.divide(u_A, u_sum_in)
        u_sum_out = np.sum(u_A, 1)
        u_sum_out[np.where(u_sum_out == 0)] = 1
        u_A_out = np.divide(u_A.transpose(), u_sum_out)

        A_in.append(u_A_in)

        A_out.append(u_A_out)
        alias_inputs.append([np.where(node == i)[0][0] for i in u_input])

    return np.array(A_in), np.array(A_out), np.array(alias_inputs), np.array(items)


def evaluate_msr(sess, model, data_directory, topk, item_num, have_dropout, have_user_emb, is_test, type='all'):
    start_time = datetime.datetime.now()
    if is_test == False:
        eval_sessions = pd.read_pickle(os.path.join(data_directory, 'val.df'))
    else:
        if type == 'all':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test.df'))
        elif type == 'clicked':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test_click.df'))
        elif type == 'unclicked':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test_unclick.df'))

    batch = 128
    bahavior_num = 2
    hit_purchase = [0, 0, 0]
    ndcg_purchase = [0, 0, 0]
    mrr_purchase = [0, 0, 0]
    user_num = eval_sessions.shape[0]
    for start in range(0, user_num, batch):
        end = start + batch if start + batch < user_num else user_num
        if start % (batch * 5) == 0:
            print("test progress:{}/{}".format(start, user_num))
        batch_df = eval_sessions.iloc[start:end]
        item_seq = batch_df['init_item_seq'].values.tolist()
        behavior_seq = batch_df['init_behavior_seq'].values.tolist()
        len_seq = batch_df['len_seq'].values.tolist()
        if len(item_seq) < batch:
            for _ in range(batch - len(item_seq)):
                item_seq.append(len(item_seq[0]) * [item_num])
                behavior_seq.append(len(item_seq[0]) * [bahavior_num])
                len_seq.append(1)
        A_in, A_out, alias_inputs, items = get_slice(item_seq, item_num)
        uid_seq = batch_df['uid'].values.tolist()
        feed_dict = {
            model.item_seq: item_seq,
            model.len_seq: len_seq,
            model.behavior_seq: behavior_seq,
            model.item: items,
            model.alias: alias_inputs,
            model.adj_in: A_in,
            model.adj_out: A_out,
        }
        if have_dropout == True: feed_dict[model.is_training] = False
        if have_user_emb == True: feed_dict[model.uid_seq] = uid_seq
        prediction = sess.run(model.output, feed_dict=feed_dict)
        prediction = prediction[:end - start]
        predict_list = np.argsort(prediction)
        true_items = batch_df['target'].values.tolist()
        calculate_hit_ndcg(predict_list, topk, true_items, hit_purchase, ndcg_purchase)
    for i in range(len(topk)):
        hit = hit_purchase[i] / user_num
        ndcg = ndcg_purchase[i] / user_num
        print("k | hit@k | ndcg@k")
        print("%d %.8f %.8f " % (topk[i], hit, ndcg))
    print('#############################################################')
    over_time = datetime.datetime.now()
    total_time = (over_time - start_time).total_seconds()
    print('total times: %s' % total_time)
    return hit_purchase[0] / user_num, ndcg_purchase[0] / user_num, \
           hit_purchase[1] / user_num, ndcg_purchase[1] / user_num, \
           hit_purchase[2] / user_num, ndcg_purchase[2] / user_num,


def evaluate_mgnn(sess, model, data_directory, topk, item_num, have_dropout, have_user_emb, is_test, type='all'):
    start_time = datetime.datetime.now()
    if is_test == False:
        eval_sessions = pd.read_pickle(os.path.join(data_directory, 'val.df'))
    else:
        if type == 'all':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test.df'))
        elif type == 'clicked':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test_click.df'))
        elif type == 'unclicked':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test_unclick.df'))

    batch = 512
    bahavior_num = 2
    hit_purchase = [0, 0, 0]
    ndcg_purchase = [0, 0, 0]
    mrr_purchase = [0, 0, 0]
    user_num = eval_sessions.shape[0]
    for start in range(0, user_num, batch):
        end = start + batch if start + batch < user_num else user_num
        if start % (batch * 5) == 0:
            print("test progress:{}/{}".format(start, user_num))
        batch_df = eval_sessions.iloc[start:end]
        item_seq = batch_df['init_item_seq'].values.tolist()
        behavior_seq = batch_df['init_behavior_seq'].values.tolist()
        len_seq = batch_df['len_seq'].values.tolist()
        if len(item_seq) < batch:
            for _ in range(batch - len(item_seq)):
                item_seq.append(len(item_seq[0]) * [item_num])
                behavior_seq.append(len(item_seq[0]) * [bahavior_num])
                len_seq.append(1)
        click_seq = []
        buy_seq = []
        state_size = len(item_seq[0])
        for b in range(batch):
            clicks = []
            buys = []
            for k in range(state_size):
                if behavior_seq[b][k] == 0:
                    clicks.append(item_seq[b][k])
                elif behavior_seq[b][k] == 1:
                    buys.append(item_seq[b][k])
                else:
                    break
            clicks = clicks + [item_num for _ in range(state_size - len(clicks))]
            buys = buys + [item_num for _ in range(state_size - len(buys))]
            click_seq.append(clicks)
            buy_seq.append(buys)
        uid_seq = batch_df['uid'].values.tolist()
        feed_dict = {
            model.click_seq: click_seq,
            model.buy_seq: buy_seq,
        }
        if have_dropout == True: feed_dict[model.is_training] = False
        if have_user_emb == True: feed_dict[model.uid_seq] = uid_seq
        prediction = sess.run(model.output, feed_dict=feed_dict)
        prediction = prediction[:end - start]
        predict_list = np.argsort(prediction)
        true_items = batch_df['target'].values.tolist()
        calculate_hit_ndcg(predict_list, topk, true_items, hit_purchase, ndcg_purchase)
    for i in range(len(topk)):
        hit = hit_purchase[i] / user_num
        ndcg = ndcg_purchase[i] / user_num
        print("k | hit@k | ndcg@k")
        print("%d %.8f %.8f " % (topk[i], hit, ndcg))
    print('#############################################################')
    over_time = datetime.datetime.now()
    total_time = (over_time - start_time).total_seconds()
    print('total times: %s' % total_time)
    return hit_purchase[0] / user_num, ndcg_purchase[0] / user_num, \
           hit_purchase[1] / user_num, ndcg_purchase[1] / user_num, \
           hit_purchase[2] / user_num, ndcg_purchase[2] / user_num,


def evaluate_srgnn(sess, model, data_directory, topk, item_num, have_dropout, have_user_emb, is_test, type='all'):
    start_time = datetime.datetime.now()
    if is_test == False:
        eval_sessions = pd.read_pickle(os.path.join(data_directory, 'val.df'))
    else:
        if type == 'all':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test.df'))
        elif type == 'clicked':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test_click.df'))
        elif type == 'unclicked':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test_unclick.df'))

    batch = 128
    hit_purchase = [0, 0, 0]
    ndcg_purchase = [0, 0, 0]
    mrr_purchase = [0, 0, 0]
    user_num = eval_sessions.shape[0]
    for start in range(0, user_num, batch):
        end = start + batch if start + batch < user_num else user_num
        if start % (batch * 5) == 0:
            print("test progress:{}/{}".format(start, user_num))
        batch_df = eval_sessions.iloc[start:end]
        item_seq = batch_df['init_item_seq'].values.tolist()
        len_seq = batch_df['len_seq'].values.tolist()
        if len(item_seq) < batch:
            for _ in range(batch - len(item_seq)):
                item_seq.append(len(item_seq[0]) * [item_num])
                len_seq.append(1)
        A_in, A_out, alias_inputs, items = get_slice(item_seq, item_num)
        uid_seq = batch_df['uid'].values.tolist()
        feed_dict = {
            model.item_seq: item_seq,
            model.len_seq: len_seq,
            model.item: items,
            model.alias: alias_inputs,
            model.adj_in: A_in,
            model.adj_out: A_out,
        }
        if have_dropout == True: feed_dict[model.is_training] = False
        if have_user_emb == True: feed_dict[model.uid_seq] = uid_seq
        prediction = sess.run(model.output, feed_dict=feed_dict)
        prediction = prediction[:end - start]
        predict_list = np.argsort(prediction)
        true_items = batch_df['target'].values.tolist()
        calculate_hit_ndcg(predict_list, topk, true_items, hit_purchase, ndcg_purchase)
    for i in range(len(topk)):
        hit = hit_purchase[i] / user_num
        ndcg = ndcg_purchase[i] / user_num
        print("k | hit@k | ndcg@k")
        print("%d %.8f %.8f " % (topk[i], hit, ndcg))
    print('#############################################################')
    over_time = datetime.datetime.now()
    total_time = (over_time - start_time).total_seconds()
    print('total times: %s' % total_time)
    return hit_purchase[0] / user_num, ndcg_purchase[0] / user_num, \
           hit_purchase[1] / user_num, ndcg_purchase[1] / user_num, \
           hit_purchase[2] / user_num, ndcg_purchase[2] / user_num,


def get_slice_v2(inputs, padding):
    items, n_node, A_in, A_out, alias_inputs, alias_inputs_inv = [], [], [], [], [], []
    for u_input in inputs:

        n_node.append(len(np.unique(u_input)))

    max_n_node = np.max(n_node)
    for u_input in inputs:

        node = np.unique(u_input)


        cur_items = node.tolist() + (max_n_node - len(node)) * [padding]
        items.append(cur_items)

        u_A = np.zeros((max_n_node, max_n_node))
        for i in np.arange(len(u_input) - 1):
            if u_input[i + 1] == padding:
                break

            u = np.where(node == u_input[i])[0][0]
            v = np.where(node == u_input[i + 1])[0][0]
            u_A[u][v] += 1
        u_sum_in = np.sum(u_A, 0)
        u_sum_in[np.where(u_sum_in == 0)] = 1
        u_A_in = np.divide(u_A, u_sum_in)
        u_sum_out = np.sum(u_A, 1)
        u_sum_out[np.where(u_sum_out == 0)] = 1
        u_A_out = np.divide(u_A.transpose(), u_sum_out)

        A_in.append(u_A_in)

        A_out.append(u_A_out)
        alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        origin_input = np.array(u_input)


        alias_inputs_inv.append([np.where(origin_input == i)[0][0] if i in origin_input else len(u_input) - 1 for i in cur_items])

    return np.array(A_in), np.array(A_out), np.array(alias_inputs), np.array(alias_inputs_inv), np.array(items)


def evaluate_srgnn_bar(sess, model, data_directory, topk, item_num, have_dropout, have_user_emb, is_test, type='all'):
    start_time = datetime.datetime.now()
    if is_test == False:
        eval_sessions = pd.read_pickle(os.path.join(data_directory, 'val.df'))
    else:
        if type == 'all':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test.df'))
        elif type == 'clicked':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test_click.df'))
        elif type == 'unclicked':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test_unclick.df'))

    batch = 128
    behavior_num = 2
    hit_purchase = [0, 0, 0]
    ndcg_purchase = [0, 0, 0]
    mrr_purchase = [0, 0, 0]
    user_num = eval_sessions.shape[0]
    for start in range(0, user_num, batch):
        end = start + batch if start + batch < user_num else user_num
        if start % (batch * 5) == 0:
            print("test progress:{}/{}".format(start, user_num))
        batch_df = eval_sessions.iloc[start:end]
        item_seq = batch_df['init_item_seq'].values.tolist()
        len_seq = batch_df['len_seq'].values.tolist()
        behavior_seq = batch_df['init_behavior_seq'].values.tolist()
        uid_seq = batch_df['uid'].values.tolist()
        if len(item_seq) < batch:
            for _ in range(batch - len(item_seq)):
                item_seq.append(len(item_seq[0]) * [item_num])
                behavior_seq.append(len(item_seq[0]) * [behavior_num])
                len_seq.append(1)
        position_info = np.zeros((len(len_seq), len(item_seq[0])))
        for idx, len_ in enumerate(len_seq):
            position_info[idx][:len_] = range(len_, 0, -1)
        next_behavior = np.ones_like(len_seq)

        A_in, A_out, alias_inputs, alias_inputs_inv, items = get_slice_v2(item_seq, item_num)
        feed_dict = {
            model.item_seq: item_seq,
            model.len_seq: len_seq,
            model.item: items,
            model.alias: alias_inputs,
            model.alias_inv: alias_inputs_inv,
            model.adj_in: A_in,
            model.adj_out: A_out,
            model.behavior_seq: behavior_seq,
            model.next_behaviors_input: next_behavior,
            model.position: position_info
        }
        if have_dropout == True: feed_dict[model.is_training] = False
        if have_user_emb == True: feed_dict[model.uid_seq] = uid_seq
        prediction = sess.run(model.output, feed_dict=feed_dict)
        prediction = prediction[:end - start]
        predict_list = np.argsort(prediction)
        true_items = batch_df['target'].values.tolist()
        calculate_hit_ndcg(predict_list, topk, true_items, hit_purchase, ndcg_purchase)
    for i in range(len(topk)):
        hit = hit_purchase[i] / user_num
        ndcg = ndcg_purchase[i] / user_num
        print("k | hit@k | ndcg@k")
        print("%d %.8f %.8f " % (topk[i], hit, ndcg))
    print('#############################################################')
    over_time = datetime.datetime.now()
    total_time = (over_time - start_time).total_seconds()
    print('total times: %s' % total_time)
    return hit_purchase[0] / user_num, ndcg_purchase[0] / user_num, \
           hit_purchase[1] / user_num, ndcg_purchase[1] / user_num, \
           hit_purchase[2] / user_num, ndcg_purchase[2] / user_num,


def evaluate_trans(sess, model, data_directory, topk, have_dropout, have_user_emb, is_test, type='all'):
    start_time = datetime.datetime.now()
    if is_test == False:
        eval_sessions = pd.read_pickle(os.path.join(data_directory, 'val.df'))
    else:
        if type == 'all':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test.df'))
        elif type == 'clicked':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test_click.df'))
        elif type == 'unclicked':
            eval_sessions = pd.read_pickle(os.path.join(data_directory, 'test_unclick.df'))

    batch = 100
    hit_purchase = [0, 0, 0]
    ndcg_purchase = [0, 0, 0]
    mrr_purchase = [0, 0, 0]
    user_num = eval_sessions.shape[0]
    trans_dict = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3, (2, 0): 4, (2, 1): 4, (2, 2): 4}
    for start in range(0, user_num, batch):
        end = start + batch if start + batch < user_num else user_num
        if start % 5000 == 0:
            print("test progress:{}/{}".format(start, user_num))
        batch_df = eval_sessions.iloc[start:end]
        item_seq = batch_df['init_item_seq'].values.tolist()
        behavior_seq = batch_df['init_behavior_seq'].values.tolist()
        len_seq = batch_df['len_seq'].values.tolist()
        uid_seq = batch_df['uid'].values.tolist()
        next_behavior = np.ones_like(len_seq)
        trans_seq = []
        for k in range(len(behavior_seq)):
            trans_seq.append([4 for _ in range(len(behavior_seq[0]))])
            for kk in range(len(behavior_seq[0])):
                if behavior_seq[k][kk] == 2:
                    continue
                if kk == len_seq[k] - 1:
                    trans_seq[k][kk] = trans_dict[(behavior_seq[k][kk], next_behavior[k])]
                elif kk == len(behavior_seq[0]) - 1:
                    continue
                else:
                    trans_seq[k][kk] = trans_dict[(behavior_seq[k][kk], behavior_seq[k][kk + 1])]
        feed_dict = {
            model.item_seq: item_seq,
            model.len_seq: len_seq,
            model.behavior_seq: behavior_seq,
            model.trans_seq: trans_seq,
        }
        if have_dropout == True: feed_dict[model.is_training] = False
        if have_user_emb == True: feed_dict[model.uid_seq] = uid_seq
        prediction = sess.run(model.output, feed_dict=feed_dict)
        predict_list = np.argsort(prediction)
        true_items = batch_df['target'].values.tolist()
        calculate_hit_ndcg(predict_list, topk, true_items, hit_purchase, ndcg_purchase)
    for i in range(len(topk)):
        hit = hit_purchase[i] / user_num
        ndcg = ndcg_purchase[i] / user_num
        print("k | hit@k | ndcg@k")
        print("%d %.8f %.8f " % (topk[i], hit, ndcg))
    print('#############################################################')
    over_time = datetime.datetime.now()
    total_time = (over_time - start_time).total_seconds()
    print('total times: %s' % total_time)
    return hit_purchase[0] / user_num, ndcg_purchase[0] / user_num, \
           hit_purchase[1] / user_num, ndcg_purchase[1] / user_num, \
           hit_purchase[2] / user_num, ndcg_purchase[2] / user_num,


def evaluate_GBGSR(sess,model,data_directory,topk,have_dropout,is_test,type='all'):
    start_time = datetime.datetime.now()
    if is_test == False:
        eval_sessions=pd.read_pickle(os.path.join(data_directory, 'val.df'))
    else:
        if type == 'all':
            eval_sessions=pd.read_pickle(os.path.join(data_directory, 'test.df'))
        elif type == 'clicked':
            eval_sessions=pd.read_pickle(os.path.join(data_directory, 'test_click.df'))
        elif type =='unclicked':
            eval_sessions=pd.read_pickle(os.path.join(data_directory, 'test_unclick.df'))

    batch = 100
    hit_purchase=[0,0,0]
    ndcg_purchase=[0,0,0]
    mrr_purchase = [0, 0, 0]
    user_num = eval_sessions.shape[0]

    data_statis = pd.read_pickle(
        os.path.join(data_directory, 'data_statis.df'))
    state_size = data_statis['state_size'][0]
    item_num = data_statis['item_num'][0]
    for start in range(0,user_num,batch):
        end = start + batch if start + batch < user_num else user_num
        if start % 5000 == 0:
            print("test progress:{}/{}".format(start,user_num))
        batch_df = eval_sessions.iloc[start:end]
        item_seq = batch_df['init_item_seq'].values.tolist()
        behavior_seq = batch_df['init_behavior_seq'].values.tolist()
        len_seq = batch_df['len_seq'].values.tolist()
        target_behavior = np.ones_like(len_seq)
        uid_seq = batch_df['uid'].values.tolist()

        local_graph_in = []
        local_graph_out = []
        for i in range(end-start):
            local_graph_in.append([])
            local_graph_out.append([])
            pre_id = item_num
            for curr_state_id in range(state_size):
                local_graph_in[i].append([])
                local_graph_out[i].append([])
                curr_id = item_seq[i][curr_state_id]
                if pre_id != item_num:
                    local_graph_in[i][curr_state_id].append(pre_id)
                    local_graph_out[i][curr_state_id-1].append(curr_id)
                pre_id = curr_id

        for i in range(end-start):
            for j in range(state_size):
                local_graph_in[i][j] = pad_history(local_graph_in[i][j], 1, item_num)
                local_graph_out[i][j] = pad_history(local_graph_out[i][j], 1, item_num)

        feed_dict = {model.item_seq: item_seq,
                     model.len_seq:len_seq,
                     model.behavior_seq : behavior_seq,
                     model.target_behavior: target_behavior,
                     model.local_graph_in: local_graph_in,
                     model.local_graph_out: local_graph_out}
        if have_dropout == True: feed_dict[model.is_training] = False
        prediction = sess.run(model.output, feed_dict=feed_dict)
        predict_list = np.argsort(prediction)
        true_items = batch_df['target'].values.tolist()
        calculate_hit_ndcg(predict_list, topk, true_items, hit_purchase, ndcg_purchase)
    for i in range(len(topk)):
        hit=hit_purchase[i]/ user_num
        ndcg=ndcg_purchase[i]/user_num
        print("k | hit@k | ndcg@k")
        print("%d %.8f %.8f " % (topk[i], hit, ndcg))
    print('#############################################################')
    over_time = datetime.datetime.now()
    total_time = (over_time - start_time).total_seconds()
    print('total times: %s' % total_time)
    return hit_purchase[0] / user_num, ndcg_purchase[0] / user_num, \
           hit_purchase[1] / user_num, ndcg_purchase[1] / user_num, \
           hit_purchase[2] / user_num, ndcg_purchase[2] / user_num,
