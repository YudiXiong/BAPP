import pandas as pd
import numpy as np
import random
import math
import itertools
import copy


def position_sampling_probabilities(length, a=0.8):
    item_indices = np.arange(length)
    item_importance = np.power(a, length - item_indices)

    total = np.sum(item_importance)
    prob = item_importance / total
    return prob


def intra_subsequence_deletion(seq, behavior, gamma):
    seq_ = seq.copy()
    behavior_ = behavior.copy()
    num_sub_seq = len(seq_)
    index = np.arange(num_sub_seq)
    sub_prob = position_sampling_probabilities(num_sub_seq)[::-1]
    num_samples = math.ceil(num_sub_seq * gamma)

    num_samples = random.sample(range(num_samples + 1), k=1)[0]
    if num_samples == 0:
        perturbed_item_seq = np.concatenate(seq_)
        perturbed_behavior_seq = np.concatenate(behavior_)
        return perturbed_item_seq, perturbed_behavior_seq
    else:


        sampled_index = np.random.choice(index, p=sub_prob, size=num_samples, replace=False)
        for i in range(num_samples):
            index = sampled_index[i]
            sub_seq = seq_[index]
            num_sampled_sub_seq = len(sub_seq)

            sub_seq_behavior = behavior_[index]
            num_sampled_sub_seq_purchase = np.count_nonzero(sub_seq_behavior)
            num_sampled_sub_seq_click = num_sampled_sub_seq - num_sampled_sub_seq_purchase

            if num_sampled_sub_seq_click == 0 or num_sampled_sub_seq == 1:
                pass
            else:
                del_index = random.sample(range(num_sampled_sub_seq_click), k=1)[0]
                deled_sub_seq = np.delete(sub_seq, del_index)
                deled_sub_behavior = np.delete(sub_seq_behavior, del_index)
                seq_[index] = deled_sub_seq
                behavior_[index] = deled_sub_behavior


        perturbed_item_seq = np.concatenate(seq_)
        perturbed_behavior_seq = np.concatenate(behavior_)


        return perturbed_item_seq, perturbed_behavior_seq


def intra_subsequence_reordering(seq, behavior, beta):
    seq_ = seq.copy()
    behavior_ = behavior.copy()
    num_sub_seq = len(seq_)
    index = np.arange(num_sub_seq)
    sub_prob = position_sampling_probabilities(num_sub_seq)[::-1]
    num_samples = math.ceil(num_sub_seq * beta)
    num_samples = random.sample(range(num_samples + 1), k=1)[0]
    if num_samples == 0:
        reorder_item_seq = np.concatenate(seq_)
        reorder_behavior_seq = np.concatenate(behavior_)
        return reorder_item_seq, reorder_behavior_seq
    else:
        sampled_index = np.random.choice(index, p=sub_prob, size=num_samples, replace=False)
        for i in range(num_samples):
            index = sampled_index[i]
            sub_seq = seq_[index]
            sub_behavior = behavior_[index]
            num_sampled_sub_seq = len(sub_seq)
            num_sampled_sub_seq_purchase = np.count_nonzero(sub_behavior)
            num_sampled_sub_seq_click = num_sampled_sub_seq - num_sampled_sub_seq_purchase
            item_index = np.arange(num_sampled_sub_seq_click)
            if num_sampled_sub_seq_click == 0:
                continue
            item_prob = position_sampling_probabilities(num_sampled_sub_seq_click)[::-1]
            if len(item_prob) == 1:
                continue
            indices = np.random.choice(item_index, p=item_prob, size=num_sampled_sub_seq_click, replace=False)
            if len(indices) == 1:
                continue
            shuffled_sub_seq = np.array(sub_seq)[indices].tolist() + sub_seq[num_sampled_sub_seq_click:]
            shuffled_sub_behavior = np.array(sub_behavior)[indices].tolist() + sub_behavior[num_sampled_sub_seq_click:]

            seq_[index] = shuffled_sub_seq
            behavior_[index] = shuffled_sub_behavior

        reorder_item_seq = np.concatenate(seq_)
        reorder_behavior_seq = np.concatenate(behavior_)

        return reorder_item_seq, reorder_behavior_seq


def inter_subsequence_distortion(seq, behavior, alpha):
    seq_ = seq.copy()
    behavior_ = behavior.copy()
    num_sub_seq = len(seq_)
    index = np.arange(num_sub_seq)

    selected_item1_index = random.sample(range(num_sub_seq), k=1)[0]
    item_importance = np.power(alpha, abs(index - selected_item1_index))
    total = np.sum(item_importance)
    prob = item_importance / total
    selected_item2_index = np.random.choice(index, p=prob, size=1, replace=False)[0]

    seq_[selected_item1_index], seq_[selected_item2_index] = seq_[selected_item2_index], seq_[selected_item1_index]
    behavior_[selected_item1_index], behavior_[selected_item2_index] = behavior_[selected_item2_index], behavior_[
        selected_item1_index]

    reorder_sub_seq = np.concatenate(seq_)
    reorder_sub_behavior = np.concatenate(behavior_)

    return reorder_sub_seq, reorder_sub_behavior


def perturb_behavior_one_items(item_sequences, behavior_sequences, item_num, epsilon=0.1):
    new_item_sequences = []
    new_behavior_sequences = []

    for seq_items, seq_behavs in zip(item_sequences, behavior_sequences):
        seq_items = np.array(seq_items)
        seq_behavs = np.array(seq_behavs)


        behavior_one_positions = np.where(seq_behavs == 1)[0]


        items_range = np.arange(item_num)

        for pos in behavior_one_positions:

            current_item = seq_items[pos]


            prob = np.ones(item_num, dtype=np.float32)
            prob[current_item] = np.exp(epsilon)
            prob /= prob.sum()


            new_item = np.random.choice(items_range, p=prob)

            seq_items[pos] = new_item

        new_item_sequences.append(seq_items.tolist())
        new_behavior_sequences.append(seq_behavs.tolist())

    return new_item_sequences, new_behavior_sequences


def report_perturbation_change_ratio(original_seq, original_behavior, perturbed_seq, perturbed_behavior):

    original_seq_flattened = np.array(list(itertools.chain.from_iterable(original_seq)))
    original_behavior_flattened = np.array(list(itertools.chain.from_iterable(original_behavior)))


    if len(original_seq_flattened) != len(perturbed_seq):
        print("警告：序列长度不一致，无法直接比较。")
        return None, None, None


    changes = original_seq_flattened != perturbed_seq
    changed_count = np.sum(changes)
    total_items = len(original_seq_flattened)


    if total_items > 0:
        change_ratio = (changed_count / total_items) * 100
    else:
        change_ratio = 0


    changes_2 = original_behavior_flattened != perturbed_behavior
    changed_count_2 = np.sum(changes_2)
    total_items_2 = len(original_behavior_flattened)


    if total_items_2 > 0:
        change_ratio_2 = (changed_count_2 / total_items_2) * 100
    else:
        change_ratio_2 = 0

    print(f"总商品数量: {total_items}")
    print(f"被改变的商品数量: {changed_count}")
    print(f"改变比例: {change_ratio:.2f}%")

    print(f"总商品数量_2: {total_items_2}")
    print(f"被改变的商品数量_2: {changed_count_2}")
    print(f"改变比例_2: {change_ratio_2:.2f}%")

    return change_ratio, changed_count, total_items


def intra_subsequence_distortion(seq, behavior, epsilon, item_num):


    seq_ = copy.deepcopy(seq)
    behavior_ = copy.deepcopy(behavior)
    flattened_seq_ = list(itertools.chain.from_iterable(seq_))
    flattened_behavior_ = list(itertools.chain.from_iterable(behavior_))


    prob = np.zeros(item_num + 1, dtype=np.float32)
    prob[item_num] = 1
    prob[flattened_seq_] = 1

    items = np.arange(item_num + 1, dtype=np.int32)


    for i in range(len(seq_)):
        sublist = seq_[i]
        for j in range(len(sublist)):
            if behavior_[i][j] != 1:
                prob[seq_[i][j]] = math.exp(epsilon)
                perturbed_item = int(
                    np.random.choice(items, p=prob / prob.sum()))


                prob[seq_[i][j]] = 1
                if perturbed_item != item_num:
                    prob[perturbed_item] = 0
                if perturbed_item != seq_[i][
                    j]:
                    ind_tuple = np.nonzero(seq_[i][j + 1:] == perturbed_item)
                    ind_list = ind_tuple[0]
                    if perturbed_item != item_num and len(ind_list) != 0:
                        ind = ind_list[0] + j + 1
                        seq_[i][j], seq_[i][ind] = seq_[i][ind], seq_[i][j]
                    else:
                        seq_[i][j] = perturbed_item


    perturbed_item_seq = np.concatenate(seq_)
    perturbed_behavior_seq = np.concatenate(behavior_)


    return perturbed_item_seq, perturbed_behavior_seq


def perturb_sequences(items, behaviors, lengths, item_num, max_seq_len, tag, alpha, beta, gamma, lamda, epsilon):
    batch_size = len(items)
    perturbed_items = []
    perturbed_behaviors = []
    perturbed_lengths = []
    for i in range(batch_size):
        item_seq = items[i]
        behavior_seq = behaviors[i]
        length = lengths[i]


        unpad_item_seq_temp = np.array(item_seq)[:length]
        unpad_behavior_seq_temp = np.array(behavior_seq)[:length]


        unpad_item_seq = np.pad(unpad_item_seq_temp, (0, max_seq_len - len(unpad_item_seq_temp)), 'constant',
                                constant_values=item_num)
        unpad_behavior_seq = np.pad(unpad_behavior_seq_temp, (0, max_seq_len - len(unpad_behavior_seq_temp)),
                                    'constant', constant_values=2)


        mask = (unpad_behavior_seq[:-1] == 1) & (
                    unpad_behavior_seq[1:] == 0)
        split_indices = np.where(mask)[0] + 1
        split_indices = np.insert(split_indices, 0, 0)
        split_indices = np.append(split_indices, length)


        item_sequences = [unpad_item_seq[start:end].tolist() for start, end in
                          zip(split_indices[:-1], split_indices[1:])]

        behavior_sequences = [unpad_behavior_seq[start:end].tolist() for start, end in
                              zip(split_indices[:-1], split_indices[1:])]


        if tag == 1:
            perturbed_seq, perturbed_behavior = intra_subsequence_distortion(item_sequences, behavior_sequences, epsilon, item_num)
        elif tag == 2:

            item_sequences = np.concatenate(item_sequences)
            behavior_sequences = np.concatenate(behavior_sequences)
            new_subseqs = split_behavior_subsequences(item_sequences, behavior_sequences)
            perturbed_seq, perturbed_behavior = inter_subsequence_distortion(new_subseqs[0], new_subseqs[1], alpha)
        elif tag == 3:
            perturbed_seq, perturbed_behavior = intra_subsequence_distortion(item_sequences, behavior_sequences, epsilon, item_num)

            new_subseqs = split_behavior_subsequences(perturbed_seq, perturbed_behavior)
            perturbed_seq, perturbed_behavior = inter_subsequence_distortion(new_subseqs[0], new_subseqs[1], alpha)
        elif tag == 0:
            perturbed_seq, perturbed_behavior = unpad_item_seq, unpad_behavior_seq


        item_seq = np.pad(perturbed_seq, (0, max_seq_len - len(perturbed_seq)), 'constant', constant_values=item_num)
        behavior_seq = np.pad(perturbed_behavior, (0, max_seq_len - len(perturbed_behavior)), 'constant', constant_values=2)

        item_seq = item_seq.tolist()
        behavior_seq = behavior_seq.tolist()

        perturbed_items.append(item_seq)
        perturbed_behaviors.append(behavior_seq)
        perturbed_lengths.append(len(perturbed_seq))

    return perturbed_items, perturbed_behaviors, perturbed_lengths


def split_behavior_subsequences(seq, behavior):
    mask = (behavior[:-1] == 1) & (behavior[1:] == 0)
    split_indices = np.where(mask)[0] + 1
    split_indices = np.insert(split_indices, 0, 0)
    split_indices = np.append(split_indices, len(seq))
    item_subseqs = [seq[start:end].tolist() for start, end in zip(split_indices[:-1], split_indices[1:])]
    behavior_subseqs = [behavior[start:end].tolist() for start, end in zip(split_indices[:-1], split_indices[1:])]
    return item_subseqs, behavior_subseqs
