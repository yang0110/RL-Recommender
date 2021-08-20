from collections import defaultdict
import os
import random
import time
import tqdm
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch.utils.data as td
import torch


class EvalDataset(td.Dataset):
    def __init__(self, positive_data, item_num, positive_mat, negative_samples=99):
        super(EvalDataset, self).__init__()
        self.positive_data = np.array(positive_data)
        self.item_num = item_num
        self.positive_mat = positive_mat
        self.negative_samples = negative_samples
        
        self.reset()
        
    def reset(self):
        print("Resetting dataset")
        data = self.create_valid_data()
        labels = np.zeros(len(self.positive_data) * (1 + self.negative_samples))
        labels[::1+self.negative_samples] = 1
        self.data = np.concatenate([
            np.array(data), 
            np.array(labels)[:, np.newaxis]], 
            axis=1
        )

    def create_valid_data(self):
        valid_data = []
        for user, positive in self.positive_data:
            valid_data.append([user, positive])
            for i in range(self.negative_samples):
                negative = np.random.randint(self.item_num)
                while (user, negative) in self.positive_mat:
                    negative = np.random.randint(self.item_num)
                    
                valid_data.append([user, negative])
        return valid_data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user, item, label = self.data[idx]
        output = {
            "user": user,
            "item": item,
            "label": np.float32(label),
        }
        return output


#https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_dim, mu=0.0, theta=0.15, max_sigma=0.4, min_sigma=0.4, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_dim
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return torch.tensor([action + ou_state]).float()


class Prioritized_Buffer(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity   = capacity
        self.buffer     = []
        self.pos        = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    
    def push(self, user, memory, action, reward, next_user, next_memory, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((user, memory, action, reward, next_user, next_memory, done))
        else:
            self.buffer[self.pos] = (user, memory, action, reward, next_user, next_memory, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        # if len(self.buffer) == self.capacity:
        #     prios = self.priorities
        # else:
        #     prios = self.priorities[:self.pos]
        
        # probs  = prios ** self.prob_alpha
        # probs /= probs.sum()
        
        # indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        indices = np.random.choice(len(self.buffer), batch_size)
        samples = [self.buffer[idx] for idx in indices]

        # total    = len(self.buffer)
        # weights  = (total * probs[indices]) ** (-beta)
        # weights /= weights.max()
        # weights  = np.array(weights, dtype=np.float32)

        batch       = list(zip(*samples))
        user        = np.concatenate(batch[0])
        memory      = np.concatenate(batch[1])
        action      = batch[2]
        reward      = batch[3]
        next_user   = np.concatenate(batch[4])
        next_memory = np.concatenate(batch[5])
        done        = batch[6]

        return user, memory, action, reward, next_user, next_memory, done

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)


def get_beta(idx, beta_start=0.4, beta_steps=100000):
    return min(1.0, beta_start + idx * (1.0 - beta_start) / beta_steps)

def preprocess_data(data_dir):
    data = pd.read_csv(data_dir)
    data = data[['uid', 'pic_id', 'label', 'day']]
    train_data = data[data.day!=14][['uid', 'pic_id', 'label']]
    test_data = data[data.day==14][['uid', 'pic_id', 'label']]

    test_pic_pool = {}
    test_related_pic = {}
    for u in test_data.uid.unique():
        test_pic_pool[u] = None
        test_related_pic[u] = None
        user_data = test_data[test_data.uid==u]
        pics = user_data.pic_id.unique().tolist()
        test_pic_pool[u] = pics
        related_pics = user_data[user_data.label==1].pic_id.unique().tolist()
        test_related_pic[u] = related_pics


    data = data[data['label']==1][['uid', 'pic_id', 'day']]
    user_num = data['uid'].max() + 1
    item_num = data['pic_id'].max() + 1
    train_data = data[data.day!=14][['uid', 'pic_id']]
    test_data = data[data.day==14][['uid', 'pic_id']]

    train_data = train_data[['uid', 'pic_id']]
    test_data = test_data[['uid', 'pic_id']]

    train_data = train_data.values.tolist()
    train_mat = defaultdict(int)
    test_mat = defaultdict(int)
    for user, item in train_data:
        train_mat[user, item] = 1.0

    test_data = test_data.values.tolist()
    for user, item in test_data:
        test_mat[user, item] = 1.0

    train_matrix = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    dict.update(train_matrix, train_mat)
    test_matrix = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    dict.update(test_matrix, test_mat)
    appropriate_users = np.where(train_matrix.sum(1)>=1)[0]

    return (train_data, train_matrix, test_data, test_matrix, 
            user_num, item_num, appropriate_users, test_pic_pool, test_related_pic)


def extract_user_history(concat_data, test_day=14):
    test = concat_data[concat_data.day==test_day]
    train = concat_data[concat_data.day!=test_day]

    #  create train user history
    users = train.uid.unique()
    days = train.day.unique()
    train_user_history = {}
    train_pic_daily_pool = {}

    for u in users:
        user_data = train[train.uid==u]
        train_user_history[u] = {}
        train_pic_daily_pool[u] = {}
        print('user', u)
        for day in days:
            day_data = user_data[user_data.day==day]
            all_pics = day_data.pic_id.values
            train_pic_daily_pool[u][day] = list(all_pics)
            pos_pics = day_data[day_data.label==1].pic_id.values
            train_user_history[u][day] = list(pos_pics) 


    #  create test user history
    users = test.uid.unique()
    days = test.day.unique()
    test_user_history = {}
    test_pic_daily_pool = {}

    for u in users:
        user_data = test[test.uid==u]
        test_user_history[u] = {}
        test_pic_daily_pool[u] = {}
        print('user', u)
        for day in days:
            day_data = user_data[user_data.day==day]
            all_pics = day_data.pic_id.values
            test_pic_daily_pool[u][day] = list(all_pics)
            pos_pics = day_data[day_data.label==1].pic_id.values
            test_user_history[u][day] = list(pos_pics) 
    return train_user_history, train_pic_daily_pool, test_user_history, test_pic_daily_pool


def extract_user_history_version_2(concat_data, test_day=14):
    test = concat_data[concat_data.day==test_day]
    train = concat_data[concat_data.day!=test_day]
    train_user_history = {}
    train_pic_pool = {}
    test_user_history = {}
    test_pic_pool = {}

    users = train.uid.unique()
    for u in users:
        user_data = train[train.uid==u]
        pos_pics = user_data[user_data.label==1].pic_id.unique()
        pool = user_data.pic_id.unique()
        train_user_history[u] = pos_pics.tolist()
        train_pic_pool[u] = pool.tolist()

    users = test.uid.unique()
    for u in users:
        user_data = test[test.uid==u]
        pos_pics = user_data[user_data.label==1].pic_id.unique()
        pool = user_data.pic_id.unique()
        test_user_history[u] = pos_pics.tolist()
        test_pic_pool[u] = pool.tolist() 

    return train_user_history, train_pic_pool, test_user_history, test_pic_pool

def to_np(tensor):
    return tensor.detach().cpu().numpy()

def hit_metric(recommended, actual):
    return int(actual in recommended)

def dcg_metric(recommended, actual):
    if actual in recommended:
        index = recommended.index(actual)
        return np.reciprocal(np.log2(index + 2))
    return 0

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def avg_recall_k(actual: list, predicted: list, k=10) -> int:
    """
    Computes the average recall at k.
    Parameters
    ----------
    actual : list
        A list of actual items to be predicted
    predicted : list
        An ordered list of predicted items
    k : int, default = 10
        Number of predictions to consider
    Returns:
    -------
    score : int
        The average recall at k.
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / len(actual)





def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


def r_precision(r):
    """Score is precision after all relevant documents have been retrieved
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> r_precision(r)
    0.33333333333333331
    >>> r = [0, 1, 0]
    >>> r_precision(r)
    0.5
    >>> r = [1, 0, 0]
    >>> r_precision(r)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        R Precision
    """
    r = np.asarray(r) != 0
    z = r.nonzero()[0]
    if not z.size:
        return 0.
    else:
        return np.mean(r[:z[-1] + 1])


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    else:
        return np.mean(out)
        
def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])

def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    else:
        raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    else:
        return dcg_at_k(r, k, method) / dcg_max



