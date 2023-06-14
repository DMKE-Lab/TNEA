import numpy as np
import pickle as pkl
import networkx as nx
import tensorflow_hub as hub
import os
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import tensorflow.compat.v1 as tf
from transformers import BertModel, BertTokenizer
import sys
sys.path.append('./bert-master')  # 替换为你克隆BERT库的路径
import bert
import tokenization
from bert.modeling import BertModel, BertConfig
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights

# 初始化tokenizer
# tokenizer = FullTokenizer(vocab_file=os.path.join("./multi_cased_L-12_H-768_A-12/", "vocab.txt"))

tf.disable_v2_behavior()
import math

flags = tf.app.flags
FLAGS = flags.FLAGS


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, placeholders):
    """Construct feed dictionary for GCN-Align."""
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def loadfile(fn, num=1):
    """Load a file and return a list of tuple containing $num integers in each line."""
    print('loading a file...' + fn)
    ret = []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            x = []
            for i in range(num):
                x.append(int(th[i]))
            ret.append(tuple(x))
    return ret


def get_ent2id(fns):
    ent2id = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                ent2id[th[1]] = int(th[0])
    return ent2id


def loadattr(fns, e, ent2id):
    """
        The most frequent attributes are selected to save space.
        This version also returns BERT embeddings for each attribute.
    """
    cnt = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                if th[0] not in ent2id:
                    continue
                for i in range(1, len(th)):   # 这里就是统计zh-关系的个数
                    if th[i] not in cnt:
                        cnt[th[i]] = 1
                    else:
                        cnt[th[i]] += 1
    # 将上面得到的结果封装起来   <class 'tuple'>: ('http://dbpedia.org/property/name', 20111)  并按照个数进行降序排列
    fre = [(k, cnt[k]) for k in sorted(cnt, key=cnt.get, reverse=True)]
    num_features = min(len(fre), 2000)
    attr2id = {}
    for i in range(num_features):   # 这里将属性和id进行封装   'http://dbpedia.org/property/name' (140305051052048) = 0
        attr2id[fre[i][0]] = i
    M = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                if th[0] in ent2id:
                    for i in range(1, len(th)):
                        if th[i] in attr2id:
                            M[(ent2id[th[0]], attr2id[th[i]])] = 1.0
    row = []
    col = []
    data = []
    for key in M:
        row.append(key[0])
        col.append(key[1])
        data.append(M[key])

    original_coo_matrix = sp.coo_matrix((data, (row, col)), shape=(e, num_features))


    # Using BERT for attribute embeddings
    bert_config = BertConfig.from_json_file('./cased_L-12_H-768_A-12/bert_config.json')
    tokenizer = tokenization.FullTokenizer(vocab_file='./cased_L-12_H-768_A-12/vocab.txt')
    input_ids = tf.placeholder(shape=[None, None], dtype=tf.int32, name="input_ids")
    input_mask = tf.placeholder(shape=[None, None], dtype=tf.int32, name="input_mask")
    segment_ids = tf.placeholder(shape=[None, None], dtype=tf.int32, name="segment_ids")

    bert_model = BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids
    )

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        attr_embeddings = {}
        for attr in attr2id.keys():
            tokens = tokenizer.tokenize(attr)
            input_ids_attr = tokenizer.convert_tokens_to_ids(tokens)
            input_ids_attr = [tokenizer.vocab["[CLS]"]] + input_ids_attr + [tokenizer.vocab["[SEP]"]]
            input_mask_attr = [1] * len(input_ids_attr)
            segment_ids_attr = [0] * len(input_ids_attr)

            outputs = sess.run(
                bert_model.sequence_output,
                feed_dict={
                    input_ids: [input_ids_attr],
                    input_mask: [input_mask_attr],
                    segment_ids: [segment_ids_attr]
                }
            )

            attr_embeddings[attr] = np.mean(outputs, axis=1)

    return original_coo_matrix, attr_embeddings

    # # 原始的稀疏矩阵
    # original_coo_matrix = sp.coo_matrix((data, (row, col)), shape=(e, num_features))
    #
    # # 使用BERT进行属性嵌入
    # bert = BertModel.from_pretrained('bert-base-multilingual-cased')
    # tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    # attr_embeddings = {}
    # for attr in attr2id.keys():
    #     inputs = tokenizer(attr, return_tensors='pt')
    #     outputs = bert(**inputs)
    #     # 这里我们取BERT的最后一层的输出的平均值作为属性的嵌入向量
    #     attr_embeddings[attr] = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    #
    # return original_coo_matrix, attr_embeddings
    # # return sp.coo_matrix((data, (row, col)), shape=(e, num_features)) # attr


def get_dic_list(e, KG):
    M = {}
    for tri in KG:
        if tri[0] == tri[2]:
            continue
        M[(tri[0], tri[2])] = 1
        M[(tri[2], tri[0])] = 1
    dic_list = {}
    for i in range(e):
        dic_list[i] = []
    for pair in M:
        dic_list[pair[0]].append(pair[1])
    return dic_list


def func(KG):
    head = {}
    cnt = {}
    for tri in KG:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            head[tri[1]] = set([tri[0]])
        else:
            cnt[tri[1]] += 1
            head[tri[1]].add(tri[0])
    r2f = {}
    for r in cnt:
        r2f[r] = len(head[r]) / cnt[r]
    return r2f


def ifunc(KG):
    tail = {}
    cnt = {}
    for tri in KG:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            tail[tri[1]] = set([tri[2]])
        else:
            cnt[tri[1]] += 1
            tail[tri[1]].add(tri[2])
    r2if = {}
    for r in cnt:
        r2if[r] = len(tail[r]) / cnt[r]
    return r2if


def get_weighted_adj(e, KG):
    r2f = func(KG)
    r2if = ifunc(KG)
    M = {}
    for tri in KG:
        if tri[0] == tri[2]:
            continue
        if (tri[0], tri[2]) not in M:
            M[(tri[0], tri[2])] = max(r2if[tri[1]], 0.3)
        else:
            M[(tri[0], tri[2])] += max(r2if[tri[1]], 0.3)
        if (tri[2], tri[0]) not in M:
            M[(tri[2], tri[0])] = max(r2f[tri[1]], 0.3)
        else:
            M[(tri[2], tri[0])] += max(r2f[tri[1]], 0.3)
    row = []
    col = []
    data = []
    for key in M:
        row.append(key[1])
        col.append(key[0])
        data.append(M[key])
    return sp.coo_matrix((data, (row, col)), shape=(e, e))

def sparse_tensor_value_to_coo(sparse_tensor_value):
    indices = np.vstack((sparse_tensor_value.indices[:, 0], sparse_tensor_value.indices[:, 1])).T
    return coo_matrix((sparse_tensor_value.values, indices.T), shape=sparse_tensor_value.dense_shape)

def get_ae_input(attr):
    print(type(attr))
    # print(attr)
    return sparse_to_tuple(sp.coo_matrix(attr))
    # return sparse_to_tuple(sparse_tensor_value_to_coo(attr))

def load_data(dataset_str):
    data_path = 'data/'+dataset_str+'/'
    names = [['ent_ids_1', 'ent_ids_2'], ['training_attrs_1', 'training_attrs_2'], ['triples_1', 'triples_2'], ['ref_ent_ids']]
    # 拼接字符串，将dataset_str与names中的各个元素拼接在一起形成完整的数据集的路径
    for fns in names:
        for i in range(len(fns)):
            fns[i] = data_path + fns[i] # 拼接字符串
    Es, As, Ts, ill = names

    KG1 = loadfile(Ts[0], 3)
    KG2 = loadfile(Ts[1], 3)
    KG = KG1 + KG2
    e = len(set(loadfile(Es[0], 1)) | set(loadfile(Es[1], 1)))  # 无重复ID
    adj = get_weighted_adj(e, KG)  # nx.adjacency_matrix(nx.from_dict_of_lists(get_dic_list(e, KG)))

    ent2id = get_ent2id([Es[0], Es[1]])
    attr, attr_embeddings = loadattr([As[0], As[1]], e, ent2id)
    ae_input = get_ae_input(attr)

    date_info = np.load(data_path + 'triples_1_tem.npy').tolist() +\
                np.load(data_path + 'triples_2_tem.npy').tolist()
    suffix = '_seed_' + str(FLAGS.seed) + '.npy'
    try:
        train = np.load(data_path + 'train' + suffix).tolist()
        test = np.load(data_path + 'test' + suffix).tolist()
        timeSource = np.load(data_path + 'timeSource' + suffix, allow_pickle=True).tolist()
    except:
        ill = ill[0]
        ILL = loadfile(ill, 2)
        illL = len(ILL)
        np.random.shuffle(ILL)
        train = ILL[:illL // 10 * FLAGS.seed]
        test = ILL[illL // 10 * FLAGS.seed:]
        timeSource = process_time(train, KG, len(KG1), data_path, suffix)
        np.save(data_path + 'train' + suffix, train)
        np.save(data_path + 'test' + suffix, test)
    return adj, ae_input, attr_embeddings, train, test, date_info, timeSource


def process_time(train_list, KG, KG1_len, data_path, suffix):
    train = set(train_list)
    left = [x[0] for x in train]
    right = [x[1] for x in train]
    print('update timeSource1. need a long time.')
    timeSource = dict()
    KG_len = len(KG)
    for rg_1 in range(KG1_len):
        id_1 = KG[rg_1][0]
        if rg_1%5000 == 0:
            print('rg_1 = ', rg_1)
        for rg_2 in range(KG1_len, KG_len):
            id_2 = KG[rg_2][0]
            if (id_1, id_2) not in train and (KG[rg_1][2], KG[rg_2][2]) in train:
                if id_1 in left:
                    if id_1 not in timeSource:
                        timeSource[id_1] = {}
                    if id_2 not in timeSource[id_1]:
                        timeSource[id_1][id_2] = [(rg_1, rg_2)]
                    else:
                        timeSource[id_1][id_2].append((rg_1, rg_2))
                if id_2 in right:
                    if id_2 not in timeSource:
                        timeSource[id_2] = {}
                    if id_1 not in timeSource[id_2]:
                        timeSource[id_2][id_1] = [(rg_2, rg_1)]
                    else:
                        timeSource[id_2][id_1].append((rg_2, rg_1))


    np.save(data_path + 'timeSource' + suffix, timeSource)
    print('over')
    return timeSource