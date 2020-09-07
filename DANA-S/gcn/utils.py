# coding=utf-8

import sys
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from collections import defaultdict


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def pkl_load(file_obj):
    if sys.version_info > (3, 0):
        return pkl.load(file_obj, encoding='latin1')
    else:
        return pkl.load(file_obj)

def load_anchors_data_one(dataset_str):
    """
    处理IONE数据：将只有一列的anchor，变成两列
    :param dataset_str:
    :return:
    """
    data_list = ["train", "test"]
    for data in data_list:
        with open("../../data/ind.{}.anchors.{}".format(dataset_str, data),'r') as file_obj:
            lines = file_obj.readlines()
        with open("../../data/ind.{}.anchors.{}".format(dataset_str, data), 'w') as file_obj:
            for line in lines:
                line = line.strip()
                line = line + '\t' + line + '\n'
                file_obj.write(line)


def load_source_data(dataset_str):
    """
    Loads input source data from gcn/data directory
    ind.dataset_str.edges.left => the edges in network left, each line as (node_index, node_index), of dataset_str
    ind.dataset_str.edges.right => the edges in network right, each line as (node_index, node_index), of dataset_str
    :param dataset_str: Dataset name
    :return: stored all data input files as format-file:
             ind.dataset_str.graph.left => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict object;
             ind.dataset_str.graph.right  => have the same format as ind.dataset_str.graph.left
    """
    network_list = ["left", "right"]
    for network in network_list:
        graph_dict = defaultdict(list)
        with open("../../data/ind.{}.edges.{}".format(dataset_str, network)) as fread:
            for line in fread:
                line = line.strip().split()
                graph_dict[int(line[0])].append(int(line[1]))
        for node in graph_dict:
            graph_dict[node] = list(set(graph_dict[node]))
        pkl.dump(graph_dict, open("../../ind.{}.graph.{}".format(dataset_str, network), 'wb'))

    data_list = ["train", "test"]
    for data in data_list:
        anchors = []
        with open("../../data/ind.{}.anchors.{}".format(dataset_str, data)) as fread:
            for line in fread:
                line = line.strip().split()
                anchors.append([int(line[0]), int(line[1])])
        anchors = np.array(anchors)
        pkl.dump(anchors, open("../../ind.{}.anchor.{}".format(dataset_str, data), 'wb'))


def dict_to_adj(the_dict, directed=True):
    if directed:
        graph = nx.from_dict_of_lists(the_dict, create_using=nx.DiGraph())
    else:
        graph = nx.from_dict_of_lists(the_dict)
    return nx.adjacency_matrix(graph, nodelist=sorted(graph.nodes()))


def load_data(dataset_str, left_directed=False, right_directed=False, split_rate="82"):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).

    """
    names = {'graph':['left', 'right'], 'anchor':['train'+split_rate,'test'+split_rate]}
    objects = []
    len_names = len(names)
    for i in names:
        for j in names[i]:
            with open("../../data/ind.{}.{}.{}".format(dataset_str, i, j), 'rb') as f:
                objects.append(pkl_load(f))
                if isinstance(objects[-1], defaultdict):
                    print('{}.{}'.format(i, j), type(objects[-1]), len(objects[-1]), objects[-1][0])
                else:
                    print('{}.{}'.format(i,j), type(objects[-1]), objects[-1].shape, objects[-1][0])

    graph_left, graph_right, anchor_train, anchor_test = tuple(objects)
    adj_left = dict_to_adj(graph_left, directed=left_directed)
    adj_right = dict_to_adj(graph_right, directed=right_directed)

    np.random.shuffle(anchor_train)
    anchor_val = anchor_train[30:30]
    anchor_train = anchor_train

    print("*" * 30 + "\nload data output information\n" + "*" * 30)
    print("adj_left:", type(adj_left), adj_left.shape)
    print("adj_right:", type(adj_right), adj_right.shape)
    print("anchor_train:", type(anchor_train), anchor_train.shape)
    print("anchor_val:", type(anchor_val), anchor_val.shape)
    print("anchor_test:", type(anchor_test), anchor_test.shape)
    return adj_left, adj_right, anchor_train, anchor_val, anchor_test


def load_edges_data(dataset_str):
    network_list = ["left", "right"]
    edges_dict = {}
    for network in network_list:
        edges_list = []
        with open("../../ind.{}.edges.{}".format(dataset_str, network)) as fread:
            for line in fread:
                line = line.strip().split()
                edges_list.append([int(x) for x in line])
        edges_dict[network] = np.array(edges_list)
    return edges_dict[network_list[0]], edges_dict[network_list[1]]

def get_next_batch(start_idx, data_len, batch_size=64):
    if start_idx + batch_size <= data_len:
        idx_list = range(start_idx, start_idx+batch_size)
    else:
        idx_list = range(start_idx, data_len)
        idx_list += range(0, batch_size-data_len+start_idx)
    next_start_idx = (start_idx + batch_size) % data_len
    return next_start_idx, idx_list


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


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj, is_transpose=False):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    if is_transpose:
        adj = adj.transpose()
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def normalize_adj_v2(adj, is_transpose=False):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    if is_transpose:
        adj = adj.transpose()
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # print("nomalization version2")
    return d_mat_inv_sqrt.dot(adj).tocoo()


def normalize_adj_v3(adj, is_transpose=False):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    if is_transpose:
        adj = adj.transpose()
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # print("nomalization version3")
    return adj.dot(d_mat_inv_sqrt).tocoo()

def normalize_adj_v4(adj):
    """Symmetrically normalize adjacency matrix."""
    adj_nor = normalize_adj_v3(adj)
    adj_nor2 = normalize_adj_v2(adj_nor)
    return adj_nor2

def preprocess_adj(adj, transpose=False):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]), transpose)
    return sparse_to_tuple(adj_normalized)


def get_nodes_probability(adj, is_transpose=False):
    adj = sp.coo_matrix(adj)
    if is_transpose:
        adj = adj.transpose()
    rowsum = np.array(adj.sum(1))
    probability = rowsum / float(np.sum(rowsum))
    probability = np.reshape(probability, [-1])
    return list(probability)

def construct_feed_dict(support_left, support_right, anchor_left_idx, anchor_right_idx,
                        anchor_left_mask, anchor_right_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['anchor_left_mask']: anchor_left_mask})
    feed_dict.update({placeholders['anchor_right_mask']: anchor_right_mask})
    feed_dict.update({placeholders['anchor_left_idx']: anchor_left_idx})
    feed_dict.update({placeholders['anchor_right_idx']: anchor_right_idx})
    feed_dict.update({placeholders['support_left'][i]: support_left[i] for i in range(len(support_left))})
    feed_dict.update({placeholders['support_right'][i]: support_right[i] for i in range(len(support_right))})
    feed_dict.update({placeholders['num_features_nonzero']: 0})
    return feed_dict



def get_domain_batch(max_num, batch_size=64, probability=None):
    batch_samples = np.random.choice(max_num, size=batch_size, replace=False, p=probability)
    return batch_samples


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


def create_metadata(file_path, left_num, right_num):
    with open(file_path, 'w') as f:
        f.write("Index\tLabel\n")
        for index in xrange(left_num):
            f.write("%d\t%d\n" % (index, 0))
        for index in xrange(left_num, left_num + right_num):
            f.write("%d\t%d\n" % (index, 1))
