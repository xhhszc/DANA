# coding=utf-8
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import candidate_sampling_ops
TINY = 1e-8

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def align_loss_with_neg_sample(train_nodes_embed, train_labels, nce_weights, nce_biases, num_classes):
    """
    :param train_nodes_embed: embedding of anchor node in one network
    :param train_labels: anchor_idx in another network
    :param nce_weights: embeddings of another network
    :param nce_biases: biases of another network
    :param num_classes: node num of another network
    :return: alinement loss from one network to another network
    """
    loss = tf.nn.sampled_softmax_loss(weights=nce_weights, biases=nce_biases,
                                      inputs=train_nodes_embed, labels=tf.reshape(train_labels, [-1, 1]),
                                      num_sampled=50, num_classes=num_classes,
                                      remove_accidental_hits=True, partition_strategy='div')
    return tf.reduce_mean(loss)



def domains_loss(preds, labels):
    prob_reshape = tf.reshape(preds, [-1])
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=prob_reshape, labels=labels)
    return tf.reduce_mean(loss)


def wasserstein_distance(distribution_1, distribution_2):
    return -tf.reduce_mean(distribution_1 - distribution_2)


def align_score(node_list, candidates):
    scores = tf.matmul(node_list, candidates, transpose_a=False, transpose_b=True)
    return tf.exp(scores)

def align_accuracy(pre_list, label_list, node_filter=None):
    """
    hits@k
    :param pre_list: the shape be [test_sample_num, class_num], where class num is the node_num in the network
    :param label_list: the shape be [test_sample_num]
    :param filter: the mask. the shape be [1, class_num]
    :return: a list of the hits@k varying k; the shape be len(k_values)*[test_sample_num]
    """
    acc_at_k = []
    k_values = [1,5,10,20,30,40,50]
    if node_filter is not None:
        pre_list = pre_list * node_filter
    for k in k_values:
        acc_k_bool  = tf.nn.in_top_k(pre_list, label_list, k)
        acc_k_float = tf.cast(acc_k_bool, dtype=tf.float32)
        acc_at_k.append(acc_k_float)
    #append mrr
    max_k = tf.shape(pre_list)[1]
    top_value, top_index = tf.nn.top_k(pre_list, max_k)
    rank_index = tf.where(tf.equal(top_index,tf.reshape(label_list,[-1,1])))[:,1] # int64 list
    mrr = tf.reduce_mean(1.0/(tf.cast(rank_index+1, dtype=tf.float32)))
    acc_at_k.append(mrr)
    return acc_at_k


def get_embedding(output):
    return tf.nn.l2_normalize(output, 1)


def variable_summaries(var, name=None):#记录变量var的信息
    # """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean_' + name, mean)#记录变量var的均值
        with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev_' + name, stddev)#记录变量var的方差
        tf.summary.scalar('max_' + name, tf.reduce_max(var))#记录变量var的最大值
        tf.summary.scalar('min_' + name, tf.reduce_min(var))#记录变量var的最小值
        tf.summary.histogram('histogram_' + name, var)#记录变量var的直方图

def get_nearest_score(scores, node_filter=None):
    if node_filter is not None:
        scores = scores * node_filter
    nearest_score = tf.reduce_max(scores, 1)
    return nearest_score

