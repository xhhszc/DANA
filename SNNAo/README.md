This is the codes for SNNAo model which is a baseline in our paper "Domain-adversarial Network Alignment".

# Prerequisites
- python2.7

- tensorflow

- cPickle

- argparse

- numpy

- linecache



# How to run

For SNNAo, you need reprocess the dataset firstly, take 'foursquare-twitter' as an example:

1. Get input files: use deepwalk (output dimension=100) to obtain embeddings of nodes in foursquare and twitter;

deepwalk--github resource:https://github.com/phanein/deepwalk

2. Update the setting in the main.py:
``--input_netl_nodes --input_netr_nodes --input_netl_weight --input_netr_weight --input_netl_anchors --input_netr_anchors --input_train_anchors --input_test_anchors --input_test_netl_embeddings --input_test_netr_embeddings --left_nodes_num --right_nodes_num --train_anchors_num --test_anchors_num --epoch_all ``

3. Create directory ``output`` and ``log``

4. run the model with ``python main.py``


