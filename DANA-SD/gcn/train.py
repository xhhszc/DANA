# coding=utf-8

from __future__ import division
from __future__ import print_function

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import shutil
import time
import pprint
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from gcn.utils import *
from gcn.models import GCN


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'flickr_twit', 'Dataset string.')
flags.DEFINE_string('model', 'gcn', 'Model string.')
flags.DEFINE_boolean('left_directed', True, 'if left-network is directed')
flags.DEFINE_boolean('right_directed', True, 'if right-network is directed')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
flags.DEFINE_float('lr_decay_step', 200, 'Initial learning decay rate.')
flags.DEFINE_integer('epochs', 2000, 'Number of epochs to train.')
flags.DEFINE_integer('iterations', 20, 'Number of iterations to train feature extractor.')
flags.DEFINE_integer('domain_iterations', 50, 'Number of iterations to train domain predictor.')
flags.DEFINE_integer('input_dim', 2048, 'Number of units in input layer.')
flags.DEFINE_integer('hidden1', 256, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('output_dim', 50, 'Number of units in output layer.')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 1e-3, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 800, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('logging_freq', 50, 'the frequency of logging.')
flags.DEFINE_boolean('copy_align_col', False, 'to make alignment file as two cols.')
flags.DEFINE_boolean('from_source_data', False, 'to prepare data.')
flags.DEFINE_boolean('debug_mode', False, 'to break up in the stop node.')
flags.DEFINE_boolean('restore_mode', False, 'to restore model form check point')
flags.DEFINE_string('suffix', 'directed_unshared_domain', 'some points mark')
flags.DEFINE_integer('domain_batch', 512, 'Num of batch for domain.')
flags.DEFINE_float('multiplier', 25, 'the multiplier for domain classifier.')
flags.DEFINE_float('mul_decay_step', 5, 'the multiplier for domain classifier.')
flags.DEFINE_string('split_rate', "82", 'the split rate of anchors traning:test')

if __name__ == '__main__':
    for arg in FLAGS.__flags:
        the_arg = eval("FLAGS." + arg)
        print(arg, the_arg)

if not os.path.exists('tmp'):
    os.makedirs('tmp')
if not os.path.exists('outputs'):
    os.makedirs('outputs')
if not FLAGS.restore_mode and os.path.exists('vars/%s' % FLAGS.dataset):
    shutil.rmtree('vars/%s' % FLAGS.dataset)
if not os.path.exists('vars/%s' % FLAGS.dataset):
    os.makedirs('vars/%s/train' % FLAGS.dataset)


# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Load data
if FLAGS.copy_align_col:
    load_anchors_data_one(FLAGS.dataset)
if FLAGS.from_source_data:
    load_source_data(FLAGS.dataset)
adj_left, adj_right, anchor_train, anchor_val, anchor_test = load_data(FLAGS.dataset, FLAGS.left_directed, FLAGS.right_directed, FLAGS.split_rate)

left_node_num = list(adj_left.shape)[0]
right_node_num = list(adj_right.shape)[0]
print("left_num: %s \t right_num: %s" % (left_node_num, right_node_num))

left_nodes_prob = get_nodes_probability(adj_left)
right_nodes_prob = get_nodes_probability(adj_right)

# The stop node
if FLAGS.debug_mode:
    exit(0)

k_values = [1, 5, 10, 20, 30, 40, 50]
print_k = ["domain_acc||"]
print_k += ["hits@%s||" % k for k in k_values]
print_k += ["MRR||"]
string_print_k = '\t'.join(print_k)

if FLAGS.model == 'gcn':
    support_left_out = [preprocess_adj(adj_left)]  # support = D^(1/2)(A+I)D^(1/2)
    support_left_in = [preprocess_adj(adj_left, transpose=True)]
    support_right_out = [preprocess_adj(adj_right)]
    support_right_in = [preprocess_adj(adj_right, transpose=True)]
    support_left = [support_left_out, support_left_in]
    support_right = [support_right_out, support_right_in]
    num_supports = 1  # support num of a network
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support_left_out = chebyshev_polynomials(adj_left, FLAGS.max_degree)
    support_left_in = chebyshev_polynomials(adj_left, FLAGS.max_degree, transpose=True)
    support_right_out = chebyshev_polynomials(adj_right, FLAGS.max_degree)
    support_right_in = chebyshev_polynomials(adj_right, FLAGS.max_degree, transpose=True)
    support_left = [support_left_out, support_left_in]
    support_right = [support_right_out, support_right_in]
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

print(num_supports)
# Define placeholders
placeholders = {
    'support_left_out': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'support_left_in': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'support_right_out': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'support_right_in': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'anchor_left_idx': tf.placeholder(tf.int32),
    'anchor_right_idx': tf.placeholder(tf.int32),  #13
    'anchor_left_mask': tf.placeholder(tf.float32, shape=(1, left_node_num)),
    'anchor_right_mask': tf.placeholder(tf.float32, shape=(1, right_node_num)),
    'domain_left_idx': tf.placeholder(tf.int32),
    'domain_right_idx': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, left_node_num=left_node_num, right_node_num=right_node_num,
                logging=True, name=FLAGS.suffix, dataset=FLAGS.dataset)

# Create options to profile the time and memory information.
builder = tf.profiler.ProfileOptionBuilder
builder_opts = builder(builder.time_and_memory())
builder_opts.with_file_output(outfile='tmp/%s_profiler.txt'%FLAGS.dataset)
builder_opts.with_min_memory(min_bytes=1024*1024)
builder_opts.order_by('bytes')
builder_opts = builder_opts.build()
run_metadata = tf.RunMetadata()
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
merged = tf.summary.merge_all()
merged_test = tf.summary.merge([model.summary_eva_loss, model.summary_l2r_acc, model.summary_r2l_acc])

# The Start Point
project_start_time = time.time()
with tf.contrib.tfprof.ProfileContext('tmp/%s_profiler'%FLAGS.dataset, trace_steps=range(0,200,100), dump_steps=range(0,200,100)) as pctx:
    # Initialize session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config_proj = projector.ProjectorConfig()
    embed_visual = config_proj.embeddings.add()
    embed_visual.tensor_name = "two_networks_embeded"
    # Specify where you find the metadata
    path_for_embeded_metadata = os.path.join('vars/%s/train'%FLAGS.dataset, 'metadata.tsv')
    create_metadata(path_for_embeded_metadata, left_node_num, right_node_num)
    embed_visual.metadata_path = 'metadata.tsv'

    with tf.Session(config=config) as sess:
        # Init variables
        train_writer = tf.summary.FileWriter('vars/%s/train'%FLAGS.dataset, sess.graph)
        test_writer = tf.summary.FileWriter('vars/%s/test'%FLAGS.dataset, sess.graph)
        # Say that you want to visualise the embeddings
        projector.visualize_embeddings(train_writer, config_proj)
        sess.run(tf.global_variables_initializer())
        sess.graph.finalize()
        if FLAGS.restore_mode:
            model.load(sess)
        cost_val = []
        best_acc_50 = 0.


        print("start training....")

        # Define model evaluation function
        def evaluate(epoch, support_left, support_right, anchor_left_idx, anchor_right_idx, placeholders):
            t_test = time.time()
            left_node_mask = np.ones([1, left_node_num])
            left_node_mask[0, anchor_train[:,0]] = 0.
            right_node_mask = np.ones([1, right_node_num])
            right_node_mask[0, anchor_train[:,1]] = 0.
            feed_dict_val = construct_feed_dict(support_left, support_right, anchor_left_idx, anchor_right_idx,
                                                left_node_mask, right_node_mask, placeholders)
            feed_dict_val.update({placeholders['domain_left_idx']: range(left_node_num)})
            feed_dict_val.update({placeholders['domain_right_idx']: range(right_node_num)})
            outs_val = sess.run([model.eva_loss, model.accuracies], feed_dict=feed_dict_val, options=run_options, run_metadata=run_metadata)
            if epoch % FLAGS.logging_freq == 0:
                summary = sess.run(merged_test, feed_dict=feed_dict_val)
                test_writer.add_summary(summary, epoch)
                test_writer.flush()
            return outs_val[0], outs_val[1], (time.time() - t_test)

        # defined train model
        def train(epoch, model_opt, feed_dict_train):
            t = time.time()

            outs = sess.run([model_opt, model.loss, model.accuracies, model.eva_loss, model.domain_loss],
                            feed_dict=feed_dict_train)
            if epoch % FLAGS.logging_freq == 0:
                summary = sess.run(merged, feed_dict=feed_dict_train)
                train_writer.add_summary(summary, epoch)
                train_writer.flush()
            return outs, (time.time() - t)

        def print_results(the_epoch, the_train_time, the_outs, the_test_acc):
            # Print results
            print("Epoch:", '%05d' % (the_epoch + 1), "time=", "{:.5f}".format(the_train_time),
                  "train_align_loss=", "{:.5f}".format(the_outs[3]), "domain_loss=", "{:.5f}".format(the_outs[4]),
                  "train_loss=", "{:.5f}".format(the_outs[1]))
            print("      ", '     ', string_print_k)
            string_list_accs = ["{:.5f}".format(the_outs[2][i]) for i in xrange(len(the_outs[2]))]
            print("      ", 'train', '||\t'.join(string_list_accs))
            string_list_accs = ["{:.5f}".format(the_test_acc[i]) for i in xrange(len(the_test_acc))]
            print("      ", 'test ', '||\t'.join(string_list_accs))


        # Train model
        left_node_mask_train = np.ones([1, left_node_num])
        right_node_mask_train = np.ones([1, right_node_num])
        # Construct feed dictionary
        feed_dict = construct_feed_dict(support_left, support_right, anchor_train[:, 0], anchor_train[:, 1],
                                        left_node_mask_train, right_node_mask_train, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        epoch = 0
        while epoch < FLAGS.epochs:
            for iter in xrange(FLAGS.iterations):
                batch_domain_left = get_domain_batch(left_node_num, batch_size=FLAGS.domain_batch,
                                                     probability=left_nodes_prob)
                batch_domain_right = get_domain_batch(right_node_num, batch_size=FLAGS.domain_batch,
                                                      probability=right_nodes_prob)
                feed_dict.update({placeholders['domain_left_idx']: batch_domain_left})
                feed_dict.update({placeholders['domain_right_idx']: batch_domain_right})
                #training
                outs, train_time = train(epoch, model.opt_op, feed_dict)
                # Testing
                test_cost, test_acc, test_duration = evaluate(epoch, support_left, support_right, anchor_test[:, 0],
                                                              anchor_test[:, 1], placeholders)
                cost_val.append(test_cost)
                if test_acc[-1] > best_acc_50:
                    model.save(sess)
                    best_acc_50 = test_acc[-1]
                print_results(epoch, train_time, outs, test_acc)
                epoch += 1
            
            for iter in xrange(FLAGS.domain_iterations):
                batch_domain_left = get_domain_batch(left_node_num, batch_size=FLAGS.domain_batch,
                                                     probability=left_nodes_prob)
                batch_domain_right = get_domain_batch(right_node_num, batch_size=FLAGS.domain_batch,
                                                      probability=right_nodes_prob)
                feed_dict.update({placeholders['domain_left_idx']: batch_domain_left})
                feed_dict.update({placeholders['domain_right_idx']: batch_domain_right})
                #training
                outs, train_time = train(epoch, model.domain_opt_op, feed_dict)
                # Testing
                test_cost, test_acc, test_duration = evaluate(epoch, support_left, support_right, anchor_test[:, 0],
                                                              anchor_test[:, 1], placeholders)
                cost_val.append(test_cost)
                if test_acc[-1] > best_acc_50:
                    model.save(sess)
                    best_acc_50 = test_acc[-1]
                print_results(epoch, train_time, outs, test_acc)
                epoch += 1

            if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
                print("Early stopping...")
                break

        print("Optimization Finished!")

        # Testing for best model
        model.load(sess)
        model.save_embeddings(sess, feed_dict)
        test_cost, test_acc, test_duration = evaluate(FLAGS.epochs + 500, support_left, support_right, anchor_test[:,0], anchor_test[:,1], placeholders)
        print("Test set results:", "cost=", "{:.5f}".format(test_cost), "time=", "{:.5f}".format(test_duration))
        print("      ", '     ', string_print_k)
        string_list_accs = ["{:.5f}".format(test_acc[i]) for i in xrange(len(test_acc))]
        print("      ", 'test ', '||\t'.join(string_list_accs))
project_end_time = time.time()
print("the run time£º%s" %(project_end_time-project_start_time))
with open("outputs/%s-dim.txt"%FLAGS.suffix, 'a') as fdim:
    fdim.write('%sd'%FLAGS.output_dim+'\t'+'\t'.join(string_list_accs[1:])+'\n')