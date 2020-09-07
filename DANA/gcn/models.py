import cPickle
from gcn.layers import *
from gcn.metrics import *
from gcn.flip_gradient import flip_gradient

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'dataset'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        dataset = kwargs.get('dataset', 'dataset')
        self.dataset = dataset

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.domain_layers = []
        self.activations = []  # input + hidden layer outputs

        self.inputs = None
        self.others_left = None
        self.others_right = None
        self.outputs_left = None
        self.outputs_right = None
        self.domain_outputs = None

        self.global_step = tf.Variable(0, trainable=False)
        self.domain_l = 1.0

        self.loss = 0  # loss for alignment
        self.domain_loss = 0  # loss for domain predictor
        self.accuracies = []
        self.optimizer = None
        self.domain_optimizer = None
        self.opt_op = None
        self.domain_opt_op = None
        self.saver = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        with tf.name_scope("procedure_feature_extractor"):
            layers_num = len(self.layers)
            # for left network
            self.activations.append(self.layers[0]())
            for l in xrange(1, layers_num/2):
                with tf.name_scope("left_conver_%s" % l):
                    layer = self.layers[l]
                    hidden = layer(self.activations[ -1], others=self.others_left)
                    self.activations.append(hidden)
            self.outputs_left = self.activations[-1]

        # for right network
            self.activations.append(self.layers[layers_num/2]())
            for l in xrange(layers_num/2 + 1, layers_num):
                with tf.name_scope("right_conver_%s" % l):
                    layer = self.layers[l]
                    hidden = layer(self.activations[-1], others=self.others_right)
                    self.activations.append(hidden)
            self.outputs_right = self.activations[-1]

        # for domain predictor
        with tf.name_scope("procedure_domain_predictor"):
            # Get anchors
            left_embedding = self.outputs_left
            right_embedding = self.outputs_right
            nodes_left = tf.nn.embedding_lookup(left_embedding, self.placeholders['domain_left_idx'])
            nodes_right = tf.nn.embedding_lookup(right_embedding, self.placeholders['domain_right_idx'])
            input_nodes = tf.concat([nodes_left, nodes_right], axis=0)
            the_input = flip_gradient(input_nodes, self.domain_l)
            self.activations.append(the_input)
            for layer in self.domain_layers:
                hidden = layer(self.activations[-1])
                self.activations.append(hidden)
            self.domain_outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}


        # Build metrics
        self._loss()
        with tf.name_scope("align_accuracy"):
            self._accuracy()
        up_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+"/domain_predictor")
        up_variables_2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + "/feature_extractor")

        with tf.name_scope("optimizer_minimize"):
            self.opt_op = self.optimizer.minimize(self.loss, self.global_step, var_list=up_variables_2)
            self.domain_opt_op = self.domain_optimizer.minimize(self.domain_loss, var_list=up_variables)

    def predict(self):
        pass

    def get_embedding(self, output):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save_embeddings(self, sess=None):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        save_path = self.saver.save(sess, "vars/%s/train/%s.ckpt" % (self.dataset, self.name), global_step=self.global_step)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        save_path = "vars/%s/train/"%self.dataset
        self.saver.restore(sess, tf.train.latest_checkpoint(save_path))
        print("Model restored from file: %s" % tf.train.latest_checkpoint(save_path))


class GCN(Model):
    """
    placeholders = {
    'support': D^{1/2}(A+I)D^{1/2}
    'features': features of all nodes
    'labels': classification labels
    'labels_mask': the indidation of train nodes
    'dropout': default 0
    'num_features_nonzero': helper variable for sparse dropout
    }
    """
    def __init__(self, placeholders, left_node_num, right_node_num, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.left_node_num = left_node_num
        self.right_node_num = right_node_num
        self.input_dim = FLAGS.input_dim
        self.output_dim = FLAGS.output_dim
        self.placeholders = placeholders
        self.others_left = [placeholders['support_left']]
        self.others_right = [placeholders['support_right']]

        self.eva_loss = 0.
        self.domain_regular_loss = 0.
        with tf.name_scope("optimizers"):
            learing_rate = tf.train.exponential_decay(FLAGS.learning_rate, self.global_step, FLAGS.lr_decay_step, 0.6, staircase=True)
            self.multiplier = tf.train.piecewise_constant(self.global_step, boundaries=[5, 25, 50], values=[200., 25., 10., 5.])
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learing_rate)
            self.domain_optimizer = tf.train.AdamOptimizer(learning_rate=learing_rate)
            x = tf.cast(self.global_step, dtype=tf.float32) / FLAGS.epochs
            self.domain_l = (2 / (1 + tf.exp(-10 * x)) - 1)

        with tf.variable_scope("soft_bias"):
            self.left_node_bias = tf.Variable(tf.zeros([self.left_node_num], dtype=tf.float32), trainable=False)
            self.right_node_bias = tf.Variable(tf.zeros([self.right_node_num], dtype=tf.float32), trainable=False)


        self.build()
        with tf.variable_scope("visualization"):
            self.visual_embeddings = tf.concat([self.outputs_left, self.outputs_right], axis=0, name="two_networks_embeded")
            self.vars["two_networks_embeded"] = tf.get_variable(shape=(self.left_node_num+self.right_node_num, self.output_dim),
                                                          name="two_networks_embeded")
            self.saver = tf.train.Saver(self.vars, max_to_keep=5)


    def _loss(self):
        # Weight decay loss: regularization
        with tf.name_scope("reg_feature_extractor"):
            feature_layer_num = len(self.layers)
            for layer in [self.layers[0], self.layers[feature_layer_num/2]]:
                for var in layer.vars.values():
                    self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
            for layer in self.layers[1:feature_layer_num/2]:
                for var in layer.vars.values():
                    self.loss += 10 * FLAGS.weight_decay * tf.nn.l2_loss(var)
            for layer in self.layers[feature_layer_num / 2 + 1 : feature_layer_num]:
                for var in layer.vars.values():
                    self.loss += 10 * FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Weight decay loss: regularization domain predictor
        with tf.name_scope("reg_domain_predictor"):
            for layer in self.domain_layers:
                for var in layer.vars.values():
                    self.domain_regular_loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # # loss for alignment
        with tf.name_scope("loss_alignment_indictor"):
            left_embedding = get_embedding(self.outputs_left)
            right_embedding = get_embedding(self.outputs_right)
            nodes_left = tf.nn.embedding_lookup(left_embedding, self.placeholders['anchor_left_idx'])
            nodes_right = tf.nn.embedding_lookup(right_embedding, self.placeholders['anchor_right_idx'])
            self.eva_loss = align_loss_with_neg_sample(nodes_left, self.placeholders['anchor_right_idx'],
                                                       right_embedding, self.right_node_bias, self.right_node_num)
            self.eva_loss += align_loss_with_neg_sample(nodes_right, self.placeholders['anchor_left_idx'],
                                                        left_embedding, self.left_node_bias, self.left_node_num)
            self.loss += self.eva_loss

        # loss for domain predictor
        with tf.name_scope("loss_domain_predictor"):
            left_domain_num = tf.shape(self.placeholders['domain_left_idx'])[0]
            right_domain_num = tf.shape(self.placeholders['domain_right_idx'])[0]
            labels = tf.concat([tf.tile([1.], [left_domain_num]), tf.tile([0.], [right_domain_num])], axis=0)
            self.domain_loss = domains_loss(self.domain_outputs, labels) * self.multiplier
            self.loss += self.domain_loss
            self.domain_loss += self.domain_regular_loss

        tf.summary.scalar('gcn_loss', self.loss)
        self.summary_eva_loss = tf.summary.scalar('gcn_align_loss', self.eva_loss)
        tf.summary.scalar('gcn_domain_loss', self.domain_loss)


    def _accuracy(self):
        # add domain accuracy
        self.domain_score = tf.nn.sigmoid(self.domain_outputs)
        variable_summaries(self.domain_score, "domain_softmax")
        labels = tf.concat([tf.tile([1.], [tf.shape(self.placeholders['domain_left_idx'])[0]]),
                            tf.tile([0.], [tf.shape(self.placeholders['domain_right_idx'])[0]])], axis=0)
        domain_acc = tf.reduce_mean(tf.nn.l2_loss(self.domain_score - labels))
        self.accuracies.append(domain_acc)

        left_embeddings = get_embedding(self.outputs_left)
        right_embeddings = get_embedding(self.outputs_right)
        nodes_left = tf.nn.embedding_lookup(left_embeddings, self.placeholders['anchor_left_idx'])
        nodes_right = tf.nn.embedding_lookup(right_embeddings, self.placeholders['anchor_right_idx'])

        #for left -> right
        right_candidates_score = align_score(nodes_left, right_embeddings)
        left2right_accuracies = align_accuracy(right_candidates_score, self.placeholders['anchor_right_idx'], self.placeholders['anchor_right_mask'])
        self.left2right_nearest_score = get_nearest_score(right_candidates_score, self.placeholders['anchor_right_mask'])

        #for right -> left
        left_candidates_score = align_score(nodes_right, left_embeddings)
        right2left_accuracies = align_accuracy(left_candidates_score, self.placeholders['anchor_left_idx'], self.placeholders['anchor_left_mask'])
        self.right2left_nearest_score = get_nearest_score(left_candidates_score, self.placeholders['anchor_left_mask'])

        self.summary_l2r_acc = tf.summary.scalar('gcn_hits@50_left2right', tf.reduce_mean(left2right_accuracies[-1]))
        self.summary_r2l_acc = tf.summary.scalar('gcn_hits@50_rigfht2left', tf.reduce_mean(right2left_accuracies[-1]))
        #for sum accuracies
        accuracies_len = len(left2right_accuracies)
        for i in xrange(accuracies_len):
            acc_for_each_samples = (left2right_accuracies[i]+right2left_accuracies[i])/2
            acc_for_all = tf.reduce_mean(acc_for_each_samples)
            self.accuracies.append(acc_for_all)

    def _build(self):

        with tf.variable_scope('feature_extractor'):
            self.layers.append(Init_Input(input_dim=self.left_node_num,
                                          output_dim=self.input_dim ,
                                          placeholders=self.placeholders,
                                          dropout=False,
                                          sparse_inputs=False,
                                          logging=self.logging))

            self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=FLAGS.hidden1,
                                                placeholders=self.placeholders,
                                                act=tf.nn.relu,
                                                dropout=False,
                                                batch_normal=False,
                                                logging=self.logging))

            self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                                output_dim=self.output_dim,
                                                placeholders=self.placeholders,
                                                act=lambda x: x,
                                                dropout=False,
                                                batch_normal=False,
                                                logging=self.logging))

            self.layers.append(Init_Input(input_dim=self.right_node_num,
                                          output_dim=self.input_dim,
                                          placeholders=self.placeholders,
                                          dropout=False,
                                          sparse_inputs=False,
                                          logging=self.logging))

            self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=FLAGS.hidden1,
                                                placeholders=self.placeholders,
                                                act=tf.nn.relu,
                                                dropout=False,
                                                batch_normal=False,
                                                logging=self.logging))

            self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                                output_dim=self.output_dim,
                                                placeholders=self.placeholders,
                                                act=lambda x: x,
                                                dropout=False,
                                                batch_normal=False,
                                                logging=self.logging))

        with tf.variable_scope("domain_predictor"):
            self.domain_layers.append(Dense(input_dim=self.output_dim,
                                            output_dim=int(self.output_dim/2),
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            bias=True,
                                            dropout=False,
                                            sparse_inputs=False,
                                            batch_normal=False,
                                            logging=self.logging))
            self.domain_layers.append(Dense(input_dim=int(self.output_dim/2),
                                            output_dim=1,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            bias=False,
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging))


    def predict(self):
        pass

    def save_embeddings(self, sess=None, feed_dict=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        embeddings = sess.run([self.outputs_left, self.outputs_right], feed_dict=feed_dict)
        cPickle.dump(embeddings, open("outputs/%s_embeddings_%s.pkl" % (self.dataset, self.name), 'w'))
