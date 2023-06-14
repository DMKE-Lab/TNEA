from layers import *
import os
import sys
from metrics import *
from inits import *
from LSTMLinear import LSTMModel
import tensorflow.compat.v1 as tf
# tf.compat.v1.disable_eager_execution()
import tensorflow_hub as hub
from bert.tokenization.bert_tokenization import FullTokenizer
from bert import tokenization, modeling


flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN_Align(Model):
    def __init__(self, placeholders, input_dim, output_dim, ILL, sparse_inputs=False, featureless=True, **kwargs):
        super(GCN_Align, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.placeholders = placeholders
        self.ILL = ILL
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)

        self.lstm = LSTM(output_dim)
        self.build()

    def _loss(self):
        self.loss += align_loss(self.outputs, self.ILL, FLAGS.gamma, FLAGS.k)

    def _accuracy(self):
        pass

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=False,
                                            featureless=self.featureless,
                                            sparse_inputs=self.sparse_inputs,
                                            transform=False,
                                            init=trunc_normal,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=self.output_dim,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=False,
                                            transform=False,
                                            logging=self.logging))


class LSTM():
    timeSource = None
    date_info = None
    time_size = None

    def __init__(self, output_dim):

        self.tem_total = 32
        self.output_dim = output_dim
        self.lstm = LSTMModel(self.output_dim, n_layer=1)

        self.tem_embeddings = tf.Variable(
            tf.truncated_normal([self.tem_total, self.output_dim], stddev=1.0 / math.sqrt(self.output_dim)))
        self.tem_embeddings = tf.nn.l2_normalize(self.tem_embeddings, 1)

    @classmethod
    def get_right(cls, lstm_left, left, right):
        n = len(lstm_left)
        left_set = set(left)
        right_set = set(right)
        lstm_right = np.array([0 for i in range(n)])
        for rg in range(n):
            if lstm_left[rg] in left_set:
                p = left.index(lstm_left[rg])
                lstm_right[rg] = right[p]
            elif lstm_left[rg] in right_set:
                p = right.index(lstm_left[rg])
                lstm_right[rg] = left[p]
            else:
                raise ValueError('model.py  line258')
        return lstm_right

    @classmethod
    def get_neg(cls, lstm_left):
        num = len(lstm_left)
        lstm_neg = np.array([0 for i in range(num)])
        tem_left, tem_neg = [], []
        size = cls.time_size
        for rg in range(num):
            lstm_neg[rg] = np.random.choice( list(cls.timeSource[lstm_left[rg]]))
            l, n = cls.timeSource[lstm_left[rg]][lstm_neg[rg]][0]
            tem_left.extend(cls.date_info[l])
            tem_neg.extend(cls.date_info[n])

            '''
            t_l, t_n = [], []
            for l, n in cls.timeSource[lstm_left[rg]][lstm_neg[rg]]:
                t_l.extend(cls.date_info[l])
                t_n.extend(cls.date_info[n])
            t_l = list(set(t_l))
            t_n = list(set(t_n))
            tem_left.append(t_l[:size])
            tem_neg.append(t_n[:size])
        for t_l in tem_left:
            t_l.extend([2 for i in range(size - len(t_l))]) #20xx年，所以填充2
        for t_n in tem_neg:
            t_n.extend([2 for i in range(size - len(t_n))])
        
        tem_left = np.array([x for y in tem_left for x in y])
        tem_neg = np.array([x for y in tem_neg for x in y])
        '''
        tem_left = np.array(tem_left)
        tem_neg = np.array(tem_neg)
        assert len(tem_left) == len(tem_neg) == len(lstm_left) * cls.time_size
        return lstm_neg, tem_left, tem_neg

    def lose(self, outlayer, lstm_left, lstm_right, gamma):
        t = len(lstm_left)
        left_x = tf.nn.embedding_lookup(outlayer, lstm_left)
        right_x = tf.nn.embedding_lookup(outlayer, lstm_right)
        A = tf.reduce_sum(tf.abs(left_x - right_x), 1)  # t个 [ , , ,]

        B = self.get_eseq(outlayer, t)
        C = - B
        D = A + gamma
        L = tf.nn.relu(tf.add(C, D))
        return tf.reduce_sum(L) / (1.0 * t)

    def get_eseq(self, outlayer, size):

        lstm_left = tf.get_default_graph().get_tensor_by_name('lstm_left' + ":0")
        lstm_neg = tf.get_default_graph().get_tensor_by_name('lstm_neg' + ":0")
        left = tf.nn.embedding_lookup(outlayer, lstm_left)
        neg = tf.nn.embedding_lookup(outlayer, lstm_neg)  # [[,,,]]
        left = tf.transpose(tf.expand_dims(left, 0), perm=[1, 0, 2])
        neg = tf.transpose(tf.expand_dims(neg, 0), perm=[1, 0, 2])

        tem_left = tf.get_default_graph().get_tensor_by_name('tem_left' + ":0")
        tem_neg = tf.get_default_graph().get_tensor_by_name('tem_neg' + ":0")
        tem_l = tf.nn.embedding_lookup(self.tem_embeddings, tem_left)  # [[,,,],[,,,]]
        tem_n = tf.nn.embedding_lookup(self.tem_embeddings, tem_neg)
        tem_l = tf.reshape(tem_l, [size, self.time_size, self.output_dim])  # [[[,,,],[,,,]]]
        tem_n = tf.reshape(tem_n, [size, self.time_size, self.output_dim])

        seq_e_l = tf.concat([left, tem_l], 1)
        seq_e_n = tf.concat([neg, tem_n], 1)

        hidden_tem = self.lstm(seq_e_l)
        hidden_tem_l = hidden_tem[0, :, :]

        hidden_tem = self.lstm(seq_e_n)
        hidden_tem_n = hidden_tem[0, :, :]

        return tf.reduce_sum(tf.abs(hidden_tem_l - hidden_tem_n), 1)


class BertEncoder:
    def __init__(self, bert_model_dir, max_seq_length=128):
        self.bert_model_dir = bert_model_dir
        self.max_seq_length = max_seq_length

        self.tokenizer = self.get_tokenizer()
        self.graph, self.sess = self.build_graph()

    def get_tokenizer(self):
        vocab_file = os.path.join(self.bert_model_dir, 'vocab.txt')
        return tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

    def build_graph(self):
        graph = tf.Graph()
        with graph.as_default():
            input_ids = tf.placeholder(tf.int32, (None, self.max_seq_length), name="input_ids")
            input_mask = tf.placeholder(tf.int32, (None, self.max_seq_length), name="input_mask")
            segment_ids = tf.placeholder(tf.int32, (None, self.max_seq_length), name="segment_ids")

            config = modeling.BertConfig.from_json_file(os.path.join(self.bert_model_dir, 'bert_config.json'))
            model = modeling.BertModel(config=config, is_training=False,
                                       input_ids=input_ids, input_mask=input_mask, token_type_ids=segment_ids,
                                       use_one_hot_embeddings=False)

            pooled_output = model.get_pooled_output()

            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, os.path.join(self.bert_model_dir, 'bert_model.ckpt'))

        return graph, sess

    def tokenize(self, text):
        tokens = self.tokenizer.tokenize(text)
        return ["[CLS]"] + tokens + ["[SEP]"]

    def encode(self, texts):
        all_input_ids, all_input_mask, all_segment_ids = [], [], []

        for text in texts:
            tokens = self.tokenize(text)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)

            while len(input_ids) < self.max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            all_input_ids.append(input_ids)
            all_input_mask.append(input_mask)
            all_segment_ids.append(segment_ids)

        with self.graph.as_default():
            encoded = self.sess.run(self.graph.get_tensor_by_name("bert/encoder/pooler/dense/BiasAdd:0"),
                                    feed_dict={
                                        "input_ids:0": np.array(all_input_ids),
                                        "input_mask:0": np.array(all_input_mask),
                                        "segment_ids:0": np.array(all_segment_ids)
                                    })

        return encoded
