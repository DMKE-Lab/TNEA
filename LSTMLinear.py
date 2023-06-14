import math
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class LSTMModel():
    def __init__(self, in_dim, n_layer):
        super(LSTMModel, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = in_dim
        # self.lstm = nn.LSTM(in_dim, self.hidden_dim, n_layer, batch_first=True)
        self.lstm = LSTMLinear(in_dim, self.hidden_dim)

    def __call__(self, x):
        out, h = self.lstm(x)
        return h[0]

class linear():
    def __init__(self, input_size, output_size):
        stdv = 1. / math.sqrt(5)
        self.Weights = tf.Variable(tf.random_uniform([input_size, output_size], -stdv, stdv))
        stdv = 1./ math.sqrt(input_size)
        self.biases = tf.Variable(tf.random_uniform((output_size,), -stdv, stdv))

    def __call__(self, x):
        return tf.matmul(x, self.Weights) + self.biases


class LSTMCell():

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.i2h = linear(input_size, 4 * hidden_size)
        self.h2h = linear(input_size, 4 * hidden_size)
        # self.linear_acti = nn.Linear(hidden_size, hidden_size)

    def __call__(self, x, hidden):
        if hidden is None:
            hidden = self._init_hidden(x, self.hidden_size)
        h, c = hidden
        h = tf.reshape(h, [h.shape[1], -1])
        c = tf.reshape(c, [c.shape[1], -1])
        preact = self.i2h(x) + self.h2h(x)

        # activations
        gates = tf.nn.sigmoid(preact[:, :3 * self.hidden_size])

        # g_t = preact[:, 3 * self.hidden_size:].tanh()
        g_t = preact[:, 3 * self.hidden_size:]
        i_t = gates[:, :self.hidden_size]
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size:]

        c_t = tf.multiply(c, f_t) + tf.multiply(i_t, g_t)
        h_t = tf.multiply(o_t, c_t)

        h_t = tf.reshape(h_t, [1, h_t.shape[0], -1])
        c_t = tf.reshape(c_t, [1, c_t.shape[0], -1])
        return h_t, c_t

    @staticmethod
    def _init_hidden(input_, hidden_size):
        h = tf.zeros_like(tf.reshape(input_, [1, input_.shape[0], -1]))
        c = tf.zeros_like(tf.reshape(input_, [1, input_.shape[0], -1]))
        return h, c


class LSTMLinear():

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMLinear, self).__init__()
        self.lstm_cell = LSTMCell(input_size, hidden_size, bias)
        self.batch_first = True

    def __call__(self, input_, hidden=None):

        if self.batch_first:
            input_ = tf.transpose(input_, perm=[1, 0, 2])
        outputs = []
        steps = range(input_.shape[0])
        for i in steps:
            hidden = self.lstm_cell(input_[i], hidden)
            if isinstance(hidden, tuple):
                outputs.append(hidden[0])
            else:
                outputs.append(hidden)
        outputs = tf.stack(outputs, 0)
        if self.batch_first:
            outputs = tf.transpose(outputs, perm=[1, 0, 2, 3])

        return outputs, hidden