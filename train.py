from __future__ import division
from __future__ import print_function
from scipy.sparse import csr_matrix
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from utils import *
from metrics import *
from models import GCN_Align, LSTM



# Set random seed
seed = 12306
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('lang', 'zh_en', 'Dataset string.')  # 'zh_en', 'ja_en', 'fr_en'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1800, 'Number of epochs to train.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('gamma', 3.0, 'Hyper-parameter for margin based loss.')
flags.DEFINE_integer('k', 5, 'Number of negative samples for each positive seed.')
flags.DEFINE_float('beta', 0.9, 'Weight for structure embeddings.')
flags.DEFINE_integer('se_dim', 100, 'Dimension for SE.')
flags.DEFINE_integer('ae_dim', 50, 'Dimension for AE.')
flags.DEFINE_integer('seed', 5, 'Proportion of seeds, 3 means 30%')
flags.DEFINE_integer('unit', 40, 'none')
assert FLAGS.epochs >= FLAGS.unit and FLAGS.epochs % FLAGS.unit == 0
# Load data
adj, ae_input, attr_embeddings, train, test, date_info, timeSource = load_data(FLAGS.lang)


#set time
LSTM.timeSource = timeSource
LSTM.date_info = date_info
LSTM.time_size = 7
train = np.array(train)

# Some preprocessing
support = [preprocess_adj(adj)]
num_supports = 1
model_func = GCN_Align
k = FLAGS.k
e = ae_input[2][0]

# Define placeholders
ph_ae = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32), #tf.placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder_with_default(0, shape=())
}
ph_se = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder_with_default(0, shape=())
}

# Create model
model_ae = model_func(ph_ae, input_dim=ae_input[2][1], output_dim=FLAGS.ae_dim, ILL=train, sparse_inputs=True, featureless=False, logging=True)
model_se = model_func(ph_se, input_dim=e, output_dim=FLAGS.se_dim, ILL=train, sparse_inputs=False, featureless=True, logging=True)
# Initialize session
sess = tf.Session()

# Init variables

cost_val = []
t = len(train)
seed_left = train[:, 0].tolist() # 获取第一列的值
seed_right = train[:,1].tolist()  # 获取第二列的值

neg_left = np.array([x for x in seed_left for i in range(k)])   # 1* 37500
neg2_right = np.array([x for x in seed_right for i in range(k)])

lstm_left = np.array([x for x in timeSource for i in range(int(len(timeSource[x]) * 0.08) + 1)])
lstm_right = LSTM.get_right(lstm_left, seed_left, seed_right)
hold = [tf.placeholder(tf.int32, (len(lstm_left),), name='lstm_left'),
        tf.placeholder(tf.int32, (len(lstm_left),), name='lstm_neg'),
        tf.placeholder(tf.int32, (len(lstm_left) * LSTM.time_size,), name='tem_left'),
        tf.placeholder(tf.int32, (len(lstm_left) * LSTM.time_size,), name='tem_neg')]

loss_ae = model_ae.lstm.lose(model_ae.outputs, lstm_left, lstm_right, FLAGS.gamma)
op_ae = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss_ae)

loss_se = model_se.lstm.lose(model_se.outputs, lstm_left, lstm_right, FLAGS.gamma)
op_se = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss_se)



del seed_left, seed_right
sess.run(tf.global_variables_initializer())
# Train model

for epoch in range(FLAGS.epochs // FLAGS.unit):
    for ri in range(3):
        neg2_left = np.random.choice(e, t * k)
        neg_right = np.random.choice(e, t * k)
        feed_dict_ae = construct_feed_dict(ae_input, support, ph_ae)
        feed_dict_ae.update({ph_ae['dropout']: FLAGS.dropout})
        feed_dict_ae.update({'neg_left:0': neg_left, 'neg_right:0': neg_right, 'neg2_left:0': neg2_left, 'neg2_right:0': neg2_right})

        feed_dict_se = construct_feed_dict(1.0, support, ph_se)
        feed_dict_se.update({ph_se['dropout']: FLAGS.dropout})
        feed_dict_se.update({'neg_left:0': neg_left, 'neg_right:0': neg_right, 'neg2_left:0': neg2_left, 'neg2_right:0': neg2_right})
        for rg in range(10):
            outs_ae = sess.run([model_ae.opt_op, model_ae.loss], feed_dict=feed_dict_ae)
            outs_se = sess.run([model_se.opt_op, model_se.loss], feed_dict=feed_dict_se)
            cost_val.append((outs_ae[1], outs_se[1]))

            # Print results
            print("Current_Epoch:", '%04d' % (epoch * FLAGS.unit + 10 * ri + rg + 1), "Attribute_Embedding_train_loss=", "{:.5f}".format(outs_ae[1]), "Structure_Embedding_train_loss=",
                  "{:.5f}".format(outs_se[1]))

    del neg2_left, neg_right
    lstm_neg, tem_left, tem_neg = LSTM.get_neg(lstm_left)
    # Construct feed dictionary
    feed_dict_ae = construct_feed_dict(ae_input, support, ph_ae)
    feed_dict_ae.update({ph_ae['dropout']: FLAGS.dropout})
    feed_dict_ae.update({'lstm_left:0': lstm_left, 'lstm_neg:0': lstm_neg, 'tem_left:0': tem_left, 'tem_neg:0': tem_neg})

    feed_dict_se = construct_feed_dict(1.0, support, ph_se)
    feed_dict_se.update({ph_se['dropout']: FLAGS.dropout})
    feed_dict_se.update({'lstm_left:0': lstm_left, 'lstm_neg:0': lstm_neg, 'tem_left:0': tem_left, 'tem_neg:0': tem_neg})
    for rg in range(10):
        # Training step
        outs_ae = sess.run([op_ae, loss_ae], feed_dict=feed_dict_ae)
        outs_se = sess.run([op_se, loss_se], feed_dict=feed_dict_se)
        cost_val.append((outs_ae[1], outs_se[1]))

        # Print results
        print("Current_Epoch:", '%04d' % (epoch * FLAGS.unit + 30 + rg + 1), "Attribute_Embedding_train_loss=", "{:.5f}".format(outs_ae[1] * 100), "Structure_Embedding_train_loss=", "{:.5f}".format(outs_se[1] * 100), ' another loss, * 100')

print("Optimization Finished!")

# Testing
feed_dict_ae = construct_feed_dict(ae_input, support, ph_ae)
feed_dict_se = construct_feed_dict(1.0, support, ph_se)
vec_ae = sess.run(model_ae.outputs, feed_dict=feed_dict_ae)
vec_se = sess.run(model_se.outputs, feed_dict=feed_dict_se)
# get_hits(vec_ae, test)
# print("SE")
get_hits(vec_se, test)
# print("SE+AE")
get_combine_hits(vec_se, vec_ae, FLAGS.beta, test)
