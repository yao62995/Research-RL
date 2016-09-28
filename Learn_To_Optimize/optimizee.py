#!/usr/bin/python
#  -*- coding: utf-8 -*-
# author: yao_62995 <yao_62995@163.com>

import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from rnn import CustomLSTMCell

tf.app.flags.DEFINE_string('data_dir', './data/', 'Directory for storing data')
tf.app.flags.DEFINE_string('model_dir', './models/', 'Directory for storing models')
tf.app.flags.DEFINE_integer('hidden_units', 1000, 'number of hidden units')
tf.app.flags.DEFINE_integer('gpu', 3, 'gpu card id')
FLAGS = tf.app.flags.FLAGS

initial_weight = lambda: tf.truncated_normal_initializer(stddev=0.1)
initial_bias = lambda: tf.constant_initializer(0.0)


class Optimizee(object):
    """ref: Learning to learn by gradient descent by gradient descent. [2016]"""

    def __init__(self, input_dim, n_classes, session):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.sess = session
        with tf.device("/gpu:%d" % FLAGS.gpu):
            # place holder
            self.input = tf.placeholder(tf.float32, shape=[None, self.input_dim], name="input")
            self.label = tf.placeholder(tf.float32, shape=[None, self.n_classes], name="label")
            # init all variables
            w1 = tf.get_variable("w1", shape=[self.input_dim, 128], initializer=initial_weight(), trainable=False)
            b1 = tf.get_variable("b1", shape=[128], initializer=initial_bias(), trainable=False)
            w2 = tf.get_variable("w2", shape=[128, n_classes], initializer=initial_weight(), trainable=False)
            b2 = tf.get_variable("b2", shape=[n_classes], initializer=initial_bias(), trainable=False)
            self.theta = [w1, b1, w2, b2]
            # network
            h1 = tf.nn.relu(tf.matmul(self.input, w1) + b1)
            h2 = tf.matmul(h1, w2) + b2
            logit = h2
            # loss value
            self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logit, self.label), name="loss")
            # gradients of all variables
            self.grads = tf.gradients(self.loss, self.theta, grad_ys=None)
            # flatten gradients
            self.flat_grads = self.flat_tensors(self.grads)
            self.flat_grads_size = np.sum([np.prod(v.get_shape().as_list()) for v in self.theta])
            # update variables
            self.grads_input = tf.placeholder(tf.float32, shape=[self.flat_grads_size], name="grads_input")
            self.update_theta_ops = self.unfold_tensors(self.grads_input)

    def compute_grads(self, X, y):
        flat_grads, loss = self.sess.run([self.flat_grads, self.loss], feed_dict={self.input: X, self.label: y})
        return flat_grads, loss

    def apply_grads(self, grads):
        self.sess.run(self.update_theta_ops, feed_dict={self.grads_input: grads})

    def flat_tensors(self, grads):
        """flat all grads to 1-D tensor"""
        flat_theta = []
        for grad in grads:
            flat_theta.append(tf.reshape(grad, [-1]))
        return tf.concat(0, flat_theta)

    def unfold_tensors(self, flat_grads):
        """unfold grads for update variables"""
        step = 0
        assign_op = []
        with tf.op_scope(self.theta, "update_grads"):
            for th in self.theta:
                th_shape = th.get_shape().as_list()
                th_shape_len = np.prod(th_shape)
                grad = tf.reshape(flat_grads[step: (step + th_shape_len)], shape=th_shape)
                assign_op.append(tf.assign_add(th.ref(), grad))
                step += th_shape_len
            return tf.group(*assign_op)


class Optimizer(object):
    """ref: Learning to learn by gradient descent by gradient descent. [2016]"""

    def __init__(self, optimizee, session):
        self.optimizee = optimizee
        self.sess = session
        self.input_units = FLAGS.hidden_units * 2
        self.input_split_num = int(np.ceil(self.optimizee.flat_grads_size * 2 / float(self.input_units)))
        self.pad_size = self.input_split_num * self.input_units - self.optimizee.flat_grads_size * 2
        self.input_size = self.input_split_num * self.input_units
        with tf.device("/gpu:%d" % FLAGS.gpu):
            # place holder
            self.input = tf.placeholder(tf.float32, shape=[1, self.input_size], name="input_grads")
            self.loss_scale = tf.placeholder(tf.float32, shape=[1], name="loss_scale")
            self.lstm_state = tf.placeholder(tf.float32, shape=[self.input_split_num, 2 * FLAGS.hidden_units])
            # add sequence dimension
            input_split = tf.split(1, self.input_split_num, self.input, name="split_input")
            lstm_state_split = tf.split(0, self.input_split_num, self.lstm_state, name="split_lstm_state")
            # init lstm variable
            lstm_weight = tf.get_variable("lstm_w",
                                          shape=[self.input_units + FLAGS.hidden_units, 4 * FLAGS.hidden_units],
                                          initializer=initial_weight())
            lstm_bias = tf.get_variable("lstm_b", shape=[4 * FLAGS.hidden_units], initializer=initial_bias())
            # construct rnn layer for all split
            lstm_out_list, state_out_list = [], []
            for idx in xrange(self.input_split_num):
                lstm_cell = CustomLSTMCell(FLAGS.hidden_units, matrix=lstm_weight, bias=lstm_bias)
                lstm_out, state_out = tf.nn.rnn(lstm_cell, [input_split[idx]],
                                                initial_state=lstm_state_split[idx], dtype=tf.float32)
                lstm_out_list.append(lstm_out[0])
                state_out_list.append(state_out)
            # concat output
            self.logits = tf.concat(1, lstm_out_list, name="concat_logit")
            self.lstm_state_out = tf.concat(0, state_out_list, name="concat_state")
            # loss values
            self.loss = tf.reduce_sum(self.logits) * self.loss_scale
            # set optimizer
            self.train_opt = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)
        # init all variables
        self.saver = tf.train.Saver(max_to_keep=3, var_list=[lstm_weight, lstm_bias] + self.optimizee.theta)
        self.sess.run(tf.initialize_all_variables())
        restore_model(self.sess, FLAGS.model_dir, self.saver)
        # reset lstm state
        self.lstm_state_init = self.reset_lstm_state()

    def reset_lstm_state(self):
        return np.zeros((self.input_split_num, 2 * FLAGS.hidden_units), dtype=np.float32)

    def preprocess_input(self, flat_grads, p=10):
        """preprocess  flat gradient"""
        new_flat_grads = np.empty(len(flat_grads) * 2, dtype=np.float32)
        threshold = np.exp(-p)
        threshold_2 = np.exp(p)
        for i, grad in enumerate(flat_grads):
            abs_grad = abs(grad)
            if abs_grad >= threshold:
                new_flat_grads[i * 2] = np.log(abs_grad) / p
                new_flat_grads[i * 2 + 1] = np.sign(grad)
            else:
                new_flat_grads[i * 2] = -1
                new_flat_grads[i * 2 + 1] = threshold_2
        return self.input_pad(new_flat_grads)

    def fit(self, X, y):
        # compute grads of optimizee
        flat_grads, opt_loss = self.optimizee.compute_grads(X, y)
        # preprocess grads
        flat_grads = self.preprocess_input(flat_grads)
        # train optimizer
        _, flat_grads_out, self.lstm_state_init = self.sess.run(
            [self.train_opt, self.logits, self.lstm_state_out],
            feed_dict={self.input: [flat_grads], self.loss_scale: [opt_loss], self.lstm_state: self.lstm_state_init}
        )
        # update variables of optimizee
        self.optimizee.apply_grads(flat_grads_out[0][:self.optimizee.flat_grads_size])
        return opt_loss

    def input_pad(self, input_data):
        """padding input data"""
        return np.lib.pad(input_data, (0, self.pad_size), 'constant', constant_values=0)


def restore_model(sess, model_dir, saver, model_file=None):
    if model_file is not None:
        model_file_path = "%s/%s" % (model_dir, model_file)
        saver.restore(sess, model_file_path)
        print("Successfully loaded:", model_file_path)
    else:
        checkpoint = tf.train.get_checkpoint_state(model_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")


def save_model(sess, model_dir, saver, prefix, global_step=None):
    checkpoint_filename = saver.save(sess, model_dir + "/" + prefix, global_step=global_step)
    return checkpoint_filename


def main(args):
    if not os.path.isdir(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)
    if not os.path.isdir(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)
    input_dim = 28 * 28
    n_classes = 10
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    optimizee = Optimizee(input_dim, n_classes, sess)
    optimizer = Optimizer(optimizee, sess)
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        loss = optimizer.fit(batch_xs, batch_ys)
        print "iter:", i, ", loss:", loss


if __name__ == "__main__":
    tf.app.run()
