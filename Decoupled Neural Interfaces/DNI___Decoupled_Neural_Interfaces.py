#!/usr/bin/python
#  -*- coding: utf-8 -*-
# author: yao_62995 <yao_62995@163.com>

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.app.flags.DEFINE_string('data_dir', './data/', 'Directory for storing data')
tf.app.flags.DEFINE_string('model_dir', './models/', 'Directory for storing models')
FLAGS = tf.app.flags.FLAGS

initial_weight = lambda: tf.truncated_normal_initializer(stddev=0.1)
initial_bias = lambda: tf.constant_initializer(0.0)


class DNILayer(object):
    """ref: decoupled neural interfaces using synthetic gradients"""

    def __init__(self, scope, _input, W_shape, activate=None):
        self.scope = scope
        self.input = _input
        self.W_shape = W_shape
        with tf.variable_scope(scope):
            # hidden layer variables
            self.h_weight = tf.get_variable(name="h_w", shape=W_shape, initializer=initial_weight())
            self.h_bias = tf.get_variable(name="h_b", shape=W_shape[-1], initializer=initial_bias())
            # M layer variables
            self.m_weight_1 = tf.get_variable(name="m_w_1", shape=[W_shape[-1], 128],
                                              initializer=initial_weight())
            self.m_bias_1 = tf.get_variable(name="m_b_1", shape=[128], initializer=initial_bias())
            self.m_weight_2 = tf.get_variable(name="m_w_2", shape=[128, W_shape[-1]], initializer=initial_weight())
            self.m_bias_2 = tf.get_variable(name="m_b_2", shape=[W_shape[-1]], initializer=initial_bias())
            self.h_vars = [self.h_weight, self.h_bias]
            self.m_vars = [self.m_weight_1, self.m_bias_1, self.m_weight_2, self.m_bias_2]
        with tf.op_scope(self.h_vars, "%s_h" % scope):  # op scope of hidden layers
            self.h_layer = tf.matmul(self.input, self.h_weight) + self.h_bias
            if activate is not None:
                self.h_layer = tf.nn.relu(self.h_layer)
        with tf.op_scope(self.m_vars, "%s_M" % scope):  # op scope of M layers
            self.m_layer_1 = tf.nn.relu(tf.matmul(self.h_layer, self.m_weight_1) + self.m_bias_1)
            self.m_layer = tf.matmul(self.m_layer_1, self.m_weight_2) + self.m_bias_2
        with tf.op_scope([], "%s_grad" % scope):  # grads
            self.h_layer_grad = self.m_layer
            # grad of hidden layer parameters
            self.grads_h_var = tf.gradients(self.h_layer, self.h_vars, grad_ys=self.h_layer_grad)
            # self.grads_h_input = tf.gradients(self.h_layer, self.input, grad_ys=None)
            self.grads_m_var = None

    def output(self):
        return self.h_layer

    def M_output(self):
        return self.m_layer

    def set_M_grads(self, optimizer, grads_m):
        self.grads_m_var = tf.gradients(self.m_layer, self.m_vars, grad_ys=grads_m)
        opt_h = optimizer.apply_gradients(zip(self.grads_h_var, self.h_vars))
        opt_m = optimizer.apply_gradients(zip(self.grads_m_var, self.m_vars))
        return opt_h, opt_m


class DNIModel(object):
    """ref: decoupled neural interfaces using synthetic gradients"""

    def __init__(self, input_dim, n_classes):
        self.input_dim = input_dim
        self.n_classes = n_classes
        with tf.device("/gpu:2"):
            # place holder for IN/OUT
            self.X = tf.placeholder(tf.float32, shape=[None, self.input_dim])
            self.Y = tf.placeholder(tf.float32, shape=[None, self.n_classes])
            # dni layers
            dni_1 = DNILayer("dni_1", self.X, [self.input_dim, 128], activate="relu")
            dni_2 = DNILayer("dni_2", dni_1.output(), [128, 64], activate="relu")
            dni_3 = DNILayer("dni_3", dni_2.output(), [64, 10])
            # loss function
            self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(dni_3.output(), self.Y, name="loss"))
            # optimizer
            self.opt = tf.train.AdamOptimizer(learning_rate=1e-3)
            # set grads for DNI_layer_3
            grads_loss_dni3 = tf.gradients(self.loss, dni_3.output())
            opt_h3, opt_m3 = dni_3.set_M_grads(self.opt, grads_loss_dni3[0])
            # set grads for DNI_layer_2
            grads_dni3_dni2 = tf.gradients(dni_3.output(), dni_3.input, grad_ys=dni_3.M_output())
            opt_h2, opt_m2 = dni_2.set_M_grads(self.opt, grads_dni3_dni2[0])
            # set grads for DNI_layer_1
            grads_dni2_dni1 = tf.gradients(dni_2.output(), dni_2.input, grad_ys=dni_2.M_output())
            opt_h1, opt_m1 = dni_1.set_M_grads(self.opt, grads_dni2_dni1[0])
            ops = [opt_h3, opt_m3, opt_h2, opt_m2, opt_h1, opt_m1]
            self.train_op = tf.group(*ops)
        # train
        h_var_list = dni_1.h_vars + dni_2.h_vars + dni_3.h_vars
        self.saver = tf.train.Saver(max_to_keep=3, var_list=h_var_list)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        self.sess.run(tf.initialize_all_variables())
        restore_model(self.sess, FLAGS.model_dir, self.saver)

    def fit(self, X, y):
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.X: X, self.Y: y})
        return loss


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
    model = DNIModel(input_dim, n_classes)
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    for i in range(5000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        loss = model.fit(batch_xs, batch_ys)
        if i % 10 == 0:
            print "iter:", i, ", loss:", loss


if __name__ == "__main__":
    tf.app.run()
