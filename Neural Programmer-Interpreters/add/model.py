#!/usr/bin/python
#  -*- coding: utf-8 -*-
# author: yao_62995 <yao_62995@163.com>

import numpy as np
import tensorflow as tf
from tensorflow.models.rnn.rnn_cell import LSTMCell

from env import Env, ProgramManager

initial_weight = lambda: tf.truncated_normal_initializer(stddev=0.1)
initial_bias = lambda: tf.constant_initializer(0.0)


class AdditionModel(object):
    def __init__(self, model_dir, gpu=0, batch_size=1, num_units=256):
        env_size = Env.FIELD_NUM * Env.FIELD_DEPTH
        args_size = Env.ARG_MAX_NUM * Env.ARG_DEPTH
        prog_size = ProgramManager.PG_NUM
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        with tf.device("/gpu:%d" % gpu):
            # global step
            self.global_step = tf.get_variable(tf.int32, shape=[None],
                                               initializer=tf.constant_initializer(0.0), trainable=False)
            # input place_holder
            self.input_env = tf.placeholder(tf.int16, shape=[None, env_size])
            self.input_arg = tf.placeholder(tf.int16, shape=[None, args_size])
            self.input_prog = tf.placeholder(tf.int16, shape=[None, prog_size])
            self.lstm_state = tf.placeholder(tf.float32, shape=[batch_size, 2 * num_units])
            # output place_holder
            self.out_end = tf.placeholder(tf.float32, shape=[None, 1])
            self.out_prog = tf.placeholder(tf.float32, shape=[None, prog_size])
            self.out_args = tf.placeholder(tf.float32, shape=[None, Env.ARG_MAX_NUM])
            # init variables
            w_enc = tf.get_variable("w_enc", shape=[env_size + args_size, 128], initializer=initial_weight)
            b_enc = tf.get_variable("b_enc", shape=[128], initializer=initial_bias)
            w_end = tf.get_variable("w_end", shape=[num_units, 1], initializer=initial_weight)
            b_end = tf.get_variable("b_end", shape=[1], initializer=initial_bias)
            w_prog = tf.get_variable("w_enc", shape=[num_units, prog_size], initializer=initial_weight)
            b_prog = tf.get_variable("b_enc", shape=[prog_size], initializer=initial_bias)
            w_args = [tf.get_variable("w_arg", shape=[num_units, Env.ARG_DEPTH], initializer=initial_weight)
                      for _ in xrange(Env.ARG_MAX_NUM)]
            b_args = [tf.get_variable("b_arg", shape=[Env.ARG_DEPTH], initializer=initial_bias)
                      for _ in xrange(Env.ARG_MAX_NUM)]
            # networks
            h_concat_1 = tf.concat(1, [self.input_env, self.input_arg], name="merge_env_arg")
            f_enc = tf.nn.relu(tf.matmul(h_concat_1, w_enc) + b_enc, name="f_enc")
            f_enc_reshape = tf.reshape(f_enc, shape=[-1, 1, 128])
            h_concat_2 = tf.concat(2, [f_enc_reshape, ], name="merge_state_prog")
            # LSTM layers
            lstm_cell = LSTMCell(256)
            h_output, self.state_out = tf.nn.rnn(lstm_cell, h_concat_2, initial_state=self.lstm_state)
            f_lstm = tf.nn.relu(h_output[-1], name="f_lstm")
            # logits out
            f_end_logits = tf.matmul(f_lstm, w_end) + b_end
            self.f_end = tf.nn.sigmoid(f_end_logits, name="f_end")
            f_prog_logits = tf.matmul(f_lstm, w_prog) + b_prog
            self.f_prog = tf.nn.softmax(f_prog_logits, name="f_prog")
            f_args_logits, self.f_args = [], []
            for arg_i in xrange(Env.ARG_MAX_NUM):
                f_args_logits.append(tf.matmul(f_lstm, w_args[arg_i]) + b_args[arg_i])
                self.f_args.append(tf.nn.softmax(f_args_logits[-1], name="f_arg_%d" % arg_i))
            # loss (objective function)
            l2_loss = tf.add_n(map(lambda arg: tf.nn.l2_loss(arg), [w_enc, w_end, w_prog] + w_args), name="l2_loss")
            f_end_loss = tf.nn.sigmoid_cross_entropy_with_logits(f_end_logits, self.out_end, name="f_end_loss")
            f_prog_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(f_prog_logits, self.out_prog,
                                                                         name="f_prog_loss")
            _out_args = tf.split(1, Env.ARG_MAX_NUM, self.out_args)
            f_args_loss = [tf.nn.sparse_softmax_cross_entropy_with_logits(f_args_logits[i], _out_args[i],
                                                                          name="f_args_loss_%d" % i)
                           for i in xrange(Env.ARG_MAX_NUM)]
            total_loss = f_prog_loss + f_end_loss + tf.add_n(f_args_loss) + l2_loss
            # optimizer
            self.train_opt = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(total_loss,
                                                                                 global_step=self.global_step)
            # summary
            summaries = [
                tf.scalar_summary("out/l2_loss", l2_loss),
                tf.scalar_summary("out/f_end_loss", f_end_loss),
                tf.scalar_summary("out/f_prog_loss", f_prog_loss),
                tf.scalar_summary("out/f_args_loss", f_args_loss),
                tf.scalar_summary("out/total_loss", total_loss),
            ]
            self.summary_op = tf.merge_summary(summaries)
            self.summary_writer = tf.train.SummaryWriter(model_dir)
        # init saver
        self.saver = tf.train.Saver()
        self.sess.run(tf.initialize_all_variables())
        restore_model(self.sess, model_dir, self.saver)
        # reset lstm state
        self.reset_lstm_state = lambda: np.zeros((batch_size, 2 * num_units), dtype=np.float32)
        self.lstm_state_init = self.reset_lstm_state()

    def fit(self, input_env, input_arg, input_prog, out_end, out_prog, out_args, summary=True):
        """
        :param input_env: numpy array shape=(batch, FIELD_NUM * FIELD_DEPTH)
        :param input_arg: numpy array shape=(batch, ARG_MAX_NUM * ARG_DEPTH)
        :param input_prog: numpy array shape=(batch, Program_NUM)
        :param out_end:  numpy array shape=(batch, 1)
        :param out_prog: numpy array shape=(batch, Program_NUM)
        :param out_args: numpy array shape=(batch, ARG_MAX_NUM)
        :param summary:  bool, whether to write summary
        :return:
        """
        fetches = [self.train_opt, self.state_out]
        if summary:
            fetches.append(self.summary_op)
        ret = self.sess.run(fetches, feed_dict={self.input_env: input_env, self.input_prog: input_prog,
                                                self.input_arg: input_arg, self.out_end: out_end,
                                                self.out_prog: out_prog, self.out_args: out_args,
                                                self.lstm_state: self.lstm_state_init})
        self.lstm_state_init = ret[1]
        if summary:
            self.summary_writer.add_summary(ret[-1])

    def predict(self, input_env, input_arg, input_prog):
        ret = self.sess.run([self.f_end, self.f_prog] + self.f_args,
                            feed_dict={self.input_env: input_env, self.input_arg: input_arg,
                                       self.input_prog: input_prog, self.lstm_state: self.lstm_state_init})
        prob_end = ret[0]
        prog_id = np.argmax(ret[1], axis=1)
        args = [np.argmax(ret[(2 + i)], axis=1) for i in xrange(Env.ARG_MAX_NUM)]
        return prob_end, prog_id, args


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
