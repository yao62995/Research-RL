#!/usr/bin/python
#  -*- coding: utf-8 -*-
# author: yao_62995 <yao_62995@163.com>

import time
import cPickle
import numpy as np
import tensorflow as tf

from env import AdditionEnv, ProgramManager

tf.app.flags.DEFINE_string('data_dir', './data/', 'Directory for storing training data')
tf.app.flags.DEFINE_string('model_dir', './models/', 'Directory for storing models')
FLAGS = tf.app.flags.FLAGS


class Addition(object):
    def __init__(self, max_digits=10):
        self.max_digits = max_digits
        self.env = AdditionEnv(max_digits=max_digits)
        self.prog_mgr = ProgramManager

    def create_corpus(self, sample_num=1e4):
        file_name = "%s/sample_%d_%d.pkl" % (FLAGS.data_dir, int(time.time()), sample_num)
        sample_list = []

        for idx in xrange(sample_num):
            input1 = self._create_input()
            input2 = self._create_input()
            # reset environment
            self.env.reset((input1, input2))
            while

    def _create_input(self):
        _input = np.random.randint(0, 10, size=self.max_digits)
        _input[0] = 0
        return _input

    def train(self):
        pass

    def test(self):
        pass
