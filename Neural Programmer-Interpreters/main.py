#!/usr/bin/python
#  -*- coding: utf-8 -*-
# author: yao_62995 <yao_62995@163.com>

import tensorflow as tf

tf.app.flags.DEFINE_string('name', 'add', 'program name, choice in [add, sort]')
tf.app.flags.DEFINE_string('action', 'create_corpus', 'handle type, choice in [create_corpus, train, test]')
tf.app.flags.DEFINE_integer('gpu', 3, 'gpu card id')
FLAGS = tf.app.flags.FLAGS

def main(args):
    pass


if __name__ == "__main__":
    tf.app.run()
