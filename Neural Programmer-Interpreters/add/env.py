#!/usr/bin/python
#  -*- coding: utf-8 -*-
# author: yao_62995 <yao_62995@163.com>

import numpy as np

"""
Program           Descriptions                                   Calls
ADD               Perform multi-digit addition                   ADD1, LSHIFT
ADD1              Perform single-digit addition                  ACT, CARRY
CARRY             Mark a 1 in the carry row one unit left        ACT
LSHIFT            Shift a specified pointer one step left        ACT
RSHIFT            Shift a specified pointer one step right       ACT
ACT               Move a pointer or write to the scratch pad     -
"""


class Env(object):
    FIELD_NUM = 4  # field num, [Input1, Input2, Carry, Output]
    FIELDS = {"INPUT1": 0, "INPUT2": 1, "CARRY": 2, "OUTPUT": 3}
    FIELD_DEPTH = 10  # field depth, digit (0~9)
    ARG_MAX_NUM = 3  # (Single_Digit_Input1, Single_Digit_Input2, Carry)
    ARG_DEPTH = 10  # digit (0~9)

    ACT_WRITE = 0  # action type of writing
    ACT_MOVE_PTR = 1  # action type of moving pointer
    MOVE_PTR_LEFT = 0  # move pointer left
    MOVE_PTR_RIGHT = 1  # move pointer right

    def __init__(self, max_digits=10):
        self.max_digits = max_digits
        self.fields = np.empty((self.FIELD_NUM, max_digits), dtype=np.int)
        self.pointers = np.empty(self.FIELD_NUM, dtype=np.int)
        self.reset()

    def reset(self, inputs=None):
        # reset all fields to zeros
        self.fields.fill(0)
        if inputs:
            self.fields[0, :] = inputs[0]
            self.fields[1, :] = inputs[1]
        # move fields to rightmost place
        self.pointers.fill(self.max_digits - 1)

    def step(self, pg, arg):
        """
            step forward for environment
        :param pg:  program id
        :param arg: argument values, shape=(MAX_ARG_NUM, ARG_DEPTH)
        :return:
        """
        pass

    @staticmethod
    def decode_env(env_obs):
        """
        :param env_obs: environment observation, shape=(FIELD_NUM, FIELD_DEPTH)
        :return: tuple of (in1, in2, carry, output)
        """
        return np.argmax(env_obs, axis=1)

    @staticmethod
    def encode_env(env_obs):
        """
        :param env_obs: environment observation, shape=(FIELD_NUM, FIELD_DEPTH)
        :return: a flatten list
        """
        return env_obs.flatten()

    @staticmethod
    def decode_arg(arg_values):
        """
        decode argument from a flatten list to one-hot matrix with shape(MAX_ARG_NUM, ARG_DEPTH)
        :param arg_values: argument values, a flatten list
        :return: tuple of (arg1, arg2, arg3)
        """
        arg_values = np.reshape(arg_values, newshape=(Env.ARG_MAX_NUM, Env.ARG_DEPTH)).astype(int)
        return np.argmax(arg_values, axis=1)

    @staticmethod
    def encode_arg(arg_list=None, arg_values=None):
        """
        encode argument from one-hot matrix to a flatten list
        :param arg_list: argument list
        :param values: argument values
        :return: transformed argument
        """
        if arg_values:
            arg_values = np.reshape(arg_values, newshape=(Env.ARG_MAX_NUM, Env.ARG_DEPTH)).astype(int)
        else:
            arg_values = np.zeros((Env.ARG_MAX_NUM, Env.ARG_DEPTH), dtype=int)
        if arg_list:
            for i, v in enumerate(arg_list):
                arg_values[i, v] = 1
        return arg_values

    @staticmethod
    def empty_arg():
        return np.zeros((Env.ARG_MAX_NUM, Env.ARG_DEPTH), dtype=int)

    @staticmethod
    def decode_prog(prog):
        """
        decode one-hot program_id to single program_id
        :param prog:
        :return: program id, an integer value
        """
        return np.argmax(prog)

    @staticmethod
    def encode_prog(prog, prog_num=6):
        """
        encode single program id to one-hot program id
        :param prog:
        :return:
        """
        one_hot_prog = np.zeros(prog_num, dtype=int)
        one_hot_prog[prog] = 1
        return one_hot_prog


AdditionEnv = Env


class ProgramManager(object):
    PG_CONTINUE = 0
    PG_RETURN = 1
    PG_NUM = 6

    def __init__(self):
        # tuple of (prog_name, prog_body)
        self.program_set = [
            ("ADD", self.pg_add),
            ("ADD1", self.pg_add1),
            ("CARRY", self.pg_carry),
            ("LSHIFT", self.pg_lshift),
            ("RSHIFT", self.pg_rshift),
            ("ACT", self.pg_act),
        ]
        # program map, {key=program_name, value=(program_id, program_body)}
        self.pg = dict((item[0], (idx, item[1])) for idx, item in enumerate(self.program_set))

    def pg_add(self, env, args):
        """perform multi-digit addition"""
        in1, in2, carry, _ = Env.decode_env(env.fields)
        if env.pointers[0] < 0:
            return None
        if (in1 + in2 + carry) == 0:  # skip ops
            return [(self.pg["LSHIFT"], Env.empty_arg())]
        else:  # enter into sub-program
            sub_prog = [(self.pg["ADD1"], Env.empty_arg()),
                        (self.pg["LSHIFT"], Env.empty_arg())]
            return sub_prog

    def pg_add1(self, env, args):
        """perform single-digit addition"""
        in1, in2, carry, output = Env.decode_env(env.fields)
        # add [in1, in2, carry] together
        result = sum(in1 + in2 + carry)
        sub_prog = [(self.pg["ACT"], Env.encode_arg(arg_list=[Env.FIELDS["OUTPUT"], result % 10, Env.ACT_WRITE]))]
        if result >= Env.FIELD_DEPTH:  # carry ops
            sub_prog.append((self.pg["CARRY"], Env.empty_arg()))
        return sub_prog

    def pg_act(self, env, args):
        """action of write into result"""
        field_id, arg2, op_type = Env.decode_arg(env.fields)
        if field_id >= Env.FIELD_NUM:
            return None
        if op_type == Env.ACT_WRITE:  # function: write(field_id, digit_value)
            digit = arg2
            env.fields[field_id, digit] = 1
        elif op_type == Env.ACT_MOVE_PTR:  # function: move_pointer(field_id, move_direct)
            direct = arg2
            if direct == Env.MOVE_PTR_LEFT and env.pointers[field_id] > 0:
                env.pointers[field_id] -= 1
            elif direct == Env.MOVE_PTR_RIGHT and env.pointers[field_id] < (env.max_digits - 1):
                env.pointers[field_id] += 1
        # no sub program any more
        return None

    def pg_carry(self, env, args):
        """mark a 1 in the carry row one unit left"""
        sub_prog = [
            (self.pg["ACT"], Env.encode_arg(arg_list=[Env.FIELDS["CARRY"], Env.MOVE_PTR_LEFT, Env.ACT_MOVE_PTR])),
            (self.pg["ACT"], Env.encode_arg(arg_list=[Env.FIELDS["CARRY"], 1, Env.ACT_WRITE])),
            (self.pg["ACT"], Env.encode_arg(arg_list=[Env.FIELDS["CARRY"], Env.MOVE_PTR_RIGHT, Env.ACT_MOVE_PTR]))
        ]
        return sub_prog

    def pg_lshift(self, env, args):
        """shift a specified pointer one step left"""
        sub_prog = [
            (self.pg["ACT"], Env.encode_arg(arg_list=[Env.FIELDS["INPUT1"], Env.MOVE_PTR_LEFT, Env.ACT_MOVE_PTR])),
            (self.pg["ACT"], Env.encode_arg(arg_list=[Env.FIELDS["INPUT2"], Env.MOVE_PTR_LEFT, Env.ACT_MOVE_PTR])),
            (self.pg["ACT"], Env.encode_arg(arg_list=[Env.FIELDS["CARRY"], Env.MOVE_PTR_LEFT, Env.ACT_MOVE_PTR])),
            (self.pg["ACT"], Env.encode_arg(arg_list=[Env.FIELDS["OUTPUT"], Env.MOVE_PTR_LEFT, Env.ACT_MOVE_PTR]))
        ]
        return sub_prog

    def pg_rshift(self, env, args):
        """shift a specified pointer one step right"""
        sub_prog = [
            (self.pg["ACT"], Env.encode_arg(arg_list=[Env.FIELDS["INPUT1"], Env.MOVE_PTR_RIGHT, Env.ACT_MOVE_PTR])),
            (self.pg["ACT"], Env.encode_arg(arg_list=[Env.FIELDS["INPUT2"], Env.MOVE_PTR_RIGHT, Env.ACT_MOVE_PTR])),
            (self.pg["ACT"], Env.encode_arg(arg_list=[Env.FIELDS["CARRY"], Env.MOVE_PTR_RIGHT, Env.ACT_MOVE_PTR])),
            (self.pg["ACT"], Env.encode_arg(arg_list=[Env.FIELDS["OUTPUT"], Env.MOVE_PTR_RIGHT, Env.ACT_MOVE_PTR]))
        ]
        return sub_prog

