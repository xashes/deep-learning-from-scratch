#!/usr/bin/env python3

import numpy as np


def perceptron(inputs, weights, biasing):
    mid_output = np.sum(inputs * weights) + biasing
    if mid_output <= 0:
        return 0
    else:
        return 1


def andgate(i1, i2):
    inputs = np.array([i1, i2])
    weights = np.array([0.5, 0.5])
    biasing = -0.7
    return perceptron(inputs, weights, biasing)


def test_andgate():
    assert andgate(0, 0) == 0
    assert andgate(1, 1) == 1
    assert andgate(1, 0) == 0
    assert andgate(0, 1) == 0


def orgate(i1, i2):
    inputs = np.array([i1, i2])
    weights = np.array([0.5, 0.5])
    biasing = -0.2
    return perceptron(inputs, weights, biasing)


def test_orgate():
    assert orgate(0, 0) == 0
    assert orgate(1, 1) == 1
    assert orgate(0, 1) == 1
    assert orgate(1, 0) == 1


def nandgate(i1, i2):
    inputs = np.array([i1, i2])
    weights = np.array([-0.5, -0.5])
    biasing = 0.7
    return perceptron(inputs, weights, biasing)


def test_nandgate():
    assert nandgate(1, 1) == 0
    assert nandgate(0, 1) == 1
    assert nandgate(1, 0) == 1
    assert nandgate(0, 0) == 1


def xorgate(i1, i2):
    i11 = nandgate(i1, i2)
    i12 = orgate(i1, i2)
    return andgate(i11, i12)


def test_xorgate():
    assert xorgate(0, 0) == 0
    assert xorgate(1, 1) == 0
    assert xorgate(0, 1) == 1
    assert xorgate(1, 0) == 1
