import numpy as np

def step_function(x):
    return np.array(x > 0, dtype=np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def test_relu():
    assert relu(-1) == 0
    assert relu(6) == 6

# 输出层常用的激活函数
# 一般地，回归使用恒等，二元分类使用sigmoid，多元分类使用softmax

def identity_function(x):
    return x