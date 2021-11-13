import math
from functools import reduce
import numpy as np

def softmax(List):
    exponents = list(map(lambda n: math.exp(n), List))
    expo_sums = reduce(lambda a, b: a + b, exponents)
    return list(map(lambda a, b=expo_sums: a / b, exponents))


def softmax_solution(L):
    expL = np.exp(L)
    sumExpL = sum(expL)
    result = []
    for i in expL:
        result.append(i * 1.0 / sumExpL)
    return result

    # Note: The function np.divide can also be used here, as follows:
    # def softmax(L):
    #     expL = np.exp(L)
    #     return np.divide (expL, expL.sum())

if __name__ == '__main__':
    r = softmax([2, 1, 0])
    print('final: ', r)