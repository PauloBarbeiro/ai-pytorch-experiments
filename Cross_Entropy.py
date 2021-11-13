import math
from functools import reduce
import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    if (len(Y) != len(P)):
        return

    sums = map(lambda i: Y[i]*np.log(P[i])+(1-Y[i])*np.log(1-P[i]), range(len(Y)))
    return reduce(lambda a, b: a+b, list(sums)) * -1

# Proposed solution
def cross_entropy_solution(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))

if __name__ == '__main__':
    r1 = cross_entropy([1, 1, 0], [.8, .7, .1])
    r2 = cross_entropy([0, 0, 1], [.8, .7, .1])
    print('---->> my final : ', r1, r2)
    rA = cross_entropy_solution([1, 1, 0], [.8, .7, .1])
    rB = cross_entropy_solution([0, 0, 1], [.8, .7, .1])
    print('final suggested : ', r1, r2)