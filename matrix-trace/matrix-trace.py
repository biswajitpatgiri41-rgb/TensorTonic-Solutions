import numpy as np

def matrix_trace(A):
    trace = 0
    for i in range(len(A)):
        trace += A[i][i]
    return trace