import numpy as np

def num_after_point(x):
    s = str(x)
    if not '.' in s:
        return 0
    return len(s) - s.index('.') - 1

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)