
import numpy as np
import numpy.linalg as LA
import sys
sys.path.append("./")
from dep.qm_gates import *

def measure(state, flag_basis='z'):
    assert len(
        state) <= 4, "Shape of state is too big. Only 1 or 2 qubit states are allowed"
    if LA.norm(state) != 1:
        state = normalize(state)

    # Measurement in z bais
    if flag_basis == 'z':
        prob_state = (np.absolute(state)**2).flatten()
        # Chooses either |00>,|10>,|01> or |11> according to their respective probabilities
        a = np.random.choice(len(prob_state), p=prob_state)
        x = int(len(prob_state)/2)
        b = np.binary_repr(a, width=x)
        c = np.array([int(x) for x in b])
        return -2*c+1

    # Measurement in x basis
    if flag_basis == 'x':
        if len(state) == 4:
            gate = np.kron(h_gate(), h_gate())
        elif len(state) == 2:
            gate = h_gate()
        prob_state = (np.absolute(gate@state)**2).flatten()
        # Chooses either |00>,|10>,|01> or |11> according to their respective probabilities
        a = np.random.choice(len(prob_state), p=prob_state)
        x = int(len(prob_state)/2)
        b = np.binary_repr(a, width=x)
        c = np.array([int(x) for x in b])
        return -2*c+1

    # Measurment in y basis
    if flag_basis == 'y':
        if len(state) == 4:
            gate = np.kron(s_dagger_gate()@h_gate(), s_dagger_gate()@h_gate())
        elif len(state) == 2:
            gate = s_dagger_gate()@h_gate()
        prob_state = (np.absolute(gate@state)**2).flatten()
        # Chooses either |00>,|10>,|01> or |11> according to their respective probabilities
        a = np.random.choice(len(prob_state), p=prob_state)
        x = int(len(prob_state)/2)
        b = np.binary_repr(a, width=x)
        c = np.array([int(x) for x in b])
        return -2*c+1

    # Measurment in zx basis
    if flag_basis == 'zx':
        return np.array([measure(state)[0], measure(state, 'x')[1]])

    # Measurement in zx basis
    if flag_basis == 'xz':
        return np.array([measure(state, 'x')[0], measure(state)[1]])
    
    if flag_basis == '0z':
        return np.array([0, measure(state)[1]])
    
    if flag_basis == 'z0':
        return np.array([measure(state)[1],0])

def normalize(state):
    norm = LA.norm(state)
    return state / norm

def expected_value(state, operator):
    state_dagger = state.conj()
    state = operator@state
    result = np.inner(state_dagger, state)
    return result.real

def h2_measure(state_input,para):
    z_m = measure(state_input,flag_basis='z')
    x_m = measure(state_input,flag_basis='x')
    y_m = measure(state_input,flag_basis='y')
    
    return para[0] + para[1]*z_m[1] + para[2]*z_m[0] + para[3]*np.prod(z_m) + para[4]*np.prod(x_m) + para[5]*np.prod(y_m)