import numpy as np

# Identity gate


def i_gate():
    return np.array([[1, 0],
                     [0, 1]])
# Pauli X_gate


def x_gate():
    return np.array([[0, 1],
                     [1, 0]])
# Pauli Y_gate


def y_gate():
    return np.array([[0, -1j],
                     [1j, 0]])
# Pauli Z_gate


def z_gate():
    return np.array([[1, 0],
                     [0, -1]])
# Hadamard gate


def h_gate():
    return (1/np.sqrt(2))*np.array([[1, 1],
                                    [1, -1]])
# Controleld NOT gate


def cnot_gate():
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]])
# General controlled gate with a single qubit gate


def cu_gate(A):
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, A[0, 0], A[0, 1]],
                     [0, 0, A[1, 0], A[1, 1]]])
# Inverted Cnot gate


def cnot_inv_gate():
    return np.array([[1, 0, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0]])
# General inverted controlled gate with a single qubit gate


def cu_inv_gate(A):
    return np.array([[1, 0, 0, 0],
                     [0, A[0, 0], 0, A[0, 1]],
                     [0, 0, 1, 0],
                     [0, A[1, 0], 0, A[1, 1]]])
# Rotation arround the X axis


def rx_gate(theta):
    return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                     [-1j*np.sin(theta/2), np.cos(theta/2)]])
# Rotation arround the Y axis


def ry_gate(theta):
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                     [np.sin(theta/2), np.cos(theta/2)]])
# Rotation arround the Z axis


def rz_gate(theta):
    return np.array([[np.exp(-1j*theta/2), 0],
                     [0, np.exp(1j*theta/2)]])
# s gate


def s_gate():
    return np.array([[1, 0],
                     [0, 1j]])
# s-dagger gate


def s_dagger_gate():
    return np.array([[1, 0],
                     [0, -1j]])
# not gate


def not_gate():
    return np.array([[0, 1],
                     [1, 0]])
# example VQE unitary from: ttps://qiskit.org/textbook/ch-applications/vqe-molecules.html


def u_gate(theta, phi, lmbda):
    return np.array([[np.cos(theta/2), -np.exp(1J*lmbda)*np.sin(theta/2)],
                     [np.exp(1J*phi)*np.sin(theta/2), np.exp(1J*lmbda+1j+phi)*np.cos(theta/2)]])


def double_gate(gate):
    return (np.kron(i_gate(), gate) @ np.kron(gate, i_gate()))


