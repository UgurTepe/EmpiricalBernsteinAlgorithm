from dep.qm_gates import *
from dep.qm_tools import *
from scipy.linalg import expm


h2_op = lambda p: p[0]*np.eye(4) + p[1]*(np.kron(i_gate(), z_gate())) + p[2]*(np.kron(z_gate(), i_gate())) + p[3]*(double_gate(z_gate())) + p[4]*(double_gate(x_gate())) + p[5]*(double_gate(y_gate()))

def h2(theta1, state):
    # Section1
    gate_1 = np.kron(ry_gate(np.pi/2), -rx_gate(np.pi/2))
    state = gate_1@state

    # Cnot
    gate_cn1 = cnot_gate()
    state = gate_cn1@state

    # Section2
    gate_2 = expm(-(1j)*theta1*(np.kron(x_gate(), i_gate()) @ np.kron(i_gate(), y_gate())))
    state = gate_2@state

    # Cnot
    gate_cn2 = cnot_gate()
    state = gate_cn2@state

    # Section3
    gate_3 = np.kron(-ry_gate(np.pi/2), rx_gate(np.pi/2))
    state = gate_3@state
    return state

PIhalf = np.pi/2

def RX_matrix(theta):
    c,s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c,complex(0,-s)],[complex(0,-s),c]])

def RY_matrix(theta):
    c,s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c,-s],[s,c]])

def RZ_matrix(theta):
    c,s = np.cos(theta/2), np.sin(theta/2)
    return np.diag([complex(c,-s),complex(c,s)])

CNOT = np.array([[0,1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]])

def circ(theta):
    state = np.array([0,1,0,0])
    state = np.kron(RY_matrix(PIhalf),-RX_matrix(PIhalf))@state
    state = CNOT@state
    state = np.kron(np.eye(2),RZ_matrix(theta))@state
    state = CNOT@state
    state = np.kron(-RY_matrix(PIhalf),RX_matrix(PIhalf))@state
    return state