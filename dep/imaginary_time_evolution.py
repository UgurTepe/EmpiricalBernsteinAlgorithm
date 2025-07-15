import numpy as np
from numpy.linalg import eigh
from dep.shadow_grouping import Hamiltonian

class ImaginaryTimeEvolution():
    
    def __init__(self,weights,obs,beta,init_state=None):
        n = len(obs[0])
        assert len(weights)==len(obs), "No. weights do not match no. observables"
        H = Hamiltonian(weights,obs)
        self.ham = H.SummedOp().to_matrix()
        #self.E_GS, groundstate = H.ground(sparse=False)
        #self.groundstate = groundstate.real

        self.eigenvals, states = eigh(self.ham)
        self.states = states.T.real # the eigenstates are real-valued because the Hamiltonian is as well
        self.E_GS, self.groundstate = self.eigenvals[0], self.states[0]
        
        self.set_beta(beta)
        self.init_state = init_state if init_state is not None else np.full(2**n , 2**(-n))
        self.reset()
        return
        
    def set_beta(self,beta):
        """ (Re-)Initializes beta value and calculates the corresponding time operator eigenvalues """
        self.beta = float(beta)
        self.probs = np.exp(-self.beta*(self.eigenvals - self.E_GS)) # the offset is cancelled when normalizing
        return
    
    def set_state(self,state):
        self.state = state
        self.total_beta = 0.0
        
    def reset(self):
        self.state = self.init_state.copy()
        self.total_beta = 0.0
    
    def step(self):
        """ Performs e^{-beta H}|s> (including normalization) for the given beta value. """
        overlaps = self.states @ self.state
        temp = overlaps * self.probs
        norm = np.sqrt(np.sum(temp**2))
        self.state = np.sum(self.states * temp[:,np.newaxis], axis=0) / norm
        self.total_beta += self.beta
        return
    
    @property
    def energy(self):
        return (self.state @ self.ham @ self.state).real
    
    @property
    def fidelity_with_groundstate(self):
        return abs(self.groundstate @ self.state)
