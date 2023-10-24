import numpy as np
from qibo import models, gates # checked for version 0.1.7
from os.path import isfile, join
from qiskit.quantum_info import Pauli # checked for version 0.43.1
from qiskit.opflow import PauliOp, SummedOp
from scipy.sparse.linalg import eigsh 


char_to_int = {"I":0,"X":1,"Y":2,"Z":3}
int_to_char = {item: key for key,item in char_to_int.items()}

class StateSampler(): 
    """ Convenience class that holds a fixed state of length 2**num_qubits. The latter number is inferred automatically.
        Provides a sampling method that obtains samples from the state in the chosen basis.

        Input:
        - state, numpy array of size 2**N with N = num_qubits. Coefficients are to be specified in the computational basis.
    """
    def __init__(self,state):
        self.state = np.array(state)
        self.num_qubits = int(np.round(np.log2(len(state)),0))
        assert len(self.state) == 2**self.num_qubits, "State size has to be of size 2**N for some integer N."
        self.circuit = models.Circuit(self.num_qubits)
        self.X = [gates.H,gates.I] # use Hadamard gate to switch to +/- basis
        S_dagger = lambda i: gates.U1(i,-np.pi/2)
        self.Y = [S_dagger,gates.H] # use Hadamard + phase gate to switch to +/-i basis
        self.Z = [gates.I,gates.I] # no need to change if already in comp. basis or not measuring at all
        self.I = self.Z
        return

    def sample(self,meas_basis=None,nshots=1):
        """ Draws <nshots> samples from the state.
            If no measurement basis is defined, samples are drawn in the computational basis.

            Inputs:
            - meas_basis as a Pauli string, i.e. a str of length num_qubits containing only X,Y,Z,I.
            - nshots (int) specifying how many samples to draw.

            Returns:
            - samples, numpy array of shape (nshots x  num_qubits).
        """
        c = self.circuit.copy(deep=True)
        if meas_basis is not None:
            assert len(meas_basis)==self.num_qubits, "Measurement basis has to be specified for each qubit."
            c.add([getattr(self,s)[0](i) for i,s in enumerate(meas_basis)])
            c.add([getattr(self,s)[1](i) for i,s in enumerate(meas_basis)])
        c.add(gates.M(*range(self.num_qubits)))
        out = c(initial_state=self.state.copy(),nshots=nshots)
        out = out.samples()
        # mask-out non-measured qubits in <meas-basis>
        out = out * np.array([s!="I" for s in meas_basis],dtype=int)[np.newaxis,:]
        return -2*out + 1 # map {0,1} to {1,-1} outcomes

    def index_to_string(self,index_list):
        """ Helper function that maps a list of Pauli indices to a Pauli string, i.e.
            0 -> I, 1 -> X, 2 -> Y, 3 -> Z
            Returns the Pauli string.
        """
        pauli_string = ""
        for ind in np.array(index_list,dtype=int):
            assert ind in range(4), "Elements of index_list have to be in {0,1,2,3}."
            pauli_string += int_to_char[ind]
        return pauli_string
    
class L1_sampler:
    """ Comparison class that does not reconstruct the Hamiltonian expectation value by its components, but by its relative signs. """

    def __init__(self,observables,weights):
        assert len(observables.shape) == 2, "Observables has to be a 2-dim array."
        M,n = observables.shape
        weights = weights.flatten()
        assert len(weights) == M, "Number of weights not matching number of provided observables."
        abs_vals = np.abs(weights)
        #print(len(abs_vals))

        self.obs         = observables
        self.num_obs     = M
        self.num_qubits  = n
        self.w           = weights
        self.prob        = abs_vals / np.sum(abs_vals)
        self.shots       = 0
        self.is_sampling = True
        self.is_adaptive = False

        return

    def reset(self):
        self.shots = 0

    def find_setting(self,num_samples=1):
        self.shots += num_samples
        inds = np.random.choice(self.num_obs,size=(num_samples,),p=self.prob)
        return inds

    def get_Hoeffding_bound(self,epsilon):
        return 2*np.exp(-0.5*epsilon**2*self.shots/np.sum(np.abs(self.w))**2)

    def get_epsilon(self,delta):
        return np.sqrt(2/self.shots*np.log(2/delta)) * np.sum(np.abs(self.w))
    
class Sign_estimator():

    def __init__(self,measurement_scheme,state,offset):
        assert measurement_scheme.num_qubits == state.num_qubits, "Measurement and state scheme do not match in terms of qubit number."
        self.measurement_scheme = measurement_scheme
        self.state        = state
        self.offset       = offset
        self.setting_inds = []
        self.outcomes     = []
        self.num_settings = 0
        self.num_outcomes = 0

    def reset(self):
        self.setting_inds = []
        self.outcomes     = []
        self.num_settings, self.num_outcomes = 0, 0
        self.measurement_scheme.reset()

    def propose_next_settings(self,num_steps=1):
        """ Find the <num_steps> next setting(s) via the provided measurement scheme. """
        inds = self.measurement_scheme.find_setting(num_steps)
        self.setting_inds = np.append((self.setting_inds,inds)) if len(self.setting_inds)>0 else inds
        self.num_settings += num_steps
        return


    def measure(self):
            """ If there are proposed settings in self.settings that have not been measured, do so.
                The internal state of the VQE does not alter by doing so.
            """
            num_meas = self.num_settings - self.num_outcomes
            if num_meas > 0:
                # run all the last prepared measurement settings
                # from the settings list, fetch the unique settings and their respective counts
                recent_settings = self.setting_inds[-num_meas:]            
                outcomes = np.zeros(num_meas,dtype=int)
                for unique,nshots in zip(*np.unique(recent_settings,return_counts=True)):
                    setting = self.measurement_scheme.obs[unique]
                    samples = self.state.sample(meas_basis=self.state.index_to_string(setting),nshots=nshots)
                    outcomes[recent_settings==unique] = np.prod(samples,axis=-1)
                self.outcomes = np.append(self.outcomes,outcomes)
                self.num_outcomes += num_meas
            else:
                print("No more measurements required at the moment. Please propose new setting(s) first.")
            return
    
    def get_energy(self):
        """ Takes the current outcomes and estimates the corresponding energy. """
        if self.num_outcomes == 0:
            # if no measurements have been done yet, just return the offset value
            return self.offset
        w = self.measurement_scheme.w
        sgn = np.sign(w)
        norm = np.sum(np.abs(w))
        energy = np.mean(self.outcomes*sgn[self.setting_inds])*norm
        return energy + self.offset
    
    def get_energy_samples(self):
        """ Takes the current outcomes and returns them """
        if self.num_outcomes == 0:
            # if no measurements have been done yet, just return the offset value
            return self.offset
        w = self.measurement_scheme.w
        sgn = np.sign(w)
        norm = np.sum(np.abs(w))
        energy = self.outcomes*sgn[self.setting_inds]*norm
        #print(energy)
        return energy + self.offset
    
class Hamiltonian():
    """ Helper class to turn a list of Pauli operators with accompanying weights into a (sparse) Hamiltonian and diagonalize it.
        Code copied and modified from https://github.com/charleshadfield/adaptiveshadows/blob/main/python/hamiltonian.py
    """

    def __init__(self, weights, observables):
        self.weights = weights
        self.observables = observables

    def SummedOp(self):
        paulis = []
        for P, coeff_P in zip(self.observables,self.weights):
            paulis.append(coeff_P * PauliOp(Pauli(P)))
        return SummedOp(paulis)

    def ground(self, sparse=False):
        if not sparse:
            mat = self.SummedOp().to_matrix()
            evalues, evectors = np.linalg.eigh(mat)
        else:
            mat = self.SummedOp().to_spmatrix()
            evalues, evectors = eigsh(mat, which='SA')
            # SA looks for algebraically small evalues
        index = np.argmin(evalues)
        ground_energy = evalues[index]
        ground_state = evectors[:,index]
        return ground_energy, ground_state

def load_pauli_list(folder_hamiltonian,molecule_name="H2",basis_name="STO3g",encoding="JW",verbose=False,sparse=False,diagonalize=True):
    """ Loads the Pauli operators and the corresponding ground-state energy from the files of
        https://github.com/charleshadfield/adaptiveshadows
        Requires the name of the folder where all the Hamiltonians are stored together with the selection of the
        molecule, basis set and encoding. If verbose is set to True, some elements of the Pauli list are printed to console.
        If sparse is set to True, carries out the numerical diagonalization on a sparse form of the Hamiltonian.
        If diagonalize is set to False, only returns the Pauli decomposition from file and sets all other return values to None.

        Returns the observables, their respective weight, the offset energy, the exact ground-state energy and the state.
    """
    filename = "{}_{}_{}.txt".format(molecule_name,basis_name,encoding)
    full_file_name = join(folder_hamiltonian,filename)
    #assert isfile(full_file_name), "File not found: {}".format(full_file_name)     
    
    # extract Pauli list from file
    data = np.loadtxt('config/H2_STO3g_JW.txt',dtype=object)
    paulis, weights = data[::2].astype(str), data[1::2].astype(complex).real

    if diagonalize:
        # use Pauli list to create Hamiltonian and diagonalize it afterwards to obtain ground-state
        H = Hamiltonian(weights,paulis)
        E_numerics, state = H.ground(sparse=sparse)
    else:
        E_numerics = None
        state = None

    # Pauli item "III...II" in list should correspond to energy offset
    ind = -1
    identity = "I"*len(paulis[0])
    for i,p in enumerate(paulis):
        if p == identity:
            ind = i
            break
    if ind == -1:
        offset = 0
        obs = paulis
        w = weights
    else:
        offset = weights[ind]
        # erase the corresponding entry in paulis and weights
        obs = np.delete(paulis,ind)
        w = np.delete(weights,ind)
        assert len(obs) == len(paulis) - 1, "Error in line eraser."
        assert len(obs) == len(w), "Both arrays are not of equal length anymore."

    # print some to console
    if verbose:
        print("Offset","\t\t",offset)
        for i, (p, we) in enumerate(zip(obs, w)):
            print(p,"\t",we)
            if i == 9:
                print("\t","...")
                break

    # convert string characters to integers
    observables = np.array([[char_to_int[c] for c in o] for o in obs],dtype=int)

    return observables, w, offset, E_numerics, state
