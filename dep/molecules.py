# various small multi-atomic molecules at equilibrium length

from qiskit_nature.drivers import Molecule
import numpy as np

available_molecules = ["H2","H2_6-31g","LiH","BeH2","H2O","NH3"]
available_molecules_E_GS = [-1.86,-1.86,-8.91,-19.05,-83.60,-66.88]

# source: http://hyperphysics.phy-astr.gsu.edu/hbase/Tables/diatomic.html
H2 = Molecule(geometry=[['H', [0., 0., 0.]],
                              ['H', [0., 0., 0.735]]],
                     charge=0, multiplicity=1)

# source: http://hyperphysics.phy-astr.gsu.edu/hbase/Tables/diatomic.html
LiH = Molecule(geometry=[['Li', [0., 0., 0.]],
                              ['H', [0., 0., 1.5475]]],
                     charge=0, multiplicity=1)

# source: https://en.wikipedia.org/wiki/Beryllium_hydride
BeH2 = Molecule(geometry=[['Be', [0., 0., 0.]],
                         ['H', [0., 0., 1.33376]],
                         ['H',[0.,0.,-1.33376],],
                        ],
                     charge=0, multiplicity=1)

# H2O
# source: https://en.wikipedia.org/wiki/Properties_of_water
alpha_H2O = np.deg2rad(14.45)
R_H2O = 0.9584



H2O = Molecule(geometry=[['O', [0., 0., 0.]],
                         ['H', [0., 0., R_H2O]],
                         ['H',[0.,np.cos(alpha_H2O)*R_H2O,-R_H2O*np.sin(alpha_H2O)],],
                        ],
                     charge=0, multiplicity=1)

#NH3 geometry as a trigonal pyramide
# source: https://en.wikipedia.org/wiki/Ammonia_(data_page)
alpha = np.deg2rad(106.7) # angle between any two H atoms
R = 1.012 # equilibrium bond distance between N and any H
beta = np.deg2rad(60) # planar angle of the base triangle spanned by all H
# geometry considerations:
d = R*np.sin(alpha/2)/np.sin(beta)
x = np.sin(beta/2)*d
y = np.cos(beta/2)*d
H = R*np.sqrt(1-np.sin(alpha/2)**2/(np.sin(beta)**2))

NH3 = Molecule(geometry=[['N', [0., 0., H]],
                         ['H', [d, 0., 0.]],
                         ['H', [-x, y, 0.]],
                         ['H', [-x, -y, 0.]]
                        ],
                     charge=0, multiplicity=1)


# geometry suggested / analyzed in http://dx.doi.org/10.1021/acs.jpclett.3c01106
H4 = Molecule(geometry=[['H', [i, 0., 0.]] for i in range(0,7,2)], charge=0, multiplicity=1)
