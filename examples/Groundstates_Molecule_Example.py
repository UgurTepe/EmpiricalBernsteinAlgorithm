import numpy as np
import sys
import matplotlib.pyplot as plt 

sys.path.append("./")
from src.algorithms import EBS
from src.algorithms import hoeffding_bound as tmin

# Set the style
plt.style.use('seaborn-v0_8-deep')

molecule = ["H2","H2_6-31g","LiH","BeH2","H2O","NH3"]
mappings = ["JW","BK","Parity","Hoeffding"]

def load_samples(filename):

    params = {}
    with open(filename, "r") as f:
        line = f.readline().strip()
        while line.find("#") >= 0:
            key, val = line.split("=")
            params[key[2:]] = float(val)
            line = f.readline().strip()
    samples = np.loadtxt(filename, skiprows=len(params.keys()))
    return samples, params

def GS_Comp_Hoeffding(molecule_name="H2",mapping_name="JW",epsilon=0.1,delta=0.1):
    samples, params = load_samples("dep/samples_gs/{}_molecule_{}_samples.txt".format(molecule_name, mapping_name))
    rng = params["R"]
    return np.round(tmin(delta=delta,epsilon=epsilon,rng=rng)),None

def GS_Comp(molecule_name="H2",mapping_name="JW",epsilon=0.1,delta=0.1):
    samples, params = load_samples("dep/samples_gs/{}_molecule_{}_samples.txt".format(molecule_name, mapping_name))
    np.random.shuffle(samples)
    ebs = EBS(epsilon=epsilon, delta=delta, range_of_rndvar=params["R"])

    ind = 0
    while ebs.cond_check():
        if ind == len(samples):
            np.random.shuffle(samples)
            ind = 0
        ebs.add_sample(samples[ind])
        ind += 1
    return np.round(ebs.get_numsamples()*params["Ngroups"]),ebs.get_estimate()

"""
Select paramterrs for the EBS algorithm
"""
epsilon = 0.0016
delta = 0.1

# Displaying Results
results = {}
for mol in molecule:
    results[mol] = {}
    for map in mappings:
        total_tasks = len(molecule) * len(mappings)
        completed_tasks = sum(len(results[mol]) for mol in results)
        remaining_tasks = total_tasks - completed_tasks
        percent_complete = (completed_tasks / total_tasks) * 100
        print(f"»»» {percent_complete:.2f}% complete | Processing molecule: {mol}, mapping: {map} «««")
        if map == "Hoeffding":
            num_samples, estimate = GS_Comp_Hoeffding(mol,delta=delta,epsilon=epsilon)
        else:
            num_samples, estimate = GS_Comp(mol, map, delta=delta, epsilon=epsilon)
        results[mol][map] = num_samples

fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.1
index = np.arange(len(molecule))

for i, map in enumerate(mappings):
    samples = [results[mol][map] for mol in molecule]
    ax.bar(index + i * bar_width, samples, bar_width, label=map)

ax.set_yscale('log')
ax.set_xlabel('Molecule')
ax.set_ylabel('Number of Samples')
ax.set_title('Groundstate Estimation for Different Molecules and Mappings')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(molecule)
ax.legend()

plt.tight_layout()
plt.show()