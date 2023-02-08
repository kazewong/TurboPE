import numpy as np
import paths
import utils

rng = np.random.default_rng(12345)

print("Processing GW170817...")

flowMC_data = np.load(paths.data / 'GW170817_flowMC_1800.npz')
bilby_data = np.genfromtxt(paths.data / 'GW170817_Bilby_flat.dat')
flowMC_chains = flowMC_data['chains'][:,:,[0,1,2,3,4,6,7,8,9,10]].reshape(-1,10)
bilby_chains = bilby_data[1:,[0,1,2,3,4,8,9,7,6,5]]
flowMC_chains[:,6] = np.arccos(flowMC_chains[:,6])
flowMC_chains[:,9] = np.arcsin(flowMC_chains[:,9])

# parameter ranges
lims = [
    None,
    [0.5, 1],
    [-0.05, 0.05],
    [-0.05, 0.05],
    None,
    [0, 2*np.pi],
    [np.pi/2, np.pi],
    [0, np.pi],
    None,
    None,
]

g = utils.plot_corner(flowMC_chains, bilby_chains, lims=lims, rng=rng)
p = paths.figures / "GW170817.pdf"
g.fig.savefig(p, bbox_inches="tight")
print(f"Saved: {p}")
