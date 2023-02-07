import paths
import numpy as np
import utils

rng = np.random.default_rng(12345)

print("Processing GW150914...")

flowMC_data = np.load(paths.data / 'GW150914_flowMC.npz')
bilby_data = np.genfromtxt(paths.data / 'GW150914_Bilby.dat')
flowMC_chains = flowMC_data['chains'][:,:,[0,1,2,3,4,6,7,8,9,10]].reshape(-1,10)
bilby_chains = bilby_data[1:,[1,0,2,3,6,11,9,10,8,7]]
flowMC_chains[:,6] = np.arccos(flowMC_chains[:,6])
flowMC_chains[:,9] = np.arcsin(flowMC_chains[:,9])

# parameter ranges
lims = [
    None,
    [0.5, 1],
    [-1, 1],
    [-1, 1],
    None,
    [0, 2*np.pi],
    [0, np.pi],
    [0, np.pi],
    None,
    [-1.3, -0.8],
]

g = utils.plot_corner(flowMC_chains, bilby_chains, lims=lims, rng=rng)
p = paths.figures / "GW150914.pdf"
g.fig.savefig(p, bbox_inches="tight")
print(f"Saved: {p}")

