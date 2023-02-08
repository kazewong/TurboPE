import paths
import numpy as np
import utils

rng = np.random.default_rng(12345)

print("Processing GW170817...")

flowMC_chains, bilby_chains = utils.get_chains('GW170817')

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
    [3.3, 3.5],
    [-0.5, -0.25],
]

g = utils.plot_corner(flowMC_chains, bilby_chains, lims=lims, rng=rng)
p = paths.figures / "GW170817.pdf"
g.fig.savefig(p, bbox_inches="tight")
print(f"Saved: {p}")
