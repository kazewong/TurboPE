import paths
import numpy as np
import utils

rng = np.random.default_rng(12345)

print("Processing GW150914...")

flowMC_chains, bilby_chains = utils.get_chains('GW150914')

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

