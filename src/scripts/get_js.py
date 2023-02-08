import numpy as np
import paths
import pandas as pd
import utils
from scipy.spatial.distance import jensenshannon
from scipy.special import kl_div
from scipy.stats import gaussian_kde

pars = [r'$M_c$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$d_{\rm{L}}$',
        r'$\phi_c$', r'$\iota$', r'$\psi$', r'$\alpha$', r'$\delta$']

def get_js(flowMC_chains, bilby_chains):
    jsd = []
    for i, l in enumerate(pars):
        total_samples = np.concatenate((flowMC_chains[:,i],bilby_chains[:,i]))
        p_range = (np.min(total_samples),np.max(total_samples))
        axis = np.linspace(p_range[0],p_range[1],100)
        flow_p = gaussian_kde(flowMC_chains[:,i])(axis)
        bilby_p = gaussian_kde(bilby_chains[:,i])(axis)
        jsd.append(jensenshannon(flow_p,bilby_p)**2)
    return pd.Series(dict(zip(pars, jsd)))

##############################################################################
# LOAD DATA
##############################################################################

flowMC_chains, bilby_chains = utils.get_chains('GW150914')
js_bbh = get_js(flowMC_chains, bilby_chains)

flowMC_chains, bilby_chains = utils.get_chains('GW170817')
js_bns = get_js(flowMC_chains, bilby_chains)

##############################################################################
# MAKE TABLE
##############################################################################

df = pd.DataFrame({'GW150914': js_bbh, 'GW170817': js_bns})
table = df.style.format(precision=5).to_latex(hrules=True)

# highlight highest values
for k, c in df.items():
    x = f'{c.max():.5f}'
    table = table.replace(x, "\\textbf{%s}" % x)

p = paths.output / "js_table.tex"
with open(p, 'w') as f:
    f.write(table)
print(f"Saved: {p}")

# cache JS divergences
df.to_csv(paths.data / "jsd.txt")
