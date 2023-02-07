import paths
import numpy as np
import pandas as pd
from scipy.stats import uniform
from scipy.stats import kstest

macros = []

# make injection p-value macro
pvalues = np.loadtxt(paths.data / "pvalues.txt")
ptotal = kstest(pvalues, cdf=uniform(0,1).cdf).pvalue
macros.append(r"\renewcommand{\ptot}{%.2f\xspace}" % ptotal)

# make JSD macros
jsd = pd.read_csv(paths.data / "jsd.txt", index_col=0)
for k,v in jsd.items():
    l = 'BBH' if '150914' in k else 'BNS'
    macros.append(r"\renewcommand{\jsdMax%s}{%.4f\xspace}" % (l, v.max()))
    macros.append(r"\renewcommand{\jsdAvg%s}{%.4f\xspace}" % (l, v.mean()))

p = paths.output / "macros.tex"
with open(p, 'w') as f:
    f.write('\n'.join(macros))
print(f"Saved: {p}")
