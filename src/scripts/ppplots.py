import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import uniform
from scipy.stats import kstest
import paths

params = {
    "font.size": 22,
    "legend.fontsize": 22,
    "legend.frameon": False,
    "axes.labelsize": 22,
    "axes.titlesize": 22,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "figure.figsize": (7, 5),
    "xtick.top": True,
    "axes.unicode_minus": False,
    "ytick.right": True,
    "xtick.bottom": True,
    "ytick.left": True,
    "xtick.major.pad": 8,
    "xtick.major.size": 8,
    "xtick.minor.size": 4,
    "ytick.major.size": 8,
    "ytick.minor.size": 4,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.linewidth": 1.5,
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": "cmr10",
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,  # needed when using cm=cmr10 for normal text
}


mpl.rcParams.update(params)


ppPlot_data = np.load(paths.data / 'combined_quantile_balance_LVK.npz')
result = ppPlot_data['result']
result_multimodal = ppPlot_data['result_multimodal']
true_param = ppPlot_data['true_param']
mean_global_accs = ppPlot_data['mean_global_accs']
mean_local_accs = ppPlot_data['mean_local_accs']

def makeCumulativeHist(data):
    h = np.histogram(data, bins=100, range=(0,1), density=True)
    return np.cumsum(h[0])/100.

counts = 1200

N = 10000
uniform_data = np.random.uniform(size=(N,counts))
cum_hist = []
for i in range(N):
    cum_hist.append(makeCumulativeHist(uniform_data[i]))
cum_hist = np.array(cum_hist)
upper_quantile_array = []
lower_quantile_array = []
percentile = 0.05
for i in range(100):
    upper_quantile_array.append(np.quantile(cum_hist[:,i], (1-percentile/2)))
    lower_quantile_array.append(np.quantile(cum_hist[:,i], (percentile/2)))

axis_labels=  [r'$M_c$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$d_{\rm{L}}$',
               r'$t_c$', r'$\phi_c$', r'$\cos\iota$', r'$\psi$', r'$\alpha$', r'$\sin{\delta}$']

plt.figure(figsize=(10,9))
bins = np.linspace(0,1,101)
bins = (bins[1:]+bins[:-1])/2
plt.fill_between(bins, lower_quantile_array, upper_quantile_array, alpha=0.5)

x = np.append(0,bins)
pvalues = []
for i, l in enumerate(axis_labels):
    p = kstest(result[:counts,i], cdf=uniform(0,1).cdf).pvalue
    y = np.append(0,makeCumulativeHist(result[:counts,i]))
    plt.plot(x, y, label=f"{l} ($p = {p:.2f}$) ")
    pvalues.append(p)
plt.legend(loc='upper left', fontsize=20, handlelength=1)
plt.xlabel(r'confidence level')
plt.ylabel(r'fraction of samples with confidence level $\leq x$')

ptotal = kstest(pvalues, cdf=uniform(0,1).cdf).pvalue

p = paths.figures / "ppplot.pdf"
plt.savefig(p, bbox_inches="tight")
print(f"Saved: {p}")

# cache p-values
np.savetxt(paths.data / "pvalues.txt", pvalues)
