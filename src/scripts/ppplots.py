import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import uniform
from scipy.stats import kstest
import paths

params = {
    "font.size": 18,
    "legend.fontsize": 18,
    "legend.frameon": False,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
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
    h = np.histogram(data,bins=100,range=(0,1),density=True)
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
    upper_quantile_array.append(np.quantile(cum_hist[:,i],(1-percentile/2)))
    lower_quantile_array.append(np.quantile(cum_hist[:,i],(percentile/2)))

axis_labels=  [r'$M_c$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$d_{\rm{L}}$', r'$t_c$', r'$\phi_c$', r'$\cos{i}$', r'$\psi$', r'$\alpha$', r'$\sin{\delta}$']

plt.figure(figsize=(10,9))
bins = np.linspace(0,1,101)
bins = (bins[1:]+bins[:-1])/2
plt.fill_between(bins,lower_quantile_array,upper_quantile_array,alpha=0.5)
pvalues = []
for i in range(11):
    pvalues.append(kstest(result[:counts,i],cdf=uniform(0,1).cdf).pvalue)
    plt.plot(np.append(0,bins),np.append(0,makeCumulativeHist(result[:counts,i])), label=axis_labels[i]+" "+str(round(pvalues[-1],2)))
plt.legend(loc='upper left',fontsize=14)
plt.xlabel(r'Confidence Level')
plt.ylabel(r'Fraction of Samples with Confidence Level $\leq$ x')
plt.title('Combined p-value: '+str(round(kstest(pvalues,cdf=uniform(0,1).cdf).pvalue,2)))

plt.savefig(paths.figures / "ppplot.pdf", bbox_inches="tight")