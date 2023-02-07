import numpy as np
import paths
import seaborn as sns
import pandas as pd
import matplotlib as mpl

sns.set(context='notebook', palette='colorblind', style='ticks', font_scale=1.5)

params = {
    "axes.labelsize": 22,
    "axes.titlesize": 22,
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": "cmr10",
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,  # needed when using cm=cmr10 for normal text
}

mpl.rcParams.update(params)

labels = [r'$M_c/M_\odot$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$d_{\rm{L}}/{\rm Mpc}$',
               r'$\phi_c$', r'$\iota$', r'$\psi$', r'$\alpha$', r'$\delta$']

CLINE = sns.color_palette(desat=0.5)[0]

def plot_corner(chains1, chains2, nsamp=1000, cline=CLINE, lims=None, rng=None):
    df1 = pd.DataFrame(chains1, columns=labels).sample(nsamp, random_state=rng)
    df2 = pd.DataFrame(chains2, columns=labels).sample(nsamp, random_state=rng)

    g = sns.PairGrid(df2, corner=True, diag_sharey=False)
    g.map_diag(sns.histplot, color='gray', alpha=0.4, element='step', fill=True)
    g.map_lower(sns.kdeplot, color='gray', alpha=0.4, levels=5, fill=True)

    g.data = df1
    g.map_diag(sns.histplot, color=cline, element='step', linewidth=2, fill=False)
    g.map_lower(sns.kdeplot, color=cline, levels=5, linewidths=2)

    # set axis limits
    if lims is not None:
        for i, axes in enumerate(g.axes):
            for j, ax in enumerate(axes):
                if ax is not None:
                    if lims[j]:
                        ax.set_xlim(lims[j])
                    if lims[i] and i != j:
                        ax.set_ylim(lims[i])
    
    # add legend
    ax = g.axes[0,0]
    ax.plot([], [], color=cline, lw=3, label="flowMC")
    ax.plot([], [], color='gray', alpha=0.4, lw=3, label="bilby")
    ax.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), frameon=False,
              fontsize=20);
    return g

