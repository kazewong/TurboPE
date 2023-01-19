import numpy as np
import matplotlib.pyplot as plt
import corner
import paths

flowMC_data = np.load('../data/GW170817_flowMC.npz')
bilby_data = np.genfromtxt('../data/GW170817_Bilby.dat')
axis_labels=  [r'$M_c$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$d_{\rm{L}}$', r'$t_c$', r'$\phi_c$', r'$\iota$', r'$\psi$', 'RA', 'DEC']
flowMC_chains = flowMC_data['chains'][:,:,[0,1,2,3,4,5,6,7,8,9,10]].reshape(-1,11)
bilby_chains = bilby_data[1:,[0,1,2,3,4,10,9,11,8,6,5]]
flowMC_chains[:,7] = np.arccos(flowMC_chains[:,7])
flowMC_chains[:,10] = np.arcsin(flowMC_chains[:,10])
bilby_chains[:,5] = bilby_chains[:,5] - 1187008882.4

fig = corner.corner(flowMC_chains,color='C0',labels=axis_labels,show_titles=True,hist_kwargs={'density':True})
fig = corner.corner(bilby_chains,fig=fig,color='C1',hist_kwargs={'density':True})

fig.savefig(paths.figures / "GW170817.pdf", bbox_inches="tight")