#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ti
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
plt.rcParams['axes.axisbelow'] = True

N_or_res = "res"
N = 600
path = "output/phasediagram/periodic"

gap = joblib.load(f"{path}/{N_or_res}{N}_gap")
x = np.linspace(0,2,num=1201)
# x = joblib.load(f"{path}/{N_or_res}{N}_x")

# hal_val = joblib.load(f"{path}/{N_or_res}{N}_hal_val")
# alph_val = joblib.load(f"{path}/{N_or_res}{N}_alph_val")

# gap0 = joblib.load(f"{path}/N{N}_gap_v0")
# hal_val0 = joblib.load(f"{path}/N{N}_hal_val_v0")
# alph_val = joblib.load(f"{path}/N{N}_alph_val_v0")

# gap1 = joblib.load(f"{path}/N{N}_gap_v1")
# hal_val1 = joblib.load(f"{path}/N{N}_hal_val_v1")

# gap2 = joblib.load(f"{path}/N{N}_gap_v2")
# hal_val2 = joblib.load(f"{path}/N{N}_hal_val_v2")

# gap3 = joblib.load(f"{path}/N{N}_gap_v3")
# hal_val3 = joblib.load(f"{path}/N{N}_hal_val_v3")

# gap4 = joblib.load(f"{path}/N{N}_gap_v4")
# hal_val4 = joblib.load(f"{path}/N{N}_hal_val_v4")

# gap = np.concatenate((gap4,gap3), axis=0)
# gap = np.concatenate((gap,gap2), axis=0)
# gap = np.concatenate((gap,gap1), axis=0)
# gap = np.concatenate((gap,gap0), axis=0)

# hal_val = np.concatenate((hal_val4,hal_val3), axis=0)
# hal_val = np.concatenate((hal_val,hal_val2), axis=0)
# hal_val = np.concatenate((hal_val,hal_val1), axis=0)
# hal_val = np.concatenate((hal_val,hal_val0), axis=0)

gap[gap>1e-2]= np.NaN
fig, ax = plt.subplots()
plt.pcolormesh(x,x,gap, norm = colors.LogNorm(), cmap='inferno')
plt.scatter((2-0.693),(2-0.466),linewidth=0.1,marker='x')
# plt.title(r"Log Scaled Phase Boundary: Periodic, $\Delta$ = 1.7e-3")
ax.grid(linestyle='--')
# ax.set_xlim([0,0.4])
ax.set_aspect(1)

labels=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, r'$\frac{1}{0.8}$',r'$\frac{1}{0.6}$',r'$\frac{1}{0.4}$',r'$\frac{1}{0.2}$',r'$\infty$']
locs=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0,1.2,1.4,1.6,1.8,2.0]
ax.set_yticklabels(labels)
ax.set_yticks(locs)
ax.set_ylabel(r'$\alpha$')

# ax1 = ax.twinx()
# labels1 =[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
# locs1 =[2.0, 1.8, 1.6, 1.4, 1.2, 1.0]
# ax1.set_yticklabels(labels1)
# ax1.set_yticks(locs1)
# ax1.set_ylabel(r'$\beta$')
# # ax1.set_axisbelow(True)
# ax1.grid(linestyle='--', zorder=0)


ax.set_xticklabels(labels)
ax.set_xticks(locs)
ax.set_xlabel(r'$\lambda$')

# ax2 = ax.twiny()
# ax2.set_xlim(ax.get_xlim())
# ax2.set_xticklabels(labels1)
# ax2.set_xticks(locs1)
# ax2.set_xlabel('t')
# # ax2.set_axisbelow(True)
# ax2.grid(linestyle='--', zorder=0)


cbar = plt.colorbar(pad = 0.15)
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_title('Energy Gap')

plt.gcf().subplots_adjust(top=0.85)
fig.tight_layout()
fig.savefig(f"{path}/{N_or_res}{N}_diagram_yo.png", dpi=500,bbox_inches='tight')

# fig1 = plt.figure()
# plt.pcolormesh(x, x, gap, norm = colors.LogNorm(), cmap='inferno')
# fig1.savefig(f"{path}/periodic.png", dpi=500)