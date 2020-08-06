#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ti
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors

N = 14
path = "output/phasediagram/0508"

gap = joblib.load(f"{path}/N{N}_gap")
hal_val = joblib.load(f"{path}/N{N}_hal_val")
alph_val = joblib.load(f"{path}/N{N}_alph_val")

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

gap[gap>0.01]= np.NaN
fig, ax = plt.subplots()
plt.pcolormesh(hal_val, alph_val, np.transpose(gap), norm = colors.LogNorm(), cmap='inferno')
cbar = plt.colorbar()
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_title('Energy Gap')
plt.title(r'Log Scaled Phase Boundary: N=6, $\Delta$ =2.5e-3')
ax.grid()
ax.set_xlim([0,2])

labels1 =[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
locs1 =[2.0, 1.8, 1.6, 1.4, 1.2, 1.0]
ax.set_yticklabels(labels1)
ax.set_yticks(locs1)
ax.set_ylabel(r'$\beta$')

labels2 =[0.0, 0.5, 1.0]
locs2 =[2.0, 1.5, 1.0]
ax.set_xticklabels(labels2)
ax.set_xticks(locs2)
ax.set_xlabel('t')

ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
labels=[0.0, 0.5, 1.0]
locs=[0.0, 0.5, 1.0]
ax2.set_xticklabels(labels)
ax2.set_xticks(locs)
ax2.set_xlabel(r'$\lambda$')
ax2.grid()

plt.gcf().subplots_adjust(top=0.85)
fig.savefig(f"{path}/N{N}_diagram.pdf")