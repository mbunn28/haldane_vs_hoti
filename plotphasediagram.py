#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ti
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors

N = 16
path = "output/phasediagram"
gap = joblib.load(f"{path}/N{N}_gap")
hal_val = joblib.load(f"{path}/N{N}_hal_val")
alph_val = joblib.load(f"{path}/N{N}_alph_val")

gap[gap>0.05]= np.NaN
fig, ax = plt.subplots()
plt.pcolormesh(hal_val, alph_val, np.transpose(gap), norm = colors.LogNorm(), cmap='inferno')
cbar = plt.colorbar()
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_title('Energy Gap')
plt.title(r'Log Scaled Phase Boundary: N=16, $\Delta$ =8.3e-3')

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
# labels = [w.get_text() for w in ax2.get_xticklabels()]
# locs=list(ax2.get_xticks())
labels=[0.0, 0.5, 1.0]
locs=[0.0, 0.25, 0.5]
ax2.set_xticklabels(labels)
ax2.set_xticks(locs)
ax2.set_xlabel(r'$\lambda$')

plt.gcf().subplots_adjust(top=0.85)
fig.savefig(f"{path}/N{N}_diagram.pdf")