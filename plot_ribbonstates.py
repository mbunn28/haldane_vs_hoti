#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from math import sqrt
from math import ceil
from math import floor
import os
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from scipy import sparse
from scipy.sparse.linalg import eigsh
from numpy import random
import joblib
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
plt.rcParams['axes.axisbelow'] = True

path = "output/phasediagram/ribbon"
if not os.path.exists(path):
            os.makedirs(path)

a = 1
b = 0.38
t = 1
l = 1
N = 96
periodic = False
res=250
phi = np.pi/2

if a == 1 and b == 1:
    aorb_name = 'ab'
    aorb = 1
elif a == 1:
    aorb_name = 'b'
    aorb = b
else:
    aorb_name = 'a'
    aorb = a

if t == 1 and l == 1:
    torl_name = 'tl'
    torl = 1
elif t == 1:
    torl_name = 'l'
    torl = l
else:
    torl_name = 't'
    torl = t

def pos_flat(i,s): # Gives the x,y coordinates of site i,s
    A = np.array([-1/2, sqrt(3)/2])
    B = np.array([ -1/2, -sqrt(3)/2])
    C = np.array([1,0])
    cell =  3*i*C
    if s==0: return(cell+B)
    if s==1: return(cell+B+C)
    if s==2: return(cell+C)
    if s==3: return(cell+A+C)
    if s==4: return(cell+A)
    if s==5: return(cell+B+A)

def pos(i,s): # Gives the x,y coordinates of site i,s
    A = np.array([sqrt(3)/2, -1/2])
    B = np.array([ -sqrt(3)/2,-1/2])
    C = np.array([0,1])
    cell =  3*i*(C+A)
    if s==0: return(cell+A+B)
    if s==1: return(cell+A)
    if s==2: return(cell+A+C)
    if s==3: return(cell+C)
    if s==4: return(cell+B+C)
    if s==5: return(cell+B)

def lat(i,s): return(6*i+s)

energies = joblib.load(f"{path}/res{res}_energies_{aorb_name}{aorb}_{torl_name}{torl}")
evecs = joblib.load(f"{path}/res{res}_evecs_{aorb_name}{aorb}_{torl_name}{torl}")
# evecs = np.transpose(evecs, axes=(0,2,1))

# k_index = 10
# wavefunction_index = 288
shift=0

low_energies = np.argwhere(np.abs(energies)<0.002)
print(low_energies.shape)
k_index = low_energies[2, 0]
wavefunction_index = low_energies[2,1]

psi = evecs[k_index, :, wavefunction_index] #wavefunction
proba = (np.abs(psi))**2
# proba = proba/np.max(proba)
# assert np.sum(proba)==1
# print(proba[:20])
# print(proba[-20:])

x = np.zeros(6*N)
for i in range(N):
    for l in range(6):
        x[lat(i,l)]=pos(i,l)[0]

x_vals = np.unique(x)
wave = np.zeros(len(x_vals))
means = np.zeros(len(x_vals))
for i in range(len(x_vals)):
    indices = np.argwhere(x==x_vals[i])
    mean = 0
    for j in range(len(indices)):
        mean = mean + proba[indices[j]]
        wave[i] = wave[i] + proba[indices[j]]
    mean = mean/len(indices)
    means[i] = mean

fig1 = plt.figure()
plt.plot(x_vals,wave)
# plt.plot(x_vals, means,'r')

fig1.tight_layout()
fig1.savefig(f"{path}/estate1.png", dpi=500, bbox_inches='tight')

print(wave.shape)
print(3*48)
print(np.sum(wave[:3*48]))

sums = np.zeros(len(x_vals))
antisums = np.zeros(len(x_vals))
for j in range(len(x_vals)):
    sums[j] = np.sum(wave[:j])
    antisums[-j] = np.sum(wave[-j:])
print(np.argwhere(sums < 0.5)[-1])
print(np.argwhere(antisums < 0.01)[0])


# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title(f"E: {np.round(energies[k_index,wavefunction_index],4)+shift}", fontsize=10)

# cmap = matplotlib.cm.get_cmap('inferno_r')
# normalize = matplotlib.colors.Normalize(vmin=0, vmax=1)
# colors = [cmap(normalize(value)) for value in proba]

# #plot the probability distribution:
# x  = np.zeros(6*N)
# y = np.zeros(6*N)
# for i in range(N):
#     for l in range(6):
#         x[lat(i,l)] = pos_flat(i,l)[0]
#         y[lat(i,l)] = pos_flat(i,l)[1]
#         circle = Circle(pos_flat(i,l),0.5,color=colors[lat(i,l)],alpha=1,ec=None,zorder=1)
#         ax.add_artist(circle)

# ax.set_ylim(pos_flat(0,0)[-1]-4,pos_flat(0,3)[-1]+4)
# ax.set_xlim(pos_flat(0,5)[0]-4,pos_flat(N,2)[0]+4)
# ax.set_yticklabels([])
# ax.set_xticklabels([])
# ax.set_aspect('equal')

# plt.scatter(x,y,s=0, c=proba, cmap= 'inferno_r',vmin=min(proba), vmax=max(proba), facecolors='none')
# plt.colorbar(ax=ax, use_gridspec=True)

# fig.tight_layout()
# fig.savefig(f"{path}/estate2.png", dpi=500, bbox_inches='tight')