#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 16:46:41 2019

@author: sdiop
"""
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

def find_mode(eigenvalues, E): # finds the index mode which energy is closest to E
    mode = np.argmin(np.abs(eigenvalues-E))
    return(mode)

def pos(i,j,s): # Gives the x,y coordinates of site i,j,s
    A = np.array([sqrt(3)/2 , 1/2])
    B = np.array([-sqrt(3)/2, 1/2])
    C = np.array([0,-1])
    cell =  (i*A + j*B)*3
    if s==0: return(cell+A)
    if s==1: return(cell+A+B)
    if s==2: return(cell+B)
    if s==3: return(cell+B+C)
    if s==4: return(cell+C)
    if s==5: return(cell+C+A)



mode = find_mode(energies,0)
psi = np.transpose(waves)[mode] #wavefunction
proba = np.abs(psi)**2
proba = proba/np.max(proba)


fig, ax = plt.subplots(figsize=(2.4,1.5))
plt.axis('equal')
plt.tight_layout()

cmap = matplotlib.cm.get_cmap('viridis')
normalize = matplotlib.colors.Normalize(vmin=min(proba), vmax=max(proba))
colors = [cmap(normalize(value)) for value in proba]


#plot the probability distribution:

for i in range(N):
    for j in range(N):
        for l in range(6):
            if h[lat(i,j,l),lat(i,j,l)] < 99:
                x, y = pos(i,j,l)
                circle = Circle(pos(i,j,l),0.5,color=colors[lat(i,j,l)],alpha=1,ec=None,zorder=1)
                ax.add_artist(circle)

plt.ylim([-2,30])
plt.xlim([-24,24])
plt.xticks([])
plt.yticks([])

fig.savefig("nice.pdf")
plt.close(fig)
