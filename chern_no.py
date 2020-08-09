#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ti
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.linalg
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

points = 250
eigensys = np.zeros((points,points,7,6),dtype=complex)

def u(r,s,n,d):
    bra = eigensys[r%points,s%points,n+1,:]
    if d == 1:
        ket = eigensys[(r+1)%points,s%points,n+1,:]
    if d == 2:
        ket = eigensys[r%points,(s+1)%points,n+1,:]
    result = np.vdot(bra,ket)
    result = result/np.absolute(result)
    return result

#Defined the lattice - 
lattice = ti.Lattice(
PBC_i = True,
PBC_j = True,
Corners = False,
alpha = 0.1,
hal = 0.4,
M=0,
N=4)

lattice.large_hal = True
lattice.large_alpha = False

b1 = np.array([0,4*np.pi*np.sqrt(3)/9])
b2 = (2*np.pi/9)*np.array([3,np.sqrt(3)])
length = 4*np.pi/np.sqrt(3)
d_area = (0.5*length**2)/(points**2)
k1 = np.zeros((2,points,points))

for r in range(0,points):
    for s in range(0, points):
    
        k = (r*b1 + s*b2)/(points)
        k1[:,r,s] = k
        lattice.initialize_periodic_hamiltonian(k)
        w, v = scipy.linalg.eigh(lattice.periodic_hamiltonian)
        idx = np.argsort(w)
        w = w[idx]
        v = v[:,idx].transpose()
        eigensys[r,s,0,:] = w
        eigensys[r,s,1:7,:] = v

chern = np.zeros(6,dtype=complex)
F = np.zeros((6,points,points),dtype=complex)
for n in range(0,6):
    for r in range(0,points):
        for s in range(0, points):
            F[n,r,s] = np.log(u(r,s,n,1)*u(r+1,s,n,2)*np.conjugate(u(r,s+1,n,1))*np.conjugate(u(r,s,n,2)))
    chern[n] = np.sum(np.imag(F[n,:,:]))/(2*np.pi)

print(chern)

chernnos = np.round(np.real(chern))
print(chernnos)
if lattice.large_alpha == True:
    a = "b"
else:
    a = "a"

if lattice.large_hal == True:
    t = "t"
else:
    t = "l"

newpath = f"output/chern/{a}{lattice.hal}_{t}{lattice.hal}_res{points}"
if not os.path.exists(newpath):
            os.makedirs(newpath)

fig, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.pcolormesh(k1[0,:,:],k1[1,:,:],np.real(eigensys[:,:,0,0]),cmap='plasma', vmin = np.real(np.amin(eigensys[:,:,0,:])), vmax=0)
ax1.set_aspect('equal')
ax1.title.set_text(r'$n=1$')
ax1.set_ylabel(r'$k_y$')
ax1.set_xlabel(r'$k_x$')

ax2.pcolormesh(k1[0,:,:],k1[1,:,:],np.real(eigensys[:,:,0,1]),cmap='plasma', vmin = np.real(np.amin(eigensys[:,:,0,:])), vmax=0)
ax2.set_aspect('equal')
ax2.title.set_text(r'$n=2$')
ax2.set_ylabel(r'$k_y$')
ax2.set_xlabel(r'$k_x$')

im = ax3.pcolormesh(k1[0,:,:],k1[1,:,:],np.real(eigensys[:,:,0,2]),cmap='plasma', vmin = np.real(np.amin(eigensys[:,:,0,:])), vmax=0)
ax3.set_aspect('equal')
ax3.title.set_text(r'$n=3$')
ax3.set_ylabel(r'$k_y$')
ax3.set_xlabel(r'$k_x$')

fig.tight_layout()
fig.text(0.5,0.88,"Energy Bands",horizontalalignment='center',fontsize=16)
fig.subplots_adjust(top=0.8, bottom=0.2)
cbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.05])
cbar_ax.set_ylabel('E',rotation=0, fontsize=12,labelpad=10)
fig.colorbar(im, cax=cbar_ax, orientation='horizontal')

fig.savefig(f"{newpath}/energybands.png", bbox_inches='tight', dpi =1200)

fig, (ax1, ax2, ax3) = plt.subplots(1,3)
# fig.suptitle("Berry Flux")
lim = np.amax(np.abs(np.imag(F[0:2,:,:])))
ax1.pcolormesh(k1[0,:,:],k1[1,:,:],np.imag(F[0,:,:]),cmap='coolwarm', vmin = -lim, vmax=lim)
ax1.set_aspect('equal')
ax1.title.set_text(f'$n=1$\n$c_n=${chernnos[0]}')
ax1.set_ylabel(r'$k_y$')
ax1.set_xlabel(r'$k_x$')

ax2.pcolormesh(k1[0,:,:],k1[1,:,:],np.imag(F[1,:,:]),cmap='coolwarm', vmin =-lim, vmax=lim)
ax2.set_aspect('equal')
ax2.title.set_text(f'$n=2$\n$c_n=${chernnos[1]}')
ax2.set_ylabel(r'$k_y$')
ax2.set_xlabel(r'$k_x$')

im = ax3.pcolormesh(k1[0,:,:],k1[1,:,:],np.imag(F[2,:,:]),cmap='coolwarm', vmin = -lim, vmax=lim)
ax3.set_aspect('equal')
ax3.title.set_text(f'$n=3$\n$c_n$={chernnos[2]}')
ax3.set_ylabel(r'$k_y$')
ax3.set_xlabel(r'$k_x$')

fig.tight_layout()
# fig.text(0.5,0.88,"Berry Flux Distribution over Brillouin Zone",horizontalalignment='center',fontsize=16)
fig.subplots_adjust(bottom=0.15)
cbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.05])
cbar_ax.title.set_text('Flux')
fig.colorbar(im, cax=cbar_ax, orientation='horizontal')

fig.savefig(f"{newpath}/chern_no.png", bbox_inches = 'tight', dpi = 1200)