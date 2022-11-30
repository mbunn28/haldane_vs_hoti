#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ti
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.gridspec as gs
from tqdm import tqdm

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

points = 500
eigensys = np.zeros((points,points,3,2),dtype=complex)

def u(r,s,n,d):
    bra = eigensys[r%points,s%points,n+1,:]
    if d == 1:
        ket = eigensys[(r+1)%points,s%points,n+1,:]
    if d == 2:
        ket = eigensys[r%points,(s+1)%points,n+1,:]
    result = np.vdot(bra,ket)
    result = result/np.absolute(result)
    return result

l = 0
M = 0.05

r_vals = np.arange(points)
r1,r2 = np.meshgrid(r_vals,r_vals)
b1 = np.array([0,4*np.pi/3])
b2 = (2*np.pi/3)*np.array([np.sqrt(3),1])
kx = (b1[0]*r1 + b2[0]*r2)/points
ky = (b1[1]*r1 + b2[1]*r2)/points

length = 4*np.pi/3
d_area = (0.5*length**2)/(points**2)
k1 = np.zeros((2,points,points))

def hamiltonian(eigensys):
    hamiltonians = np.zeros((points,points,2,2),dtype=complex)
    phi= np.pi/2

    A = -l*(np.sin(np.sqrt(3)*kx)+np.sin(-0.5*(np.sqrt(3)*kx+3*ky))+np.sin(-0.5*(np.sqrt(3)*kx-3*ky)))
    hamiltonians[:,:,0,0] = A + M
    hamiltonians[:,:,1,1] = -A - M
    hamiltonians[:,:,0,1] = (np.cos(ky)+np.cos(0.5*(np.sqrt(3)*kx-ky))+np.cos(0.5*(-np.sqrt(3)*kx-ky)))+1j*(np.sin(ky)+np.sin(0.5*(np.sqrt(3)*kx-ky))+np.sin(0.5*(-np.sqrt(3)*kx-ky)))

    hamiltonians = hamiltonians + np.conjugate(np.swapaxes(hamiltonians,2,3))

    eigvals, eigvec = np.linalg.eigh(hamiltonians)
    eigensys[:,:,0,:] = eigvals
    eigensys[:,:,1:,:] = np.swapaxes(eigvec, 2, 3)

    eigvalz = np.sort(eigvals, axis=2)
    emask = (~np.isclose(np.diff(eigvalz,axis=2),0)).sum(2)+1
    emaskmask = (emask != 2)
    print(np.sum(emaskmask))

    return eigensys


newpath = f"output/chern/haldane/l{l}_res{points}_M{M}"
newpath = newpath.replace('.','')
if not os.path.exists(newpath):
    os.makedirs(newpath)

if os.path.exists(f'{newpath}/chern'):
    imagF = joblib.load(f'{newpath}/chern')
    eigensys = joblib.load(f'{newpath}/eigensys')
else:
    eigensys = hamiltonian(eigensys)
    F = np.zeros((2,points,points),dtype=complex)
    with tqdm(total=2 * points * points) as pbar:
            for n, r, s in np.ndindex(2, points, points):
                F[n,r,s] = -np.log(u(r,s,n,1)*u(r+1,s,n,2)*np.conjugate(u(r,s+1,n,1))*np.conjugate(u(r,s,n,2)))
                while np.imag(F[n,r,s])>np.pi:
                    print('too high')
                    F[n,r,s]=F[n,r,s]-2*np.pi*1j
                while np.imag(F[n,r,s])<=-np.pi:
                    print('too low')
                    F[n,r,s]=F[n,r,s]+2*np.pi*1j
                pbar.update(1)
    imagF = np.imag(F)

    joblib.dump(imagF, f'{newpath}/chern')
    joblib.dump(eigensys, f'{newpath}/eigensys')
    
chern = np.sum(imagF,axis=(1,2))/(2*np.pi)
chernnos = np.round(np.real(chern))
chernnos[chernnos == 0] = 0
print(chernnos)

# print(eigensys[1:5,1:5,0,0])
# print(k1[0,1:5,1:5])

##########################################################################
#                        ENERGY EIGENSTATES PLOT                         #
##########################################################################
fig = plt.figure(figsize=(3.4,4.7))
grd = gs.GridSpec(1,2,figure=fig,wspace=0)
axs = grd.subplots(sharex=True,sharey=True)
emin = np.real(np.amin(eigensys[:,:,0,:]))
im = axs[0].pcolormesh(kx,ky,np.real(eigensys[:,:,0,0]),cmap='Spectral', vmin = emin, vmax=-emin, shading='auto')
axs[0].set_aspect('equal')
axs[0].title.set_text('\n$n=1$')
axs[0].set_ylabel(r'$k_y$')
axs[0].set_xticks([0,np.pi/np.sqrt(3),2*np.pi/np.sqrt(3)])
axs[0].set_xticklabels([0,r'$\frac{\pi}{\sqrt{3}}$',r'$\frac{2 \pi}{\sqrt{3}}$'])
axs[0].set_yticks([0,2*np.pi/3,4*np.pi/3,2*np.pi])
axs[0].set_yticklabels([0,r'$\frac{2\pi}{3}$',r'$\frac{4\pi}{3}$',r'$2\pi$'])
axs[0].set_xlim((-0.1,2*np.pi/np.sqrt(3)+0.1))
axs[0].get_xaxis().majorTicks[2].label1.set_horizontalalignment('right')

axs[1].pcolormesh(kx,ky,np.real(eigensys[:,:,0,1]),cmap='Spectral', vmin = emin, vmax=-emin, shading='auto')
axs[1].set_aspect('equal')
axs[1].title.set_text('\n$n=2$')
# axs[1].set_xlabel(r'$k_x$')
axs[1].set_xticks([0,np.pi/np.sqrt(3),2*np.pi/np.sqrt(3)])
axs[1].set_xticklabels([0,r'$\frac{\pi}{\sqrt{3}}$',r'$\frac{2 \pi}{\sqrt{3}}$'])
axs[1].set_xlim((-0.1,2*np.pi/np.sqrt(3)+0.1))
axs[1].set_ylim((-0.1,2*np.pi+0.1))
axs[1].get_xaxis().majorTicks[0].label1.set_horizontalalignment('left')
# axs[1].get_xaxis().majorTicks[2].label1.set_horizontalalignment('right')
axs[1].yaxis.set_visible(False)

fig.subplots_adjust(top=0.8, bottom=0.2)
cbar_ax = fig.add_axes([0.125, 0.1, 0.775, 0.05])
cbar_ax.title.set_text(r'$k_x$')
cbar_ax.set_xlabel('')
label = cbar_ax.set_ylabel(r'$E$',rotation=0, fontsize=12)
cbar_ax.yaxis.set_label_coords(-0.05,0.06)
fig.colorbar(im, cax=cbar_ax, orientation='horizontal')

fig.savefig(f"{newpath}/energybands.png", dpi =500)


##########################################################################
#                          BERRY CURVATURE PLOT                          #
##########################################################################
fig1 = plt.figure(figsize=(3.4,4.7))
grd = gs.GridSpec(1,2,figure=fig1,wspace=0)
axs1 = grd.subplots(sharex=True,sharey=True)
cmp = 'seismic'#'coolwarm'

lim = np.amax(np.abs(imagF[0:2,:,:]))
im1 = axs1[0].pcolormesh(kx,ky,imagF[0,:,:],cmap=cmp, vmin = -lim, vmax=lim, shading='auto')
axs1[0].set_aspect('equal')
axs1[0].title.set_text(f'$n=1$\n$c_n={chernnos[0]}$')
axs1[0].set_ylabel(r'$k_y$')
axs1[0].set_xticks([0,np.pi/np.sqrt(3),2*np.pi/np.sqrt(3)])
axs1[0].set_xticklabels([0,r'$\frac{\pi}{\sqrt{3}}$',r'$\frac{2 \pi}{\sqrt{3}}$'])
axs1[0].set_yticks([0,2*np.pi/3,4*np.pi/3,2*np.pi])
axs1[0].set_yticklabels([0,r'$\frac{2\pi}{3}$',r'$\frac{4\pi}{3}$',r'$2\pi$'])
axs1[0].set_xlim((-0.1,2*np.pi/np.sqrt(3)+0.1))
axs1[0].set_facecolor('lightgrey')
axs1[0].get_xaxis().majorTicks[2].label1.set_horizontalalignment('right')

axs1[1].pcolormesh(kx,ky,imagF[1,:,:],cmap=cmp, vmin =-lim, vmax=lim, shading='auto')
axs1[1].set_aspect('equal')
axs1[1].title.set_text(f'$n=2$\n$c_n={chernnos[1]}$')
# axs1[1].set_xlabel(r'$k_x$')
axs1[1].set_xticks([0,np.pi/np.sqrt(3),2*np.pi/np.sqrt(3)])
axs1[1].set_xticklabels([0,r'$\frac{\pi}{\sqrt{3}}$',r'$\frac{2 \pi}{\sqrt{3}}$'])
axs1[1].set_xlim((-0.1,2*np.pi/np.sqrt(3)+0.1))
axs1[1].set_ylim((-0.1,2*np.pi+0.1))
axs1[1].set_facecolor('lightgrey')
axs1[1].get_xaxis().majorTicks[0].label1.set_horizontalalignment('left')
# axs1[1].get_xaxis().majorTicks[2].label1.set_horizontalalignment('right')
axs1[1].yaxis.set_visible(False)

fig1.subplots_adjust(top=0.8, bottom=0.2)
cbar_ax1 = fig1.add_axes([0.125, 0.1, 0.775, 0.05])
cbar_ax1.title.set_text(r'$k_x$')
label = cbar_ax1.set_ylabel(r'$F_{12}$',rotation=0, fontsize=12)
cbar_ax1.yaxis.set_label_coords(-0.06,0.12)
cb= fig1.colorbar(im1, cax=cbar_ax1, orientation='horizontal')
cb.formatter.set_powerlimits((0,0))

fig1.savefig(f"{newpath}/chern_no.png", dpi = 500)



###################################################################
#                     SLICES OF BERRY CURVATURE                   #
###################################################################
# fig2, (ax21, ax22, ax23) = plt.subplots(3,1)

# ax21.plot(np.imag(F[0,r_vals,r_vals]))
# ax21.title.set_text('$n=0$')
# ax21.set_ylabel('Flux')
# ax21.set_xticks((0,int(np.round(points/3)),int(np.round(points/2)),int(np.round(2*points/3)),points))
# ax21.set_xticklabels(('$\Gamma$','$K$','$M$','$K\'$','$\Gamma$'))

# ax22.plot(np.imag(F[1,r_vals,r_vals]))
# ax22.title.set_text('$n=1$')
# ax22.set_ylabel('Flux')
# ax22.set_xticks((0,int(np.round(points/3)),int(np.round(points/2)),int(np.round(2*points/3)),points))
# ax22.set_xticklabels(('$\Gamma$','$K$','$M$','$K\'$','$\Gamma$'))

# ax23.plot(np.imag(F[2,r_vals,r_vals]))
# ax23.title.set_text('$n=2$')
# ax23.set_ylabel('Flux')
# ax23.set_xticks((0,int(np.round(points/3)),int(np.round(points/2)),int(np.round(2*points/3)),points))
# ax23.set_xticklabels(('$\Gamma$','$K$','$M$','$K\'$','$\Gamma$'))

# fig2.tight_layout()
# fig2.savefig(f"{newpath}/chern_no_cross.png", bbox_inches = 'tight', dpi = 500)