#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ti
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rc
import matplotlib.gridspec as gs
from tqdm import tqdm

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

points = 500
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

l = 1
t = 0.2

a = 1
b = 0.5595

r_vals = np.arange(points)
r1,r2 = np.meshgrid(r_vals,r_vals)
b1 = np.array([0,4*np.pi*np.sqrt(3)/9])
b2 = (2*np.pi/9)*np.array([3,np.sqrt(3)])
kx = (b1[0]*r1 + b2[0]*r2)/points
ky = (b1[1]*r1 + b2[1]*r2)/points

length = 4*np.pi/np.sqrt(3)
d_area = (0.5*length**2)/(points**2)
k1 = np.zeros((2,points,points))

def hamiltonian(eigensys):
    hamiltonians = np.zeros((points,points,6,6),dtype=complex)
    phi= np.pi/2

    hamiltonians[:,:,2,0] = l*np.exp(-1j*phi)*(b+a*(np.exp(3*1j*kx)+np.exp(1.5*1j*(kx+ky*np.sqrt(3)))))
    hamiltonians[:,:,3,0] = t*a*np.exp(1.5*1j*(kx+ky*np.sqrt(3)))
    hamiltonians[:,:,4,0] = l*np.exp(1j*phi)*(b+a*(np.exp(1.5*1j*(kx+ky*np.sqrt(3)))+np.exp(-1.5*1j*(kx-ky*np.sqrt(3)))))

    hamiltonians[:,:,3,1] = l*np.exp(-1j*phi)*(b+a*(np.exp(1.5*1j*(kx+ky*np.sqrt(3)))+np.exp(-1.5*1j*(kx-ky*np.sqrt(3)))))
    hamiltonians[:,:,4,1] = t*a*np.exp(-1.5*1j*(kx-ky*np.sqrt(3)))
    hamiltonians[:,:,5,1] = l*np.exp(1j*phi)*(b+a*(np.exp(-3*1j*kx)+np.exp(-1.5*1j*(kx-ky*np.sqrt(3)))))

    hamiltonians[:,:,4,2] = l*np.exp(-1j*phi)*(b+a*(np.exp(-3*1j*kx)+np.exp(-1.5*1j*(kx-ky*np.sqrt(3)))))
    hamiltonians[:,:,5,2] = t*a*np.exp(-3*1j*kx)

    hamiltonians[:,:,5,3] = l*np.exp(-1j*phi)*(b+a*(np.exp(-3*1j*kx)+np.exp(-1.5*1j*(kx+ky*np.sqrt(3)))))
    
    hamiltonians[:,:,1,0] = b*t
    hamiltonians[:,:,2,1] = b*t
    hamiltonians[:,:,3,2] = b*t
    hamiltonians[:,:,4,3] = b*t
    hamiltonians[:,:,5,4] = b*t
    hamiltonians[:,:,0,5] = b*t

    hamiltonians = hamiltonians + np.conjugate(np.swapaxes(hamiltonians,2,3))

    eigvals, eigvec = np.linalg.eigh(hamiltonians)
    eigensys[:,:,0,:] = eigvals
    eigensys[:,:,1:7,:] = np.swapaxes(eigvec, 2, 3)

    eigvalz = np.sort(eigvals, axis=2)
    emask = (~np.isclose(np.diff(eigvalz,axis=2),0)).sum(2)+1
    emaskmask = (emask != 6)
    print(np.sum(emaskmask))

    return eigensys

if a ==1:
    alphname = "b"
    alph = b
else:
    alphname = "a"
    alph = a

if l == 1:
    halname = "t"
    hal = t
else:
    halname = "l"
    hal = l

newpath = f"output/chern/{halname}{hal}_{alphname}{alph}_res{points}"
if not os.path.exists(newpath):
    os.makedirs(newpath)

if os.path.exists(f'{newpath}/chern'):
    imagF = joblib.load(f'{newpath}/chern')
    eigensys = joblib.load(f'{newpath}/eigensys')
else:
    eigensys = hamiltonian(eigensys)
    F = np.zeros((6,points,points),dtype=complex)
    with tqdm(total=6 * points * points) as pbar:
            for n, r, s in np.ndindex(6, points, points):
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
fig = plt.figure(figsize=(3.4,3.4))
grd = gs.GridSpec(1,3,figure=fig,wspace=0)
axs = grd.subplots(sharex=True,sharey=True)

axs[0].pcolormesh(kx,ky,np.real(eigensys[:,:,0,0]),cmap='plasma', vmin = np.real(np.amin(eigensys[:,:,0,:])), vmax=0, shading='auto')
axs[0].set_aspect('equal')
axs[0].title.set_text(r'$n=1$')
axs[0].set_ylabel(r'$k_y$')
axs[0].set_xticks([0,np.pi/3,2*np.pi/3])
axs[0].set_xticklabels([0,r'$\frac{\pi}{3}$',r'$\frac{2 \pi}{3}$'])
axs[0].set_yticks([0,2*np.pi/(3*np.sqrt(3)),4*np.pi/(3*np.sqrt(3)),2*np.pi/(np.sqrt(3))])
axs[0].set_yticklabels([0,r'$\frac{2\pi}{3\sqrt{3}}$',r'$\frac{4\pi}{3\sqrt{3}}$',r'$\frac{2\pi}{\sqrt{3}}$'])
axs[0].set_xlim((-0.1,2*np.pi/3+0.1))
axs[0].get_xaxis().majorTicks[2].label1.set_horizontalalignment('right')

axs[1].pcolormesh(kx,ky,np.real(eigensys[:,:,0,1]),cmap='plasma', vmin = np.real(np.amin(eigensys[:,:,0,:])), vmax=0, shading='auto')
axs[1].set_aspect('equal')
axs[1].title.set_text(r'$n=2$')
axs[1].set_xlabel(r'$k_x$')
axs[1].set_xticks([0,np.pi/3,2*np.pi/3])
axs[1].set_xticklabels([0,r'$\frac{\pi}{3}$',r'$\frac{2 \pi}{3}$'])
axs[1].set_xlim((-0.1,2*np.pi/3+0.1))
axs[1].get_xaxis().majorTicks[0].label1.set_horizontalalignment('left')
axs[1].get_xaxis().majorTicks[2].label1.set_horizontalalignment('right')
axs[1].yaxis.set_visible(False)

im = axs[2].pcolormesh(kx,ky,np.real(eigensys[:,:,0,2]),cmap='plasma', vmin = np.real(np.amin(eigensys[:,:,0,:])), vmax=0, shading='auto')
axs[2].set_aspect('equal')
axs[2].title.set_text(r'$n=3$')
axs[2].set_xticks([0,np.pi/3,2*np.pi/3])
axs[2].set_xticklabels([0,r'$\frac{\pi}{3}$',r'$\frac{2 \pi}{3}$'])
axs[2].set_xlim((-0.1,2*np.pi/3+0.1))
axs[2].set_ylim((-0.1,2*np.pi/(np.sqrt(3))+0.1))
axs[2].yaxis.set_visible(False)
axs[2].get_xaxis().majorTicks[0].label1.set_horizontalalignment('left')

fig.subplots_adjust(top=0.8, bottom=0.2)
cbar_ax = fig.add_axes([0.125, 0.1, 0.775, 0.05])
label = cbar_ax.set_ylabel(r'$E$',rotation=0, fontsize=12)
cbar_ax.yaxis.set_label_coords(-0.05,0.06)
fig.colorbar(im, cax=cbar_ax, orientation='horizontal')

fig.savefig(f"{newpath}/energybands.png", bbox_inches='tight', dpi =500)


##########################################################################
#                          BERRY CURVATURE PLOT                          #
##########################################################################
fig1 = plt.figure(figsize=(3.4,3.4))
grd = gs.GridSpec(1,3,figure=fig1,wspace=0)
axs1 = grd.subplots(sharex=True,sharey=True)
cmp = 'seismic'#'coolwarm'

lim = np.amax(np.abs(imagF[0:2,:,:]))
axs1[0].pcolormesh(kx,ky,imagF[0,:,:],cmap=cmp, vmin = -lim, vmax=lim, shading='auto')
axs1[0].set_aspect('equal')
axs1[0].title.set_text(f'$n=1$\n$c_n={chernnos[0]}$')
axs1[0].set_ylabel(r'$k_y$')
axs1[0].set_xticks([0,np.pi/3,2*np.pi/3])
axs1[0].set_xticklabels([0,r'$\frac{\pi}{3}$',r'$\frac{2 \pi}{3}$'])
axs1[0].set_yticks([0,2*np.pi/(3*np.sqrt(3)),4*np.pi/(3*np.sqrt(3)),2*np.pi/(np.sqrt(3))])
axs1[0].set_yticklabels([0,r'$\frac{2\pi}{3\sqrt{3}}$',r'$\frac{4\pi}{3\sqrt{3}}$',r'$\frac{2\pi}{\sqrt{3}}$'])
axs1[0].set_xlim((-0.1,2*np.pi/3+0.1))
axs1[0].get_xaxis().majorTicks[2].label1.set_horizontalalignment('right')

axs1[1].pcolormesh(kx,ky,imagF[1,:,:],cmap=cmp, vmin =-lim, vmax=lim, shading='auto')
axs1[1].set_aspect('equal')
axs1[1].title.set_text(f'$n=2$\n$c_n={chernnos[1]}$')
axs1[1].set_xlabel(r'$k_x$')
axs1[1].set_xticks([0,np.pi/3,2*np.pi/3])
axs1[1].set_xticklabels([0,r'$\frac{\pi}{3}$',r'$\frac{2 \pi}{3}$'])
axs1[1].set_xlim((-0.1,2*np.pi/3+0.1))
axs1[1].get_xaxis().majorTicks[0].label1.set_horizontalalignment('left')
axs1[1].get_xaxis().majorTicks[2].label1.set_horizontalalignment('right')
axs1[1].yaxis.set_visible(False)

im1 = axs1[2].pcolormesh(kx,ky,imagF[2,:,:],cmap=cmp, vmin =-lim, vmax=lim, shading='auto')
axs1[2].set_aspect('equal')
axs1[2].title.set_text(f'$n=3$\n$c_n={chernnos[2]}$')
axs1[2].set_xticks([0,np.pi/3,2*np.pi/3])
axs1[2].set_xticklabels([0,r'$\frac{\pi}{3}$',r'$\frac{2 \pi}{3}$'])
axs1[2].set_xlim((-0.1,2*np.pi/3+0.1))
axs1[2].set_ylim((-0.1,2*np.pi/(np.sqrt(3))+0.1))
axs1[2].yaxis.set_visible(False)
axs1[2].get_xaxis().majorTicks[0].label1.set_horizontalalignment('left')

fig1.subplots_adjust(top=0.8, bottom=0.2)
cbar_ax1 = fig1.add_axes([0.125, 0.1, 0.775, 0.05])
label = cbar_ax1.set_ylabel(r'$F_{12}$',rotation=0, fontsize=12)
cbar_ax1.yaxis.set_label_coords(-0.06,0.12)
cb= fig1.colorbar(im1, cax=cbar_ax1, orientation='horizontal')
cb.formatter.set_powerlimits((0,0))

fig1.savefig(f"{newpath}/chern_no.png", bbox_inches = 'tight', dpi = 500)



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