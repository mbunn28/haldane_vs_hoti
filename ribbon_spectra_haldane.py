#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import ti
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.optimize as optimise
from matplotlib import rc
import matplotlib.gridspec as gs
from tqdm import tqdm
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
plt.rcParams['axes.axisbelow'] = True


def main():
    l = 0.05
    periodic = False
    M = 0
    
    N = 100
    res= 250
    zoom = False
    if zoom == False: 
        N_zoom = 100
        res_zoom = 150


    def hamiltonian(k,N):
        phi = np.pi/2
        A = np.zeros((2,2),dtype=complex)
        A[0,0] = 2*l*np.cos(k+phi)+2*l*np.cos(phi)
        A[1,1] = 2*l*np.cos(k-phi)+2*l*np.cos(phi)
        A[0,1] = np.exp(1j*k)+1
        bigA = np.kron(np.eye(N,dtype=complex),A)

        B = np.zeros((2,2),dtype=complex)
        B[0,0] = l*(np.exp(1j*(phi+k))+np.exp(-1j*phi))
        B[0,1] = 1
        B[1,1] = l*(np.exp(1j*(-phi+k))+np.exp(1j*phi))
        bigB = np.kron(np.eye(N,dtype=complex),B)
        bigB = np.roll(bigB, 2, axis=1)

        if periodic == False:
            bigB[:,:2] = np.zeros((2*N,2),dtype=complex)

        little_M = np.array([[M/2,0],[0,-M/2]])
        bigM = np.kron(np.eye(N,dtype=complex),little_M)

        hamiltonian = bigA + bigB + bigM
        hamiltonian = hamiltonian + np.transpose(np.conjugate(hamiltonian))
        
        return hamiltonian

    newpath = f"output/ribbon_spectra/haldane/res{res}_N{N}_l{l}"
    newpath = newpath.replace('.','')
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    energy_path = f"{newpath}/energies"
    mask_left_path = f"{newpath}/left"
    mask_right_path = f"{newpath}/right"

    k = np.linspace(-np.pi,np.pi,num=res)

    if (os.path.exists(energy_path) and os.path.exists(mask_left_path) and os.path.exists(mask_right_path)):
        energies = joblib.load(energy_path)
        mask_left = joblib.load(mask_left_path)
        mask_right = joblib.load(mask_right_path)
    else:
        energies = np.zeros((res, 2*N))
        mask_left = np.zeros((res, 2*N),dtype=bool)
        mask_right = np.zeros((res, 2*N),dtype=bool)

        for i in tqdm(range(res)):
            energies[i,:], evecs = np.linalg.eigh(hamiltonian(k[i],N))
            evecs = np.transpose(evecs, axes=(1,0))
            waves = np.abs(evecs)**2
            mask_left[i,:] = np.sum(waves[:,:int(np.rint(N))],axis=1) > 0.75
            mask_right[i,:] = np.sum(waves[:,:int(np.rint(N))],axis=1) < 0.25

        energies = -energies
        joblib.dump(energies, energy_path)
        joblib.dump(mask_left, mask_left_path)
        joblib.dump(mask_right, mask_right_path)

    mask_other = np.logical_not(np.logical_or(mask_left,mask_right))

    if zoom == True:
        newpath_zoom = f"output/ribbon_spectra/res{res_zoom}_N{N_zoom}_l{l}_M{M}"
        newpath_zoom = newpath_zoom.replace('.','')
        if not os.path.exists(newpath_zoom):
            os.makedirs(newpath_zoom)
        energy_path_zoom = f"{newpath_zoom}/energies"
        mask_left_path_zoom = f"{newpath_zoom}/left"
        mask_right_path_zoom = f"{newpath_zoom}/right"
        k_zoom = np.linspace(-np.pi,np.pi,num=res_zoom)

        if (os.path.exists(energy_path_zoom) and os.path.exists(mask_left_path_zoom) and os.path.exists(mask_right_path_zoom)):
            energies_zoom = joblib.load(energy_path_zoom)
            mask_left_zoom = joblib.load(mask_left_path_zoom)
            mask_right_zoom = joblib.load(mask_right_path_zoom)
            
        else:
            energies_zoom = np.zeros((res_zoom, 6*N_zoom))
            mask_left_zoom = np.zeros((res_zoom, 6*N_zoom),dtype=bool)
            mask_right_zoom = np.zeros((res_zoom, 6*N_zoom),dtype=bool)

            for i in tqdm(range(res_zoom)):

                energies_zoom[i,:], evecs_zoom = np.linalg.eigh(hamiltonian(k_zoom[i],N_zoom))
                evecs_zoom = np.transpose(evecs_zoom, axes=(1,0))
                waves_zoom = np.abs(evecs_zoom)**2
                mask_left_zoom[i,:] = np.sum(waves_zoom[:,:int(np.rint(3*N_zoom))],axis=1) > 0.7
                mask_right_zoom[i,:] = np.sum(waves_zoom[:,:int(np.rint(3*N_zoom))],axis=1) < 0.2

            joblib.dump(energies_zoom, energy_path_zoom)
            joblib.dump(mask_left_zoom, mask_left_path_zoom)
            joblib.dump(mask_right_zoom, mask_right_path_zoom)

        mask_other_zoom = np.logical_not(np.logical_or(mask_left_zoom,mask_right_zoom))

    _, k =np.meshgrid(np.zeros(2*N),k)

    if zoom == True:
        _, k_zoom =np.meshgrid(np.zeros(6*N_zoom),k_zoom)
        y_max = np.amax(energies_zoom)
        top_aspect = 1
        zoom_y = 0.03
        zoom_aspect = 25
        fig= plt.figure(figsize=(3.4,7.5))
        grd = gs.GridSpec(2,1,figure=fig,hspace=0,height_ratios=[top_aspect*(y_max+0.1),zoom_aspect*zoom_y])
        axs = grd.subplots(sharex=True)

        axs[0].scatter(k[mask_left],energies[mask_left],c='b',s=0.75,linewidth=0.3)
        axs[0].scatter(k[mask_right],energies[mask_right],c='r',s=0.75,linewidth=0.3)
        axs[0].scatter(k[mask_other],energies[mask_other],c='black',s=0.3,linewidth=0.06)
        axs[0].set_ylabel(r'$E$')
        axs[0].set_xlim((-np.pi,np.pi))
        axs[0].set_ylim((-y_max-0.1,y_max+0.1))
        # axs[0].get_yaxis().majorTicks[-1].label1.set_verticalalignment('bottom')

        axs[1].set_ylim((-zoom_y,zoom_y))
        axs[1].scatter(k_zoom[mask_left_zoom],energies_zoom[mask_left_zoom],c='b',s=0.75,linewidth=0.3)
        axs[1].scatter(k_zoom[mask_right_zoom],energies_zoom[mask_right_zoom],c='r',s=0.75,linewidth=0.3)
        axs[1].scatter(k_zoom[mask_other_zoom],energies_zoom[mask_other_zoom],c='black',s=0.3,linewidth=0.06)
        axs[1].set_xlabel(r'$k$')
        axs[1].set_ylabel(r'$E$')
        axs[1].set_xticks((-np.pi,-np.pi/2,0,np.pi/2,np.pi))
        axs[1].set_xticklabels((r'$-\pi$',r'$-\frac{\pi}{2}$',0,r'$\frac{\pi}{2}$',r'$\pi$'))
        axs[1].set_xlim((-np.pi,np.pi))
        # axs[1].get_yaxis().majorTicks[0].label1.set_verticalalignment('top')

        fig_path = f"{newpath}/ribbonspectrum"
        fig.savefig(f"{fig_path}.png", dpi=500, bbox_inches='tight')

    else:
        y_max = np.amax(energies)
        aspect = 0.75
        h = 2
        fig, ax = plt.subplots(figsize=(h/aspect,h))

        ax.set_aspect(aspect)
        ax.scatter(k[mask_left],energies[mask_left],c='b',s=0.75,linewidth=0.3)
        ax.scatter(k[mask_right],energies[mask_right],c='r',s=0.75,linewidth=0.3)
        ax.scatter(k[mask_other],energies[mask_other],c='black',s=0.3,linewidth=0.06)
        ax.set_ylabel(r'$E$')
        ax.set_xlim((-np.pi,np.pi))
        ax.set_ylim((-y_max-0.1,y_max+0.1))
        ax.set_xlabel(r'$k$')
        ax.set_xticks((-np.pi,-np.pi/2,0,np.pi/2,np.pi))
        ax.set_xticklabels((r'$-\pi$',r'$-\frac{\pi}{2}$',0,r'$\frac{\pi}{2}$',r'$\pi$'))

        fig_path = f"{newpath}/ribbonspectrum"
        fig.savefig(f"{fig_path}.png", dpi=500, bbox_inches='tight')

        return

if __name__ == '__main__':
    main()

# mask_other = np.logical_not(np.logical_or(mask_left,mask_right))

# _, k =np.meshgrid(np.zeros(6*N),k)

# fig = plt.figure(figsize=(10,20))
# axs[0] = fig.add_subplot(111)
# axs[0].set_aspect(2)
# # axs[0].set_ylim((-0.25,0.25))
# axs[0].scatter(k[mask_left],energies[mask_left],c='b',s=1)
# axs[0].scatter(k[mask_right],energies[mask_right],c='r',s=1)
# axs[0].scatter(k[mask_other],energies[mask_other],c='black',s=0.5,marker='x',linewidth=0.25)
# axs[0].set_xlabel(r'$ak$')
# axs[0].set_ylabel(r'$E$')
# axs[0].set_xticks((-np.pi,-np.pi/2,0,np.pi/2,np.pi))
# axs[0].set_xticklabels((r'$-\pi$',r'$-\frac{\pi}{2}$',0,r'$\frac{\pi}{2}$',r'$\pi$'))
# fig.tight_layout()

# fig_path = f"{path}/res{res}_N{N}_ribbonspectrum_{aorb_name}{aorb}_{torl_name}{torl}"
# fig.savefig(f"{fig_path}.png", dpi=500, bbox_inches='tight')
