#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ti
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.optimize as optimise
from matplotlib import rc
from tqdm import tqdm
import copy
import matplotlib.cm as cm

# just a code snippet that makes all the fonts used in the plot LaTeX font
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
plt.rcParams['axes.axisbelow'] = True

def rule(y):
    a = np.zeros(len(y))
    b = np.zeros(len(y))
    for i in range(len(y)):
        if 0 <= y[i] <= 1:
            a[i] = y[i]
            b[i] = 1
        if y[i] > 1:
            a[i] = 1
            b[i] = 2 - y[i]
    return a, b


def main():
    #creating the folder where you want to store the output
    path = "output/phasediagram/periodic/sixth"
    if not os.path.exists(path):
                os.makedirs(path)

    #the resolution of the energy diagram
    res = 100
    run = res #ignore this: I use this when I want to create a zoomed in portion of the hase diagram

    #Defining my parameter values:
    #meshgrid is an easy way to create a Cartesian plane of coords
    s_vals = np.linspace(0,2,num=run)
    up, down = rule(s_vals)
    l,a = np.meshgrid(up,up)
    t,b = np.meshgrid(down,down)

    M = 0

    #lattice vectors
    b1 = np.array([0,4*np.pi*np.sqrt(3)/9])
    b2 = (2*np.pi/9)*np.array([3,np.sqrt(3)])

    #the periodic Hamiltonian
    #input: n,m values that tell you the lattice parameter vals
    #       kvals = usually a 1 by 2 vector with (k_x, k_y). I have vectorised it though, so you can put in multiple vectors
    #output: abs min eigenvalue of hamiltonian
    def hamil(n,m,kvals):
        if kvals.size != 2:
            kx = kvals[:,0]
            ky = kvals[:,1]
        else:
            kx = kvals[0]
            ky = kvals[1]

        hamiltonians = np.zeros((kx.size,6,6), dtype=complex)
        phi = np.pi/2

        hamiltonians[:,1,0] = t[n,m]*b[n,m]
        hamiltonians[:,5,0] = t[n,m]*b[n,m]
        hamiltonians[:,2,1] = t[n,m]*b[n,m]
        hamiltonians[:,3,2] = t[n,m]*b[n,m]
        hamiltonians[:,4,3] = t[n,m]*b[n,m]
        hamiltonians[:,5,4] = t[n,m]*b[n,m]

        hamiltonians[:,2,0] = l[n,m]*np.exp(-1j*phi)*(b[n,m]+a[n,m]*(np.exp(3*1j*kx)+np.exp(1.5*1j*(kx+ky*np.sqrt(3)))))
        hamiltonians[:,3,0] = t[n,m]*a[n,m]*np.exp(1.5*1j*(kx+ky*np.sqrt(3)))
        hamiltonians[:,4,0] = l[n,m]*np.exp(1j*phi)*(b[n,m]+a[n,m]*(np.exp(1.5*1j*(kx+ky*np.sqrt(3)))+np.exp(-1.5*1j*(kx-ky*np.sqrt(3)))))

        hamiltonians[:,3,1] = l[n,m]*np.exp(-1j*phi)*(b[n,m]+a[n,m]*(np.exp(1.5*1j*(kx+ky*np.sqrt(3)))+np.exp(-1.5*1j*(kx-ky*np.sqrt(3)))))
        hamiltonians[:,4,1] = t[n,m]*a[n,m]*np.exp(-1.5*1j*(kx-ky*np.sqrt(3)))
        hamiltonians[:,5,1] = l[n,m]*np.exp(1j*phi)*(b[n,m]+a[n,m]*(np.exp(-3*1j*kx)+np.exp(-1.5*1j*(kx-ky*np.sqrt(3)))))

        hamiltonians[:,4,2] = l[n,m]*np.exp(-1j*phi)*(b[n,m]+a[n,m]*(np.exp(-3*1j*kx)+np.exp(-1.5*1j*(kx-ky*np.sqrt(3)))))
        hamiltonians[:,5,2] = t[n,m]*a[n,m]*np.exp(-3*1j*kx)

        hamiltonians[:,5,3] = l[n,m]*np.exp(-1j*phi)*(b[n,m]+a[n,m]*(np.exp(-3*1j*kx)+np.exp(-1.5*1j*(kx+ky*np.sqrt(3)))))

        hamiltonians[:,0,0] = M/2
        hamiltonians[:,2,2] = M/2
        hamiltonians[:,4,4] = M/2

        hamiltonians[:,1,1] = -M/2
        hamiltonians[:,3,3] = -M/2
        hamiltonians[:,5,5] = -M/2

        hamiltonians = hamiltonians + np.conjugate(np.swapaxes(hamiltonians,1,2))
        evals = np.linalg.eigvalsh(hamiltonians)
        return evals

    def energy_5(n,m,k):
        en = hamil(n,m,k)
        return en[0,1]

    def energy_6(n,m,k):
        en = hamil(n,m,k)
        return en[0,5]

    #reduced zone functions
    #the reduced zone is a path through the BZ which I've parameterised. Split into two functions
    #input: parameter val -reduced zone path parameter
    #       n,m values that tell you the lattice parameter vals
    #output: abs min energy for that point along the reduced zone
    def reduced_zone1(parameter,n,m,a):
        kpoint = parameter*(b1+b2)
        if a == 5:
            energy = energy_5(n,m,kpoint)
        if a == 6:
            energy = -energy_6(n,m,kpoint)
        return energy

    def reduced_zone2(parameter,n,m,a):
        kpoint = parameter*b1
        if a == 5:
            energy = energy_5(n,m,kpoint)
        if a == 6:
            energy = -energy_6(n,m,kpoint)
        return energy


    #function that numerically solves for the minimum energy in the reduced BZ
    #Function evaluates energy eigenvalues at high symmetry points along the BZ first, 
    #and then runs a root finding algorithm along the red BZ using several starting guesses.
    #input: n,m values that tell you the lattice parameter vals
    #output: the minimum energy in the reduced BZ (according to the root finding)
    #       the path parameter values for the reduced BZ where the min energy occured
    def min_en(n,m):

        #high symmetry points in BZ
        Kpoint = (1/3)*(b1+b2)
        Kdashpoint = (1/3)*(b1+b2)
        Mpoint = (1/2)*b1
        Gamma = np.array([0,0])
        Ks = np.array([Kpoint,Kdashpoint,Mpoint,Gamma])

        max_6 = -100
        min_5 = 100
        for i in range(len(Ks)):
            en_5 = energy_5(n,m,Ks[i,:])
            if en_5 < min_5:
                min_5 = en_5
            en_6 = -energy_6(n,m,Ks[i,:])
            if en_6 > max_6:
                max_6 = en_6
        
        diff_56 = min_5 - max_6
        #root finding in reduced BZ, with a bunch of initial guesses
        # note: if you go down this root, you'll want to make sure you have enough initial guesses 
        if diff_56 > 0:
            result = optimise.minimize(reduced_zone1,5/9,args=(n,m,5), bounds=[(0,2/3)],tol=1e-50)
            if result.fun < min_5:
                min_5 = result.fun
            result = optimise.minimize(reduced_zone1,11/18,args=(n,m,5), bounds=[(0,2/3)],tol=1e-50)
            if result.fun < min_5:
                min_5 = result.fun

            result = optimise.minimize(reduced_zone1,5/9,args=(n,m,6), bounds=[(0,2/3)],tol=1e-50)
            if result.fun > max_6:
                max_6 = -result.fun
            result = optimise.minimize(reduced_zone1,11/18,args=(n,m,6), bounds=[(0,2/3)],tol=1e-50)
            if result.fun > max_6:
                max_6 = -result.fun

            diff_56 = min_5 - max_6
        return diff_56

    #defining the arrays where I store the min energy and where it was along the red BZ
    gap = np.zeros((run,run))
    #summing over each point in the discretized BZ
    with tqdm(total=run * run) as pbar:
        for m, n in np.ndindex(run, run):

            #if so, find the min energy at that point and where in the red BZ it occurred
            gap[n,m] = min_en(n,m)
            pbar.update(1)

    #the values I plot against. You could prob switch this to alpha_vals
    x = np.linspace(0,2,num=run)

    #deleteing all the values greater than 0.01. Better resolution, and you can drop this down

    gap[gap <= 0]= 0
    gap_mask = gap == 0

    #saves all the data so you can plot again without having to calculate again
    joblib.dump(gap_mask, f"{path}/res{res}_gapmask")
    joblib.dump(gap, f"{path}/res{res}_gap")
    joblib.dump(x, f"{path}/res{res}_x")

    #plot the data
    fig, ax = plt.subplots(figsize=(3.4,3.4))
    my_cmap = copy.copy(cm.get_cmap('inferno'))
    my_cmap.set_bad('k')

    im = plt.pcolormesh(x,x,gap, norm = colors.LogNorm(), cmap=my_cmap)
    # plt.title(r"Log Scaled Phase Boundary: Periodic, $\Delta$ = 1.7e-3")
    ax.grid(linestyle='--')
    # ax.set_xlim([0,0.4])
    ax.set_aspect(1)

    labels=[0.0, 0.5, 1.0,r'$\frac{1}{0.5}$',r'$\infty$']
    locs=[0.0, 0.5, 1.0,1.5,2.0]
    ax.set_yticklabels(labels)
    ax.set_yticks(locs)
    ax.set_ylabel(r'$\alpha$')

    ax.set_xticklabels(labels)
    ax.set_xticks(locs)
    ax.set_xlabel(r'$\lambda$')

    cbar = ti.colorbar(im) #pad = 0.15)
    # cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_title('Gap')

    plt.gcf().subplots_adjust(top=0.85)
    fig.tight_layout()
    fig.savefig(f"{path}/periodicphasediagram.png", dpi=500)


    #ignore this: this just plotted where in the BZ the gap closing occured
    # vmax = (2 + np.sqrt(3)/2)
    # kgap[gap>0.01] = np.NaN
    # cmap1 = colors.LinearSegmentedColormap.from_list('mycmap', [(0/vmax,    '#984ea3'), (0.5/vmax,    '#e41a1c'), (1/vmax, '#4daf4a'), (2/vmax, '#377eb8'), ((2 + np.sqrt(3)/2)/vmax,    '#e41a1c')], N=256)
    # fig1, ax = plt.subplots()
    # im = ax.pcolormesh(x,x,kgap, cmap=cmap1, vmin=0, vmax=vmax)
    # cbar = fig1.colorbar(im)
    # ax.set(aspect=1)
    # cbar.set_ticks([0,0.5,1,2,vmax]) # Integer colorbar tick locations
    # cbar.set_ticklabels(["K\'", "M", "K", "$\Gamma$","M"])
    # fig1.savefig(f"{path}/kpointstopmid.png", dpi=500)
    
    return

if __name__ == "__main__":
    main()