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

# just a code snippet that makes all the fonts used in the plot LaTeX font
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
plt.rcParams['axes.axisbelow'] = True

#creating the folder where you want to store the output
path = "output/phasediagram/periodic"
if not os.path.exists(path):
            os.makedirs(path)

#the resolution of the energy diagram
res = 300
run = res #ignore this: I use this when I want to create a zoomed in portion of the hase diagram

#Defining my parameter values:
#meshgrid is an easy way to create a Cartesian plane of coords
s_vals = np.linspace(0,1,num=res+1)
ones = np.ones(res)
up = np.append(s_vals, ones)
down = np.append(ones, np.flipud(s_vals))
l,a = np.meshgrid(up,up)
t,b = np.meshgrid(down,down)
#mine are odd bc my paramter space is odd, but a basic one can be made as:
#alpha_vals = np.linspace(min, max, num=res)
#mu_vals = np.linspace(min,max,num=res)
#alpha, mu = np.meshgrid(alpha_vals,mu_vals)
#alpha and mu are two arrays which tell you your alpha and mu vals at each point in a 2d space.
#highly recommend v nifty



# Ignore this: this is just my zoomed in version
# t_min = 0.675
# t_max = 0.725

# tvals = np.linspace(t_min,t_max,num=res)

# b_min = 0.45
# b_max = 0.5

# bvals = np.linspace(b_min,b_max,num=res)

# t,b = np.meshgrid(np.flipud(tvals), np.flipud(bvals))
# ones = np.ones((res,res))
# a = ones
# l = ones




#lattice vectors
b1 = np.array([0,4*np.pi*np.sqrt(3)/9])
b2 = (2*np.pi/9)*np.array([3,np.sqrt(3)])

#reduced zone functions
#the reduced zone is a path through the BZ which I've parameterised. Split into two functions
#input: parameter val -reduced zone path parameter
#       n,m values that tell you the lattice parameter vals
#output: abs min energy for that point along the reduced zone
def reduced_zone1(parameter,n,m):
    kpoint = parameter*(b1+b2)
    energy = hamil(n,m,kpoint)
    return energy

def reduced_zone2(parameter,n,m):
    kpoint = parameter*b1
    energy = hamil(n,m,kpoint)
    return energy


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


    hamiltonians = hamiltonians + np.conjugate(np.swapaxes(hamiltonians,1,2))
    evals = np.abs(np.linalg.eigvalsh(hamiltonians))
    return np.amin(evals, axis=1)



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
    #finding min energy at all of these points
    energies = hamil(n,m,np.array([Kdashpoint, Kpoint,Gamma, Mpoint]))
    #storing the min energy of the lot and where it was 
    min_energy = np.amin(energies)
    lam_val = np.argmin(energies)
    if lam_val ==3:
        lam = 2+np.sqrt(3)/2
    else:
        lam = lam_val
    
    #root finding in reduced BZ, with a bunch of initial guesses
    # note: if you go down this root, you'll want to make sure you have enough initial guesses 
    if min_energy != 0:
        # result = optimise.minimize(reduced_zone1,0.1,args=(n,m), bounds=[(0,2/3)],tol=1e-50)
        # if result.fun < min_energy:
        #     min_energy = result.fun[0]
        #     lam = 3*((2/3) - result.x)
        # result = optimise.minimize(reduced_zone1,0.2,args=(n,m), bounds=[(0,2/3)],tol=1e-50)
        # if result.fun < min_energy:
        #     min_energy = result.fun[0]
        #     lam = 3*((2/3) - result.x)
        # result = optimise.minimize(reduced_zone1,7/18,args=(n,m), bounds=[(0,2/3)],tol=1e-50)
        # if result.fun < min_energy:
        #     min_energy = result.fun[0]
        #     lam = 3*((2/3) - result.x)
        # result = optimise.minimize(reduced_zone1,4/9,args=(n,m), bounds=[(0,2/3)],tol=1e-50)
        # if result.fun < min_energy:
        #     min_energy = result.fun[0]
        #     lam = 3*((2/3) - result.x)
        result = optimise.minimize(reduced_zone1,5/9,args=(n,m), bounds=[(0,2/3)],tol=1e-50)
        if result.fun < min_energy:
            min_energy = result.fun[0]
            lam = 3*((2/3) - result.x)
        result = optimise.minimize(reduced_zone1,11/18,args=(n,m), bounds=[(0,2/3)],tol=1e-50)
        if result.fun < min_energy:
            min_energy = result.fun[0]
            lam = 3*((2/3) - result.x)

        # result = optimise.minimize(reduced_zone2,0.15,args=(n,m), bounds=[(0,1/2)],tol=1e-50)
        # if result.fun < min_energy:
        #     min_energy = result.fun
        #     lam = 2 + np.sqrt(3)*result.x
        # result = optimise.minimize(reduced_zone2,0.35,args=(n,m), bounds=[(0,1/2)],tol=1e-50)
        # if result.fun < min_energy:
        #     min_energy = result.fun
        #     lam = 2 + np.sqrt(3)*result.x
        
    return min_energy, lam 

#Because I knew roughly where the gap closed, I made sure not to compute evals where I didn't need to. 
#this function checks is the lattice parameter vals are in a zone where I want to evaluate the min energy, or not
#input: n,m values that tell you the lattice parameter vals
#output: Boolean True/False, True if it's somewhere I want to evaluate, False otherwise
def check_eval(n,m):
    if (l[n,m]<=0.6) and (b[n,m]==1) and (a[n,m] >= (-3/5)*l[n,m] + 0.3) and (a[n,m] <= -1.5*(l[n,m]-0.1)+1):
        check = True
    elif (l[n,m]>0.6) and (b[n,m] == 1) and (t[n,m]==1) and (a[n,m] >= l[n,m] - 0.75) and (a[n,m] <= (5/8)*l[n,m]-1/8):
        check = True
    elif (t[n,m]<0.1) or ((l[n,m]==1) and (a[n,m] >= -(2/9)*(t[n,m]-0.1)+0.45) and (a[n,m] <= -(5/9)*(t[n,m]-1)+0.5)):
        check = True
    elif (a[n,m]==1) and (l[n,m] ==1) and ((b[n,m] >= -(1/6)*(t[n,m]-1)+0.3) and (b[n,m] <= -(5/9)*(t[n,m]-1)+0.5)):
        check = True
    elif (l[n,m] >= 0.6) and (t[n,m]==1) and (b[n,m] >= 0.75*(l[n,m]-1) + 0.3) and (b[n,m] <= 0.5*(l[n,m]-1)+0.5):
        check = True
    elif (0.3 <= l[n,m] < 0.6) and (b[n,m] <= (6/7)*(l[n,m]-0.25)):
        check = True
    elif (l[n,m] < 0.4) and (a[n,m]==1) and (b[n,m] >= (-5/3)*(l[n,m] -0.2)) and (b[n,m] <= (-61/21)*(l[n,m]-0.1)+1):
        check = True
    else:
        check = False
    return check 


#defining the arrays where I store the min energy and where it was along the red BZ
gap = np.zeros((run,run))
kgap = np.zeros((run,run))
#summing over each point in the discretized BZ
for n in range(0,run):
    for m in range(0,run):

        #a v handy ticker that tells you how far along your calc is
        print(f"{n*(run)+m}/{(run)**2}", end='\r')

        #check if the I want to evaluate the point or not
        check = check_eval(n,m)
        if check == True:
            #if so, find the min energy at that point and where in the red BZ it occurred
            gap[n,m], kgap[n,m] = min_en(n,m)
        if check == False:
            #otherwise write "No" in the array 
            gap[n,m] = np.NaN
            kgap[n,m] = np.NaN

#the values I plot against. You could prob switch this to alpha_vals
x = np.linspace(0,2,num=run)

#deleteing all the values greater than 0.01. Better resolution, and you can drop this down
gap[gap>0.01]= np.NaN

#saves all the data so you can plot again without having to calculate again
joblib.dump(gap, f"{path}/res{res}_gap_topmid")
joblib.dump(kgap, f"{path}/res{res}_kgap_topmid")
joblib.dump(x, f"{path}/res{res}_x")

#plot the data
fig, ax = plt.subplots()
plt.pcolormesh(x,x,gap, norm = colors.LogNorm(), cmap='inferno')
# plt.title(r"Log Scaled Phase Boundary: Periodic, $\Delta$ = 1.7e-3")
ax.grid(linestyle='--')
# ax.set_xlim([0,0.4])
ax.set_aspect(1)

labels=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, r'$\frac{1}{0.8}$',r'$\frac{1}{0.6}$',r'$\frac{1}{0.4}$',r'$\frac{1}{0.2}$',r'$\infty$']
locs=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0,1.2,1.4,1.6,1.8,2.0]
ax.set_yticklabels(labels)
ax.set_yticks(locs)
ax.set_ylabel(r'$\alpha$')

ax.set_xticklabels(labels)
ax.set_xticks(locs)
ax.set_xlabel(r'$\lambda$')

cbar = plt.colorbar(pad = 0.15)
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_title('Energy Gap')

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