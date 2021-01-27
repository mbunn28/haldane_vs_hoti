#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ti
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rc
import scipy.linalg
import numpy.random
import numpy.ma

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble= r'\usepackage{amsfonts}')

path = "output/zq"
if not os.path.exists(path):
            os.makedirs(path)


l = 0.01
t = 1

a = 0.4
b = 1

N = 5
mass = 0

PBC_i = True
PBC_j = True
Corners = False
location = np.array([3,3],dtype=int)



def plot(ax, x1, x2,col='k'):
    if x1[0] == x2[0]:
        if x1[1] < x2[1]:
            ax.vlines(x=x1[0],ymin=x1[1],ymax=x2[1],color=col)
        else:
            ax.vlines(x=x1[0],ymin=x2[1],ymax=x1[1],color=col)
    else:
        grad = (x1[1]-x2[1])/(x1[0]-x2[0])
        if x1[0] < x2[0]:
            x = np.linspace(x1[0],x2[0])
        else:
            x = np.linspace(x2[0],x1[0])  
        y = grad*(x-x1[0])+x1[1]
        ax.plot(x,y,c=col)  
    return

def pos(i,j,s): # Gives the x,y coordinates of site i,j,s
    A = np.array([np.sqrt(3)/2 , -1/2])
    B = np.array([-np.sqrt(3)/2, -1/2])
    C = np.array([0,1])
    cell =  (i*C + j*A)*3
    if s==0: return(cell+C)
    if s==1: return(cell+A+C)
    if s==2: return(cell+A)
    if s==3: return(cell+A+B)
    if s==4: return(cell+B)
    if s==5: return(cell+B+C)

def lat(i,j,s): return(6*N*i+6*j+s)



def twist_hamiltonian(the, zq):
    vv = 100000
    h = np.zeros((6*(N)**2,6*(N)**2), dtype = complex)
    fig,ax=plt.subplots()
    
    for i in range(N):
        for j in range(N):
            if (i == location[0] and j == location[1] and zq == 'z6'):
                the5 = -np.sum(the)
                h[lat(i,j,0), lat(i,j,1)] = -t*b*np.exp(-1j*the[1])
                plot(ax,pos(i,j,0), pos(i,j,1), col='b')
                h[lat(i,j,0), lat(i,j,5)] = -t*b*np.exp(1j*the[0])
                plot(ax,pos(i,j,0), pos(i,j,5), col='b')
                h[lat(i,j,0), lat((i+1)%N,j,3)] = -t*a
                plot(ax,pos(i,j,0), pos((i+1),j,3),col='y')

                h[lat(i,j,2), lat(i,j,1)] = -t*b*np.exp(1j*the[2])
                plot(ax,pos(i,j,2), pos(i,j,1), col='b')
                h[lat(i,j,2), lat(i,j,3)] = -t*b*np.exp(-1j*the[3])
                plot(ax,pos(i,j,2), pos(i,j,3), col='b')
                h[lat(i,j,2), lat(i,(j+1)%N,5)] = -t*a
                plot(ax,pos(i,j,2), pos(i,(j+1),5),col='y')

                h[lat(i,j,4), lat(i,j,3)] = -t*b*np.exp(1j*the[4])
                plot(ax,pos(i,j,4), pos(i,j,3), col='b')
                h[lat(i,j,4), lat(i,j,5)] = -t*b*np.exp(-1j*the5)
                plot(ax,pos(i,j,4), pos(i,j,5), col='b')
                h[lat(i,j,4), lat((i-1)%N,(j-1)%N,1)] = -t*a
                plot(ax,pos(i,j,4), pos((i-1),(j-1),1),col='y')

                h[lat(i,j,0), lat(i,j,4)] = -1j*l*b*np.exp(-1j*(the[4]+the[3]+the[2]+the[1]))
                plot(ax,pos(i,j,0), pos(i,j,4), col='b')
                h[lat(i,j,1), lat(i,j,5)] = -1j*l*b*np.exp(-1j*(the[1]+the[0]))
                plot(ax,pos(i,j,1), pos(i,j,5), col='b')
                h[lat(i,j,2), lat(i,j,0)] = -1j*l*b*np.exp(-1j*(the[2]+the[1]))
                plot(ax,pos(i,j,2), pos(i,j,0), col='b')
                h[lat(i,j,3), lat(i,j,1)] = -1j*l*b*np.exp(-1j*(the[3]+the[2]))
                plot(ax,pos(i,j,3), pos(i,j,1), col='b')
                h[lat(i,j,4), lat(i,j,2)] = -1j*l*b*np.exp(-1j*(the[4]+the[3]))
                plot(ax,pos(i,j,4), pos(i,j,2), col='b')
                h[lat(i,j,5), lat(i,j,3)] = -1j*l*b*np.exp(-1j*(the[3]+the[2]+the[1]+the[0]))
                plot(ax,pos(i,j,5), pos(i,j,3), col='b')

            elif (i == location[0] and j == location[1] and zq == 'z2'):
                h[lat(i,j,0), lat(i,j,1)] = -t*b
                plot(ax,pos(i,j,0), pos(i,j,1))
                h[lat(i,j,0), lat(i,j,5)] = -t*b
                plot(ax,pos(i,j,0), pos(i,j,5))
                h[lat(i,j,0), lat((i+1)%N,j,3)] = -t*a
                plot(ax,pos(i,j,0), pos((i+1),j,3))

                h[lat(i,j,2), lat(i,j,1)] = -t*b
                plot(ax,pos(i,j,2), pos(i,j,1))
                h[lat(i,j,2), lat(i,j,3)] = -t*b
                plot(ax,pos(i,j,2), pos(i,j,3))
                h[lat(i,j,2), lat(i,(j+1)%N,5)] = -t*a*np.exp(1j*the)
                plot(ax,pos(i,j,2), pos(i,(j+1),5), col='b')

                h[lat(i,j,4), lat(i,j,3)] = -t*b
                plot(ax,pos(i,j,4), pos(i,j,3))
                h[lat(i,j,4), lat(i,j,5)] = -t*b
                plot(ax,pos(i,j,4), pos(i,j,5))
                h[lat(i,j,4), lat((i-1)%N,(j-1)%N,1)] = -t*a
                plot(ax,pos(i,j,4), pos((i-1),(j-1),1))

                h[lat(i,j,0), lat(i,j,4)] = -1j*l*b
                plot(ax,pos(i,j,0), pos(i,j,4))
                h[lat(i,j,1), lat(i,j,5)] = -1j*l*b
                plot(ax,pos(i,j,1), pos(i,j,5))
                h[lat(i,j,2), lat(i,j,0)] = -1j*l*b
                plot(ax,pos(i,j,2), pos(i,j,0))
                h[lat(i,j,3), lat(i,j,1)] = -1j*l*b
                plot(ax,pos(i,j,3), pos(i,j,1))
                h[lat(i,j,4), lat(i,j,2)] = -1j*l*b
                plot(ax,pos(i,j,4), pos(i,j,2))
                h[lat(i,j,5), lat(i,j,3)] = -1j*l*b
                plot(ax,pos(i,j,5), pos(i,j,3))

            else:
                h[lat(i,j,0), lat(i,j,1)] = -t*b
                plot(ax,pos(i,j,0), pos(i,j,1))
                h[lat(i,j,0), lat(i,j,5)] = -t*b
                plot(ax,pos(i,j,0), pos(i,j,5))
                h[lat(i,j,0), lat((i+1)%N,j,3)] = -t*a
                plot(ax,pos(i,j,0), pos((i+1),j,3),col='y')

                h[lat(i,j,2), lat(i,j,1)] = -t*b
                plot(ax,pos(i,j,2), pos(i,j,1))
                h[lat(i,j,2), lat(i,j,3)] = -t*b
                plot(ax,pos(i,j,2), pos(i,j,3))
                h[lat(i,j,2), lat(i,(j+1)%N,5)] = -t*a
                plot(ax,pos(i,j,2), pos(i,(j+1),5),col='y')

                h[lat(i,j,4), lat(i,j,3)] = -t*b
                plot(ax,pos(i,j,4), pos(i,j,3))
                h[lat(i,j,4), lat(i,j,5)] = -t*b
                plot(ax,pos(i,j,4), pos(i,j,5))
                h[lat(i,j,4), lat((i-1)%N,(j-1)%N,1)] = -t*a
                plot(ax,pos(i,j,4), pos((i-1),(j-1),1),col='y')

                h[lat(i,j,0), lat(i,j,4)] = -1j*l*b
                plot(ax,pos(i,j,0), pos(i,j,4))
                h[lat(i,j,1), lat(i,j,5)] = -1j*l*b
                plot(ax,pos(i,j,1), pos(i,j,5))
                h[lat(i,j,2), lat(i,j,0)] = -1j*l*b
                plot(ax,pos(i,j,2), pos(i,j,0))
                h[lat(i,j,3), lat(i,j,1)] = -1j*l*b
                plot(ax,pos(i,j,3), pos(i,j,1))
                h[lat(i,j,4), lat(i,j,2)] = -1j*l*b
                plot(ax,pos(i,j,4), pos(i,j,2))
                h[lat(i,j,5), lat(i,j,3)] = -1j*l*b
                plot(ax,pos(i,j,5), pos(i,j,3))

            if N !=1:
                h[lat(i,j,0), lat((i+1)%N,j,4)] = -1j*l*a
                plot(ax,pos(i,j,0), pos(i+1,j,4),col='r')
                h[lat(i,j,0), lat((i+1)%N,(j+1)%N,4)] = -1j*l*a
                plot(ax,pos(i,j,0), pos(i+1,j+1,4),col='r')

                h[lat(i,j,1), lat((i+1)%N,(j+1)%N,5)] = -1j*l*a
                plot(ax,pos(i,j,1), pos(i+1,j+1,5),col='r')
                h[lat(i,j,1), lat(i,(j+1)%N,5)] = -1j*l*a
                plot(ax,pos(i,j,1), pos(i,j+1,5),col='r')

                h[lat(i,j,2), lat(i,(j+1)%N,0)] = -1j*l*a
                plot(ax,pos(i,j,2), pos(i,j+1,0),col='r')
                h[lat(i,j,2), lat((i-1)%N,j,0)] = -1j*l*a
                plot(ax,pos(i,j,2), pos(i-1,j,0),col='r')

                h[lat(i,j,3), lat((i-1)%N,j,1)] = -1j*l*a
                plot(ax,pos(i,j,3), pos(i-1,j,1),col='r')
                h[lat(i,j,3), lat((i-1)%N,(j-1)%N,1)] = -1j*l*a
                plot(ax,pos(i,j,3), pos(i-1,j-1,1),col='r')

                h[lat(i,j,4), lat((i-1)%N,(j-1)%N,2)] = -1j*l*a
                plot(ax,pos(i,j,4), pos(i-1,j-1,2),col='r')
                h[lat(i,j,4), lat(i,(j-1)%N,2)] = -1j*l*a
                plot(ax,pos(i,j,4), pos(i,j-1,2),col='r')

                h[lat(i,j,5), lat(i,(j-1)%N,3)] = -1j*l*a
                plot(ax,pos(i,j,5), pos(i,j-1,3),col='r')
                h[lat(i,j,5), lat((i+1)%N,j,3)] = -1j*l*a
                plot(ax,pos(i,j,5), pos(i+1,j,3),col='r')

            for s in [0,2,4]:
                h[lat(i,j,s), lat(i,j,s)] = +mass/2
            for s in [1,3,5]:
                h[lat(i,j,s), lat(i,j,s)] = -mass/2



    if PBC_i == False:
        for j in range(N):
            h[lat(N-1,j,0), lat(0,j,3)] = 0
            h[lat(0,j,4), lat(N-1,(j-1)%N,1)] = 0

            h[lat(0,(j+1)%N,3), lat(N-1,j,1)] = 0
            h[lat(N-1,j,0), lat(0,j,4)] = 0
            h[lat(0,(j+1)%N,4), lat(N-1,j,2)] = 0
            h[lat(N-1,j,5), lat(0,j,3)] = 0

            h[lat(N-1,j,1), lat(0,(j+1)%N,5)] = 0
            h[lat(N-1,j,0), lat(0,(j+1)%N,4)] = 0
            h[lat(0,j,3), lat(N-1,j,1)] = 0
            h[lat(0,j,2), lat(N-1,j,0)] = 0

    if PBC_j == False:
        for i in range(N):
            h[lat(i,N-1,2), lat(i,0,5)] = 0
            h[lat(i,0,4), lat((i-1)%N,N-1,1)] = 0

            h[lat((i+1)%N,0,3), lat(i,N-1,1)] = 0
            h[lat(i,N-1,1), lat(i,0,5)] = 0
            h[lat((i+1)%N,0,4), lat(i,N-1,2)] = 0
            h[lat(i,N-1,2), lat(i,0,0)] = 0

            h[lat(i,N-1,1), lat((i+1)%N,0,5)] = 0
            h[lat(i,0,5), lat(i,N-1,3)] = 0
            h[lat(i,N-1,0), lat((i+1)%N,0,4)] = 0
            h[lat(i,0,4), lat(i,N-1,2)] = 0

    #dimer geometry
    if PBC_j == False and Corners == True:
        for i in range(0,N):
            for s in [0,3,4,5]:
                h[lat(i,0,s),lat(i,0,s)] = vv

            for s in [0,1,2,3]:
                h[lat(i,N-1,s),lat(i,N-1,s)] = vv


    if PBC_i==False and Corners == True:
        for j in range(0,N):
            for s in [2,3,4,5]:
                h[lat(0,j,s),lat(0,j,s)] = vv

            for s in [0,1,2,5]:
                h[lat(N-1,j,s),lat(N-1,j,s)] = vv

    h = np.conjugate(h.transpose()) + h

    ener, evecs = scipy.linalg.eigh(h)
    if min(np.abs(ener)) < 1e-2:
        print('energy very small!\n')

    fig_path = f"{path}/N{N}_bonds"
    fig.savefig(f"{fig_path}.png", dpi=500, bbox_inches='tight')
    return ener, evecs

twist_hamiltonian(np.array([0.1,0.1,0.1,0.1,0.1]),'z6')