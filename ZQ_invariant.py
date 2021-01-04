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

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

l = 0
t = 1

a = 0.15
b = 1

N = 10
M = 0

PBC_i = False
PBC_j = False
Corners = False

location = np.array([5,5], dtype=int)

def lat(i,j,s): return(6*N*i+6*j+s)

def twist_hamiltonian(theta):
    vv = 100000
    h = np.zeros((6*(N)**2,6*(N)**2), dtype = complex)
    theta5 = -np.sum(theta)

    for i in range(N):
        for j in range(N):
            if i == location[0] and j == location[1]:
                h[lat(i,j,0), lat(i,j,1)] = -t*b*np.exp(-1j*theta[1])
                h[lat(i,j,0), lat(i,j,5)] = -t*b*np.exp(1j*theta[0])
                h[lat(i,j,0), lat((i+1)%N,j,3)] = -t*a

                h[lat(i,j,2), lat(i,j,1)] = -t*b*np.exp(1j*theta[2])
                h[lat(i,j,2), lat(i,j,3)] = -t*b*np.exp(-1j*theta[3])
                h[lat(i,j,2), lat(i,(j+1)%N,5)] = -t*a

                h[lat(i,j,4), lat(i,j,3)] = -t*b*np.exp(1j*theta[4])
                h[lat(i,j,4), lat(i,j,5)] = -t*b*np.exp(-1j*theta5)
                h[lat(i,j,4), lat((i-1)%N,(j-1)%N,1)] = -t*a

                h[lat(i,j,0), lat(i,j,4)] = -1j*l*b*np.exp(1j*(theta[4]+theta[3]+theta[2]+theta[1]))
                h[lat(i,j,1), lat(i,j,5)] = -1j*l*b*np.exp(-1j*(theta[1]+theta[0]))
                h[lat(i,j,2), lat(i,j,0)] = -1j*l*b*np.exp(-1j*(theta[2]+theta[1]))
                h[lat(i,j,3), lat(i,j,1)] = -1j*l*b*np.exp(-1j*(theta[3]+theta[2]))
                h[lat(i,j,4), lat(i,j,2)] = -1j*l*b*np.exp(-1j*(theta[4]+theta[3]))
                h[lat(i,j,5), lat(i,j,3)] = -1j*l*b*np.exp(1j*(theta[3]+theta[2]+theta[1]+theta[0]))
            
            else:
                h[lat(i,j,0), lat(i,j,1)] = -t*b
                h[lat(i,j,0), lat(i,j,5)] = -t*b
                h[lat(i,j,0), lat((i+1)%N,j,3)] = -t*a

                h[lat(i,j,2), lat(i,j,1)] = -t*b
                h[lat(i,j,2), lat(i,j,3)] = -t*b
                h[lat(i,j,2), lat(i,(j+1)%N,5)] = -t*a

                h[lat(i,j,4), lat(i,j,3)] = -t*b
                h[lat(i,j,4), lat(i,j,5)] = -t*b
                h[lat(i,j,4), lat((i-1)%N,(j-1)%N,1)] = -t*a

                h[lat(i,j,0), lat(i,j,4)] = -1j*l*b
                h[lat(i,j,1), lat(i,j,5)] = -1j*l*b
                h[lat(i,j,2), lat(i,j,0)] = -1j*l*b
                h[lat(i,j,3), lat(i,j,1)] = -1j*l*b
                h[lat(i,j,4), lat(i,j,2)] = -1j*l*b
                h[lat(i,j,5), lat(i,j,3)] = -1j*l*b

                if N !=1:
                    h[lat(i,j,0), lat((i+1)%N,j,4)] = -1j*l*a
                    h[lat(i,j,0), lat((i+1)%N,(j+1)%N,4)] = -1j*l*a

                    h[lat(i,j,1), lat((i+1)%N,(j+1)%N,5)] = -1j*l*a
                    h[lat(i,j,1), lat(i,(j+1)%N,5)] = -1j*l*a

                    h[lat(i,j,2), lat(i,(j+1)%N,0)] = -1j*l*a
                    h[lat(i,j,2), lat((i-1)%N,j,0)] = -1j*l*a

                    h[lat(i,j,3), lat((i-1)%N,j,1)] = -1j*l*a
                    h[lat(i,j,3), lat((i-1)%N,(j-1)%N,1)] = -1j*l*a

                    h[lat(i,j,4), lat((i-1)%N,(j-1)%N,2)] = -1j*l*a
                    h[lat(i,j,4), lat(i,(j-1)%N,2)] = -1j*l*a

                    h[lat(i,j,5), lat(i,(j-1)%N,3)] = -1j*l*a
                    h[lat(i,j,5), lat((i+1)%N,j,3)] = -1j*l*a

            for s in [0,2,4]:
                h[lat(i,j,s), lat(i,j,s)] = +M/2
            for s in [1,3,5]:
                h[lat(i,j,s), lat(i,j,s)] = -M/2



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

    _, evecs = scipy.linalg(h)
    return evecs

def u(theta1,theta2,n):
    evecs1 = twist_hamiltonian(theta1)
    evecs2 = twist_hamiltonian(theta2)
    res = np.vdot(evecs1[n,:],evecs2[n,:])
    res = res/np.abs(res)
    return res

def curve(tau):
    CoG = (2*np.pi/6)*np.array([1,1,1,1,1])
    e0 = np.zeros(6,dtype=float)
    e1 = np.array([2*np.pi,0,0,0,0])
    # e2 = np.array([0,2*np.pi,0,0,0])
    # e3 = np.array([0,0,2*np.pi,0,0])
    # e4 = np.array([0,0,0,2*np.pi,0])
    # e5 = np.array([0,0,0,0,2*np.pi])

    if tau <= 0.5:
        theta = 2*tau*CoG
    else:
        theta = (1-2*tau)*CoG+(2*tau-1)*e1

    return theta







