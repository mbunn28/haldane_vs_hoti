#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ti
import numpy as np
import numpy.linalg
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rc
import scipy.linalg
import numpy.random
import numpy.ma
from tqdm import tqdm
from scipy.signal import argrelextrema

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble= r'\usepackage{amsfonts}')

def curve(tau, zq):
    if zq == 'z6':
        CoG = (2*np.pi/6)*np.array([1,1,1,1,1])
        # e0 = np.zeros(6,dtype=float)
        e1 = np.array([2*np.pi,0,0,0,0])
        # e2 = np.array([0,2*np.pi,0,0,0])
        # e3 = np.array([0,0,2*np.pi,0,0])
        # e4 = np.array([0,0,0,2*np.pi,0])
        # e5 = np.array([0,0,0,0,2*np.pi])

        if tau <= 0.5:
            the = 2*tau*CoG
        elif tau <= 1:
            the = 2*(1-tau)*CoG+(2*tau-1)*e1
        else:
            print("error! poorly parameterised curve")
            the = None

    elif zq == 'z2':
        the = 2*np.pi*tau

    else:
        the = None
        print('error in curve parameterisation!')

    return the

class zq_lattice:
    def __init__(
            self, 
            a=1, l=0, b=1, t=1, M=0, N=6, zq = 'z6',
            the = np.zeros(5), loc = np.array([4,4], dtype=int)
            ):
        self.PBC_i = True
        self.PBC_j = True
        self.Corners = False
        self.N = N
        self.a = a
        self.l = l
        self.b = b
        self.t = t
        self.M = M
        self.h = None
        self.energies = None
        self.waves = None
        self.the = the
        self.location = loc
        self.proj = None
        self.zq = zq

    def lat(self,i,j,s): return(6*self.N*i+6*j+s)

    def twist_hamiltonian(self):
            vv = 100000
            h = np.zeros((6*(self.N)**2,6*(self.N)**2), dtype = complex)
            
            for i in range(self.N):
                for j in range(self.N):
                    if (i == self.location[0] and j == self.location[1] and self.zq == 'z6'):
                        the5 = -np.sum(self.the)
                        h[self.lat(i,j,0), self.lat(i,j,1)] = -self.t*self.b*np.exp(-1j*self.the[1])
                        h[self.lat(i,j,0), self.lat(i,j,5)] = -self.t*self.b*np.exp(1j*self.the[0])
                        h[self.lat(i,j,0), self.lat((i+1)%self.N,j,3)] = -self.t*self.a

                        h[self.lat(i,j,2), self.lat(i,j,1)] = -self.t*self.b*np.exp(1j*self.the[2])
                        h[self.lat(i,j,2), self.lat(i,j,3)] = -self.t*self.b*np.exp(-1j*self.the[3])
                        h[self.lat(i,j,2), self.lat(i,(j+1)%self.N,5)] = -self.t*self.a

                        h[self.lat(i,j,4), self.lat(i,j,3)] = -self.t*self.b*np.exp(1j*self.the[4])
                        h[self.lat(i,j,4), self.lat(i,j,5)] = -self.t*self.b*np.exp(-1j*the5)
                        h[self.lat(i,j,4), self.lat((i-1)%self.N,(j-1)%self.N,1)] = -self.t*self.a

                        h[self.lat(i,j,0), self.lat(i,j,4)] = -1j*self.l*self.b*np.exp(-1j*(self.the[4]+self.the[3]+self.the[2]+self.the[1]))
                        h[self.lat(i,j,1), self.lat(i,j,5)] = -1j*self.l*self.b*np.exp(1j*(self.the[1]+self.the[0]))
                        h[self.lat(i,j,2), self.lat(i,j,0)] = -1j*self.l*self.b*np.exp(1j*(self.the[2]+self.the[1]))
                        h[self.lat(i,j,3), self.lat(i,j,1)] = -1j*self.l*self.b*np.exp(1j*(self.the[3]+self.the[2]))
                        h[self.lat(i,j,4), self.lat(i,j,2)] = -1j*self.l*self.b*np.exp(1j*(self.the[4]+self.the[3]))
                        h[self.lat(i,j,5), self.lat(i,j,3)] = -1j*self.l*self.b*np.exp(-1j*(self.the[3]+self.the[2]+self.the[1]+self.the[0]))
                    
                    elif (i == self.location[0] and j == self.location[1] and self.zq == 'z2'):
                        h[self.lat(i,j,0), self.lat(i,j,1)] = -self.t*self.b
                        h[self.lat(i,j,0), self.lat(i,j,5)] = -self.t*self.b
                        h[self.lat(i,j,0), self.lat((i+1)%self.N,j,3)] = -self.t*self.a

                        h[self.lat(i,j,2), self.lat(i,j,1)] = -self.t*self.b
                        h[self.lat(i,j,2), self.lat(i,j,3)] = -self.t*self.b
                        h[self.lat(i,j,2), self.lat(i,(j+1)%self.N,5)] = -self.t*self.a*np.exp(1j*self.the)

                        h[self.lat(i,j,4), self.lat(i,j,3)] = -self.t*self.b
                        h[self.lat(i,j,4), self.lat(i,j,5)] = -self.t*self.b
                        h[self.lat(i,j,4), self.lat((i-1)%self.N,(j-1)%self.N,1)] = -self.t*self.a

                        h[self.lat(i,j,0), self.lat(i,j,4)] = -1j*self.l*self.b
                        h[self.lat(i,j,1), self.lat(i,j,5)] = -1j*self.l*self.b
                        h[self.lat(i,j,2), self.lat(i,j,0)] = -1j*self.l*self.b
                        h[self.lat(i,j,3), self.lat(i,j,1)] = -1j*self.l*self.b
                        h[self.lat(i,j,4), self.lat(i,j,2)] = -1j*self.l*self.b
                        h[self.lat(i,j,5), self.lat(i,j,3)] = -1j*self.l*self.b

                    else:
                        h[self.lat(i,j,0), self.lat(i,j,1)] = -self.t*self.b
                        h[self.lat(i,j,0), self.lat(i,j,5)] = -self.t*self.b
                        h[self.lat(i,j,0), self.lat((i+1)%self.N,j,3)] = -self.t*self.a

                        h[self.lat(i,j,2), self.lat(i,j,1)] = -self.t*self.b
                        h[self.lat(i,j,2), self.lat(i,j,3)] = -self.t*self.b
                        h[self.lat(i,j,2), self.lat(i,(j+1)%self.N,5)] = -self.t*self.a

                        h[self.lat(i,j,4), self.lat(i,j,3)] = -self.t*self.b
                        h[self.lat(i,j,4), self.lat(i,j,5)] = -self.t*self.b
                        h[self.lat(i,j,4), self.lat((i-1)%self.N,(j-1)%self.N,1)] = -self.t*self.a

                        h[self.lat(i,j,0), self.lat(i,j,4)] = -1j*self.l*self.b
                        h[self.lat(i,j,1), self.lat(i,j,5)] = -1j*self.l*self.b
                        h[self.lat(i,j,2), self.lat(i,j,0)] = -1j*self.l*self.b
                        h[self.lat(i,j,3), self.lat(i,j,1)] = -1j*self.l*self.b
                        h[self.lat(i,j,4), self.lat(i,j,2)] = -1j*self.l*self.b
                        h[self.lat(i,j,5), self.lat(i,j,3)] = -1j*self.l*self.b

                    if self.N !=1:
                        h[self.lat(i,j,0), self.lat((i+1)%self.N,j,4)] = -1j*self.l*self.a
                        h[self.lat(i,j,0), self.lat((i+1)%self.N,(j+1)%self.N,4)] = -1j*self.l*self.a

                        h[self.lat(i,j,1), self.lat((i+1)%self.N,(j+1)%self.N,5)] = -1j*self.l*self.a
                        h[self.lat(i,j,1), self.lat(i,(j+1)%self.N,5)] = -1j*self.l*self.a

                        h[self.lat(i,j,2), self.lat(i,(j+1)%self.N,0)] = -1j*self.l*self.a
                        h[self.lat(i,j,2), self.lat((i-1)%self.N,j,0)] = -1j*self.l*self.a

                        h[self.lat(i,j,3), self.lat((i-1)%self.N,j,1)] = -1j*self.l*self.a
                        h[self.lat(i,j,3), self.lat((i-1)%self.N,(j-1)%self.N,1)] = -1j*self.l*self.a

                        h[self.lat(i,j,4), self.lat((i-1)%self.N,(j-1)%self.N,2)] = -1j*self.l*self.a
                        h[self.lat(i,j,4), self.lat(i,(j-1)%self.N,2)] = -1j*self.l*self.a

                        h[self.lat(i,j,5), self.lat(i,(j-1)%self.N,3)] = -1j*self.l*self.a
                        h[self.lat(i,j,5), self.lat((i+1)%self.N,j,3)] = -1j*self.l*self.a

                    for s in [0,2,4]:
                        h[self.lat(i,j,s), self.lat(i,j,s)] = +self.M/2
                    for s in [1,3,5]:
                        h[self.lat(i,j,s), self.lat(i,j,s)] = -self.M/2



            if self.PBC_i == False:
                for j in range(self.N):
                    h[self.lat(self.N-1,j,0), self.lat(0,j,3)] = 0
                    h[self.lat(0,j,4), self.lat(self.N-1,(j-1)%self.N,1)] = 0

                    h[self.lat(0,(j+1)%self.N,3), self.lat(self.N-1,j,1)] = 0
                    h[self.lat(self.N-1,j,0), self.lat(0,j,4)] = 0
                    h[self.lat(0,(j+1)%self.N,4), self.lat(self.N-1,j,2)] = 0
                    h[self.lat(self.N-1,j,5), self.lat(0,j,3)] = 0

                    h[self.lat(self.N-1,j,1), self.lat(0,(j+1)%self.N,5)] = 0
                    h[self.lat(self.N-1,j,0), self.lat(0,(j+1)%self.N,4)] = 0
                    h[self.lat(0,j,3), self.lat(self.N-1,j,1)] = 0
                    h[self.lat(0,j,2), self.lat(self.N-1,j,0)] = 0

            if self.PBC_j == False:
                for i in range(self.N):
                    h[self.lat(i,self.N-1,2), self.lat(i,0,5)] = 0
                    h[self.lat(i,0,4), self.lat((i-1)%self.N,self.N-1,1)] = 0

                    h[self.lat((i+1)%self.N,0,3), self.lat(i,self.N-1,1)] = 0
                    h[self.lat(i,self.N-1,1), self.lat(i,0,5)] = 0
                    h[self.lat((i+1)%self.N,0,4), self.lat(i,self.N-1,2)] = 0
                    h[self.lat(i,self.N-1,2), self.lat(i,0,0)] = 0

                    h[self.lat(i,self.N-1,1), self.lat((i+1)%self.N,0,5)] = 0
                    h[self.lat(i,0,5), self.lat(i,self.N-1,3)] = 0
                    h[self.lat(i,self.N-1,0), self.lat((i+1)%self.N,0,4)] = 0
                    h[self.lat(i,0,4), self.lat(i,self.N-1,2)] = 0

            #dimer geometry
            if self.PBC_j == False and self.Corners == True:
                for i in range(0,self.N):
                    for s in [0,3,4,5]:
                        h[self.lat(i,0,s),self.lat(i,0,s)] = vv

                    for s in [0,1,2,3]:
                        h[self.lat(i,self.N-1,s),self.lat(i,self.N-1,s)] = vv


            if self.PBC_i==False and self.Corners == True:
                for j in range(0,self.N):
                    for s in [2,3,4,5]:
                        h[self.lat(0,j,s),self.lat(0,j,s)] = vv

                    for s in [0,1,2,5]:
                        h[self.lat(self.N-1,j,s),self.lat(self.N-1,j,s)] = vv

            h = np.conjugate(h.transpose()) + h
            self.h = h

            ener, evecs = scipy.linalg.eigh(h)
            if (ener[3*(self.N**2)]-ener[3*(self.N**2)-1]) < 1e-3:
                print('energy very small!')
            self.energies = ener
            self.waves = evecs
            return