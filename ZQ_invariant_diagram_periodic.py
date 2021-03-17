#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from numpy.core.defchararray import multiply
import numpy.linalg
import joblib
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rc
import scipy.linalg
import numpy.random
import numpy.ma
from tqdm import tqdm
from scipy.signal import argrelextrema
import zq_lib

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('text.latex', preamble= r'\usepackage{amsfonts}')

path_zq = "output/zq/diagrams/bz"
if not os.path.exists(path_zq):
            os.makedirs(path_zq)
    
points = 8
iterations = 20000
max_x = 2
min_x = 0
max_y = 2
min_y = 0

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

res = points
x = np.linspace(min_x, max_x, num=points)
y = np.linspace(min_y, max_y, num=points)
a_vals, b_vals = rule(y)
l_vals, t_vals = rule(x)

b1 = np.array([0,4*np.pi*np.sqrt(3)/9])
b2 = (2*np.pi/9)*np.array([3,np.sqrt(3)])
kx_in = (1/3)*(b1[0]+b2[0])*(1-np.linspace(0,1,iterations))
ky_in = (1/3)*(b1[1]+b2[1])*(1-np.linspace(0,1,iterations))
kx_out = (1/3)*(2*b1[0]-b2[0])*np.linspace(0,1,iterations)
ky_out = (1/3)*(2*b1[1]-b2[1])*np.linspace(0,1,iterations)
kx_out = kx_out[1:]
ky_out = ky_out[1:]
kx_vals = np.append(kx_in,kx_out)
ky_vals = np.append(ky_in,ky_out)

l, a, kx = np.meshgrid(l_vals,a_vals,kx_vals)
t, b, ky = np.meshgrid(t_vals,b_vals,ky_vals)


hamiltonians = np.zeros((res,res,2*iterations-1,6,6),dtype=complex)
phi= np.pi/2

hamiltonians[:,:,:,2,0] = l*np.exp(-1j*phi)*(b+a*(np.exp(3*1j*kx)+np.exp(1.5*1j*(kx+ky*np.sqrt(3)))))
hamiltonians[:,:,:,3,0] = t*a*np.exp(1.5*1j*(kx+ky*np.sqrt(3)))
hamiltonians[:,:,:,4,0] = l*np.exp(1j*phi)*(b+a*(np.exp(1.5*1j*(kx+ky*np.sqrt(3)))+np.exp(-1.5*1j*(kx-ky*np.sqrt(3)))))

hamiltonians[:,:,:,3,1] = l*np.exp(-1j*phi)*(b+a*(np.exp(1.5*1j*(kx+ky*np.sqrt(3)))+np.exp(-1.5*1j*(kx-ky*np.sqrt(3)))))
hamiltonians[:,:,:,4,1] = t*a*np.exp(-1.5*1j*(kx-ky*np.sqrt(3)))
hamiltonians[:,:,:,5,1] = l*np.exp(1j*phi)*(b+a*(np.exp(-3*1j*kx)+np.exp(-1.5*1j*(kx-ky*np.sqrt(3)))))

hamiltonians[:,:,:,4,2] = l*np.exp(-1j*phi)*(b+a*(np.exp(-3*1j*kx)+np.exp(-1.5*1j*(kx-ky*np.sqrt(3)))))
hamiltonians[:,:,:,5,2] = t*a*np.exp(-3*1j*kx)

hamiltonians[:,:,:,5,3] = l*np.exp(-1j*phi)*(b+a*(np.exp(-3*1j*kx)+np.exp(-1.5*1j*(kx+ky*np.sqrt(3)))))

hamiltonians[:,:,:,1,0] = b*t
hamiltonians[:,:,:,2,1] = b*t
hamiltonians[:,:,:,3,2] = b*t
hamiltonians[:,:,:,4,3] = b*t
hamiltonians[:,:,:,5,4] = b*t
hamiltonians[:,:,:,0,5] = b*t

hamiltonians = hamiltonians + np.conjugate(np.transpose(hamiltonians,(0,1,2,4,3)))

eigvals, eigvec = np.linalg.eigh(hamiltonians)

gap = np.round(eigvals[:,:,:,3]-eigvals[:,:,:,2],2)
gap_mask = gap == 0
print(np.sum(gap_mask))
zq_phases = np.zeros((res,res))
M = 3
occstates_a = eigvec[:,:,0,:,:M]
phi = np.random.rand(6,3)
phi = scipy.linalg.orth(phi)
proja = np.einsum('...ij,...jk->...ik',occstates_a,np.conjugate(np.transpose(occstates_a,(0,1,3,2))))
pa = np.einsum('...ij,jk->...ik',proja,phi)
for i in range(2*iterations-2):
    occstates_b = eigvec[:,:,i+1,:,:M]
    projb = np.einsum('...ij,...jk->...ik',occstates_b,np.conjugate(np.transpose(occstates_b,(0,1,3,2))))
    pb = np.einsum('...ij,jk->...ik',projb,phi)
    Di = np.einsum('...ij,...jk->...ik',np.conjugate(np.transpose(pa,(0,1,3,2))),pb)
    (sign,_) = np.linalg.slogdet(Di)
    # Gi = np.einsum('...ij,...jk->...ik',np.conjugate(np.transpose(occstates_a,(0,1,3,2))),occstates_b)
    # Ui, _, Vi = np.linalg.svd(Gi)
    # Fi = np.einsum('...ij,...jk->...ik',Ui,Vi)
    # (sign_Fi,_) = np.linalg.slogdet(Fi)
    if i==0:
        W = sign
    else:
        W = np.multiply(sign,W)
    proja=projb

vals = np.round((6*np.angle(W))/(2*np.pi),2)

gap_mask = np.any(gap_mask,axis=-1)
vals[gap_mask==True] = np.NaN
print(vals)

# joblib.dump(zq_phases,f'{path_zq}/zq_phases_N{N}_it{iterations}_res{points}')

# N_or_res = "res"
# Nphase = 600
# path_phasediagram = "output/phasediagram/periodic"
# x_to_plot = joblib.load(f"{path_phasediagram}/{N_or_res}{Nphase}_x_to_plot")
# y_to_plot = joblib.load(f"{path_phasediagram}/{N_or_res}{Nphase}_y_to_plot")

# fig1, ax1 = plt.subplots()
# # x = np.linspace(0,2,num=res)
# for i in range(np.shape(x_to_plot)[0]):
#     ax1.plot(x_to_plot[i,:],y_to_plot[i,:],c='k',lw=0.75)

# cmap = plt.cm.Dark2  # define the colormap
# # extract all colors from the .jet map
# cmaplist = [cmap(i) for i in range(6)]
# # create the new map
# cmap = mpl.colors.ListedColormap(cmaplist)

# im = ax1.pcolormesh(x,y,zq_phases[:,:,0],cmap=cmap)
# cb1 = fig1.colorbar(im,cmap=cmap, format='%1i')
# labels = np.arange(0,6,1)
# loc    = np.array([5/12,15/12,25/12,35/12,45/12,55/12])
# cb1.set_ticks(loc)
# cb1.set_ticklabels(labels)
# title = '$\mathbb{Z}_6$ Berry Phase'
# ax1.set_title(rf'{title}: $ N = {N},$ it $= {iterations}$, $res = {points}$')
# ax1.set_ylabel(r'$\alpha$')
# ax1.set_xlabel(r'$\lambda$')
# ax1.set_xlim(min_x,max_x)
# ax1.set_ylim(min_y,max_y)

# def format_func(value, tick_number):
#     if value <= 1:
#         return f'{np.round(value,3)}'
#     else:
#         v = np.round(2 - value, 3)
#         part1 = r'$\frac{1}{'
#         part2 = r'}$'
#         return fr'{part1}{v}{part2}'

# ax1.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
# ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_func))

# fig_path = f"{path_zq}/N{N}_iter{iterations}_res{points}_z6"
# fig1.savefig(f"{fig_path}.png", dpi=500, bbox_inches='tight')

# fig2, ax2 = plt.subplots()
# for i in range(np.shape(x_to_plot)[0]):
#     ax2.plot(x_to_plot[i,:],y_to_plot[i,:],c='k',lw=0.75)

# cmap = plt.cm.tab10  # define the colormap
# # extract all colors from the .jet map
# cmaplist = [cmap(i) for i in range(2)]
# # create the new map
# cmap = mpl.colors.ListedColormap(cmaplist)

# im1 = ax2.pcolormesh(x,y,zq_phases[:,:,1]/3,cmap=cmap)
# cb2 = fig2.colorbar(im1,cmap=cmap, format='%1i')
# labels1 = [0,1]
# loc1    = np.array([1/4,3/4])
# cb2.set_ticks(loc1)
# cb2.set_ticklabels(labels1)
# title1 = '$\mathbb{Z}_2$ Berry Phase'
# ax2.set_title(rf'{title1}: $ N = {N},$ it $= {iterations}$, res $= {points}$')
# ax2.set_ylabel(r'$\alpha$')
# ax2.set_xlabel(r'$\lambda$')
# ax2.set_xlim(min_x,max_x)
# ax2.set_ylim(min_y,max_y)
# ax2.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
# ax2.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
# fig_path1 = f"{path_zq}/N{N}_iter{iterations}_res{points}_z2"
# fig2.savefig(f"{fig_path1}.png", dpi=500, bbox_inches='tight')

# small_energy[small_energy>1e-2] = np.NaN

# fig3, ax3 = plt.subplots()
# for i in range(np.shape(x_to_plot)[0]):
#     ax3.plot(x_to_plot[i,:],y_to_plot[i,:],c='k',lw=0.75)
# im2 = ax3.pcolormesh(x,y,small_energy[:,:,0], norm = colors.LogNorm(), cmap='inferno')
# fig3.colorbar(im2)
# title2 = f'Small energy in {title} calc'
# ax3.set_title(rf'{title2}: $ N = {N},$ it $= {iterations}$, $res = {points}$')
# ax3.set_ylabel(r'$\alpha$')
# ax3.set_xlabel(r'$\lambda$')
# ax3.set_xlim(min_x,max_x)
# ax3.set_ylim(min_y,max_y)
# ax3.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
# ax3.yaxis.set_major_formatter(plt.FuncFormatter(format_func))

# fig_path2 = f"{path_zq}/N{N}_iter{iterations}_res{points}_z6_energy"
# fig3.savefig(f"{fig_path2}.png", dpi=500, bbox_inches='tight')

# fig4, ax4 = plt.subplots()
# for i in range(np.shape(x_to_plot)[0]):
#     ax4.plot(x_to_plot[i,:],y_to_plot[i,:],c='k',lw=0.75)
# im3 = ax4.pcolormesh(x,y,small_energy[:,:,1], norm = colors.LogNorm(), cmap='inferno')
# fig4.colorbar(im3)
# title3 = f'Small energy in {title1} calc'
# ax4.set_title(rf'{title3}: $ N = {N},$ it $= {iterations}$, res $= {points}$')
# ax4.set_ylabel(r'$\alpha$')
# ax4.set_xlabel(r'$\lambda$')
# ax4.set_xlim(min_x,max_x)
# ax4.set_ylim(min_y,max_y)
# ax4.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
# ax4.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
# fig_path3 = f"{path_zq}/N{N}_iter{iterations}_res{points}_z2_energy"
# fig4.savefig(f"{fig_path3}.png", dpi=500, bbox_inches='tight')
