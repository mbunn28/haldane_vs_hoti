#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
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
from tqdm.auto import trange
from tqdm import tqdm
from scipy.signal import argrelextrema
import zq_lib

import cProfile
from multiprocessing import Pool

def format_func(value, tick_number):
    if value <= 1:
        return f'{np.round(value,3)}'
    else:
        v = np.round(2 - value, 3)
        part1 = r'$\frac{1}{'
        part2 = r'}$'
        return fr'{part1}{v}{part2}'

def fetch_halffill_phases():
    N_or_res = "res"
    Nphase = 600
    path_phasediagram = "output/phasediagram/periodic"
    x_to_plot = joblib.load(f"{path_phasediagram}/{N_or_res}{Nphase}_x_to_plot")
    y_to_plot = joblib.load(f"{path_phasediagram}/{N_or_res}{Nphase}_y_to_plot")
    return x_to_plot, y_to_plot

# fig2, ax2 = plt.subplots()
# ax2.set_aspect(1)
# # for i in range(np.shape(x_to_plot)[0]):
# #     ax2.plot(x_to_plot[i,:],y_to_plot[i,:],c='k',lw=0.75)


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

# small_energy[small_energy>1e-1] = np.NaN

# fig3, ax3 = plt.subplots()
# for i in range(np.shape(x_to_plot)[0]):
#     ax3.plot(x_to_plot[i,:],y_to_plot[i,:],c='k',lw=0.75)
# im2 = ax3.pcolormesh(x,y,small_energy[:,:,0], norm = colors.LogNorm(), cmap='inferno')
# fig3.colorbar(im2)
# ax3.set_aspect(1)
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

# fig5, ax5 = plt.subplots()
# for i in range(np.shape(x_to_plot)[0]):
#     ax5.plot(x_to_plot[i,:],y_to_plot[i,:],c='k',lw=0.75)
# im4 = ax5.pcolormesh(x,y,small_energy_loc[:,:,0])
# fig5.colorbar(im4)
# ax5.set_aspect(1)
# title4 = f'Small energy in {title} calc'
# ax5.set_title(rf'{title2}: $ N = {N},$ it $= {iterations}$, $res = {points}$')
# ax5.set_ylabel(r'$\alpha$')
# ax5.set_xlabel(r'$\lambda$')
# ax5.set_xlim(min_x,max_x)
# ax5.set_ylim(min_y,max_y)
# ax5.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
# ax5.yaxis.set_major_formatter(plt.FuncFormatter(format_func))

# fig_path4 = f"{path_zq}/N{N}_iter{iterations}_res{points}_z6_energy_locs"
# fig5.savefig(f"{fig_path4}.png", dpi=500, bbox_inches='tight')

def main():
    
    # TODO implement argparse
    points = 80
    iterations = 4
    location = np.array([2,2], dtype=int)
    N = 18
    max_x = 2
    min_x = 0
    max_y = 2
    min_y = 0
    filling = 'half'

    zq = ['z6']

    if filling == 'third' or filling == 'sixth':
        gapless = True
    else:
        gapless = False

    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    rc('text.latex', preamble= r'\usepackage{amsfonts}')

    def make_filenames():
        path_zq = "output/zq/diagrams"
        if not os.path.exists(path_zq):
            os.makedirs(path_zq)
        zq_phases_path = f'{path_zq}/zq_phases_N{N}_it{iterations}_res{points}'
        small_energy_path = f'{path_zq}/small_energy_N{N}_it{iterations}_res{points}'
        small_energy_loc_path = f'{path_zq}/smallen_loc_N{N}_it{iterations}_res{points}'
        return [path_zq,zq_phases_path,small_energy_path,small_energy_loc_path]

    path_zq,zq_phases_path,small_energy_path,small_energy_loc_path = make_filenames()

    if not (os.path.exists(zq_phases_path)):
        raise ValueError('data not yet calculated')

    zq_phases = joblib.load(zq_phases_path)
    small_energy = joblib.load(small_energy_path)
    small_energy_loc = joblib.load(small_energy_loc_path)


    def define_col_map(zq_type):
        if zq_type == 'z6':
            cmap = plt.cm.Dark2  # define the colormap
            # extract all colors from the .jet map
            cmaplist = [cmap(i) for i in range(6)]
            v_max = 5
        if zq_type == 'z2':
            cmap = plt.cm.tab10  # define the colormap
            # extract all colors from the .jet map
            cmaplist = [cmap(i) for i in range(2)]
            v_max = 1
        cmaplist[0] = (0,0,0,0)
        v_min = 0
        if gapless == True:
            cmaplist = np.insert(cmaplist,0,(0,0,0,1),axis=0)
            v_min = -1
        # create the new map     
        cmap = mpl.colors.ListedColormap(cmaplist)
        return cmap, v_min, v_max

    def make_colbar_labels(zq_type):
        if zq_type == 'z6' and gapless == False:
            labels = [0,1,2,3,4,5]
            loc    = np.array([5/12,15/12,25/12,35/12,45/12,55/12])
        elif zq_type == 'z6' and gapless == True:
            labels = ['Gapless',0,1,2,3,4,5]
            loc    = np.array([-5/12,5/12,15/12,25/12,35/12,45/12,55/12])
        elif zq_type == 'z2' and gapless == False:
            labels = [0,1]
            loc    = np.array([1/4,3/4])
        elif zq_type == 'z2' and gapless == True:
            labels = ['Gapless',0,1]
            loc    = np.array([-3/4,1/4,3/4])
        return labels, loc

    def plot_phasediagram(zq_phases,zq_type):
        fig1, ax = plt.subplots(figsize=(3.4,3.4))
        ax.set_aspect(1)
        cmap, v_min, v_max  = define_col_map(zq_type)
        x = np.linspace(min_x, max_x, num=points)
        y = np.linspace(min_y, max_y, num=points)

        im = ax.pcolormesh(x,y,zq_phases,cmap=cmap,vmin=v_min,vmax=v_max)

        #plotting the halffilling phases
        if filling == 'half':
            x_to_plot, y_to_plot = fetch_halffill_phases()
            for i in range(np.shape(x_to_plot)[0]):
                ax.plot(x_to_plot[i,:],y_to_plot[i,:],c='k',lw=0.75)

        #making the colourbar    
        cb = fig1.colorbar(im,cmap=cmap, format='%1i')
        labels, loc = make_colbar_labels(zq_type)
        cb.set_ticks(loc)
        cb.set_ticklabels(labels)
        ax.set_ylabel(r'$\alpha$')
        ax.set_xlabel(r'$\lambda$')
        ax.set_xlim(min_x,max_x)
        ax.set_ylim(min_y,max_y)

        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))

        fig_path = f"{path_zq}/N{N}_iter{iterations}_res{points}_z6"
        fig1.savefig(f"{fig_path}.png", dpi=500, bbox_inches='tight')
        return
    
    for i in range(len(zq)):
        plot_phasediagram(zq_phases[:,:,i],zq[i])
    return

if __name__ == "__main__":
    # cProfile.run('main()', sort='cumtime')
    main()