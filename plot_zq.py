#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import joblib
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rc
import zq_lib, ti
import copy
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy.ma import masked_array
import matplotlib.gridspec as gs

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

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def main():
    
    # TODO implement argparse
    points = 20
    iterations = 4
    location = np.array([2,2], dtype=int)
    N = 14
    max_x = 2
    min_x = 0
    max_y = 2
    min_y = 0
    filling = 'half'

    zq = ['z6']
    inset_iterations=40
    
    if filling == 'third' or filling == 'sixth':
        gapless = True
    else:
        gapless = False

    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    rc('text.latex', preamble= r'\usepackage{amsfonts}')

    def make_filenames():
        path_zq = f"output/zq/diagrams/no_gauge_fix/{filling}"
        if not os.path.exists(path_zq):
            os.makedirs(path_zq)
        zq_phases_path = f'{path_zq}/zq_phases_N{N}_it{iterations}_res{points}'
        small_energy_path = f'{path_zq}/small_energy_N{N}_it{iterations}_res{points}'
        small_energy_loc_path = f'{path_zq}/smallen_loc_N{N}_it{iterations}_res{points}'
        return [path_zq,zq_phases_path,small_energy_path,small_energy_loc_path]

    def make_minen_filenames():
        # path_zq_minen = f"output/zq/min_energy/{filling}"
        path_zq_minen = f"output/zq/diagrams/no_gauge_fix/{filling}"
        # min_en_data = f"{path_zq_minen}/small_energy_N{N}_it{inset_iterations}"
        min_en_data = f"{path_zq_minen}/small_energy_N{N}_it{iterations}_res{points}"
        return [path_zq_minen, min_en_data]

    def load_gapless():
        if filling == 'half':
            return np.zeros((points,points), dtype=bool)
        else:
            path = f"output/phasediagram/periodic/{filling}"
            gapless = joblib.load(f"{path}/res{points}_gapmask")
            return gapless
    
    def add_axes_labels(ax):
        ax.set_ylabel(r'$\alpha$')
        ax.set_xlabel(r'$\lambda$')
        ax.set_xlim(min_x,max_x)
        ax.set_ylim(min_y,max_y)

        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
        ticks = ax.xaxis.get_major_ticks()
        ticks[-1].label1.set_horizontalalignment('right')
        return
    
    def plot_phases(ax):
        if filling == 'half':
                x_to_plot, y_to_plot = fetch_halffill_phases()
                for i in range(np.shape(x_to_plot)[0]):
                    ax.plot(x_to_plot[i,:],y_to_plot[i,:],c='k',lw=0.75)
        return

    path_zq,zq_phases_path,small_energy_path,small_energy_loc_path = make_filenames()
    _, min_en_data_path = make_minen_filenames()

    if not (os.path.exists(zq_phases_path)):
        raise ValueError('data not yet calculated')

    zq_phases = joblib.load(zq_phases_path)
    small_energy = joblib.load(small_energy_path)
    small_energy_loc = joblib.load(small_energy_loc_path)
    min_en_data = joblib.load(min_en_data_path)
    x = np.linspace(min_x, max_x, num=points)
    y = np.linspace(min_y, max_y, num=points)

    print(zq_phases)

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

    def make_colbar_labels(cb, zq_type):
        if zq_type == 'z6' and gapless == False:
            labels = [0,1,2,3,4,5]
            loc    = np.array([5/12,15/12,25/12,35/12,45/12,55/12])
        elif zq_type == 'z6' and gapless == True:
            labels = ['No\nGap',0,1,2,3,4,5]
            loc    = np.array([-4/7,2/7,8/7,2,20/7,26/7,32/7])
        elif zq_type == 'z2' and gapless == False:
            labels = [0,1]
            loc    = np.array([1/4,3/4])
        elif zq_type == 'z2' and gapless == True:
            labels = ['No\nGap',0,1]
            loc    = np.array([-2/3,0,2/3])
        cb.set_ticks(loc)
        cb.set_ticklabels(labels)
        # if gapless == True:
        #     ticks = cb.ax.yaxis.get_major_ticks()
        #     ticks[0].label2.set_rotation(90)
        #     ticks[0].label2.set_verticalalignment('center')
        return

    def add_cbar_tag(cb, zq_type):
        if zq_type == 'z6':
            a = 6
        else:
            a = 2
        part1 = r'$\times \frac{2 \pi}{'
        part2 = r'}$'
        cb.ax.set_xlabel(rf'{part1}{a}{part2}',horizontalalignment='left')
        return
    
    def plot_phasediagram(zq_phases,zq_type):
        fig1, ax = plt.subplots(figsize=(3.4,3.4))
        ax.set_aspect(1)
        cmap, v_min, v_max  = define_col_map(zq_type)
        if zq_type == 'z2':
            zq_phases[zq_phases==3] = 1
        im = ax.pcolormesh(x,y,zq_phases,cmap=cmap,vmin=v_min,vmax=v_max)
        plot_phases(ax)
        cb = ti.colorbar(im)
        make_colbar_labels(cb, zq_type)
        add_cbar_tag(cb, zq_type)
        add_axes_labels(ax)

        fig_path = f"{path_zq}/N{N}_iter{iterations}_res{points}_{zq_type}"
        fig1.savefig(f"{fig_path}.png", dpi=500, bbox_inches='tight')
        return
    
    def plot_gapdiagram(small_energy,zq_type):
        small_energy[small_energy>1e-1] = 1e-1
        gapless = load_gapless()
        small_energy[gapless]= -1
        fig, ax = plt.subplots(figsize=(3.4,3.4))
        plot_phases(ax)
        my_cmap = copy.copy(cm.get_cmap('inferno'))
        my_cmap.set_bad('k')
        im = ax.pcolormesh(x,y,small_energy, norm = colors.LogNorm(), cmap=my_cmap)
        ti.colorbar(im)
        ax.set_aspect(1)
        add_axes_labels(ax)
        fig_path2 = f"{path_zq}/N{N}_iter{iterations}_res{points}_{zq_type}_energy"
        fig.savefig(f"{fig_path2}.png", dpi=500, bbox_inches='tight')
        return

    def add_inset(ax):
        subax = ax.inset_axes((0.45,0.18,0.51,0.37))
        subax.plot(np.linspace(0,2*np.pi,inset_iterations+1),min_en_data[:,0])
        subax.set_ylabel('Gap')
        subax.set_xlabel(r'$\theta$')
        subax.set_xlim(0,2*np.pi)
        subax.set_ylim(0,np.amax(min_en_data)+0.1)
        subax.set_xticks((0,np.pi,2*np.pi))
        subax.set_xticklabels((0,r'$\pi$',r'$2\pi$'))
        return

    def plot_phase_gap(zq_phases, small_energy, zq_type):
        if zq_type == 'z2':

            gridspec = dict(wspace=0.0, width_ratios=[1, 0.02, 0.05, 0.05])
            fig, axs = plt.subplots(nrows=1, ncols=4, gridspec_kw=gridspec,figsize=(3.4,3.4/1.12))
            axs[1].set_visible(False)
            # fig, ax = plt.subplots(figsize=(3.4,3.4))
            
            cmap, v_min, v_max  = define_col_map(zq_type)
            if zq_type == 'z2':
                zq_phases[zq_phases==3] = 1
            # ax.imshow(zq_phases, cmap=cmap,vmin=v_min,vmax=v_max,origin='lower',extent=[min_x,max_x,min_y,max_y])

            min_val = 1e-5
            max_val = 5e-1
            small_energy[0,0]=min_val
            small_energy[small_energy>max_val] = max_val
            gapless = load_gapless()
            small_energy[gapless]= -1
            plot_phases(axs[0])
            my_cmap_g0 = copy.copy(cm.get_cmap('gray'))
            my_cmap_g = truncate_colormap(my_cmap_g0,maxval=0.8)
            my_cmap_g.set_over('w')
            my_cmap_g.set_bad('k')
            no_phase = masked_array(small_energy,zq_phases==1)
            im = axs[0].imshow(no_phase, cmap=my_cmap_g,origin='lower',interpolation='nearest', norm = colors.LogNorm(min_val, max_val),vmax=5e-2,extent=[min_x,max_x,min_y,max_y])
            my_cmap_r0 = plt.get_cmap('Oranges_r')
            my_cmap_r = truncate_colormap(my_cmap_r0,maxval=0.8)
            # my_cmap_r.set_bad('k')
            yes_phase = masked_array(small_energy, zq_phases==0)
            im1 = axs[0].imshow(yes_phase, cmap=my_cmap_r,interpolation='nearest', norm = colors.LogNorm(min_val,max_val),origin='lower',extent=[min_x,max_x,min_y,max_y])
            cbar = fig.colorbar(im,cax=axs[2])
            cbar.set_ticks([])
            fig.colorbar(im1,cax=axs[3])
            add_axes_labels(axs[0])
            axs[0].scatter(0.2,1.5,s=80,facecolors='none',edgecolors='C0')
            add_inset(axs[0])
            fig_path2 = f"{path_zq}/N{N}_iter{iterations}_res{points}_{zq_type}_energyphases"
            fig.savefig(f"{fig_path2}.png", dpi=500, bbox_inches='tight')
        return
    
    def plot_gaplocs(gap_locs,zq_type):
        gapless = load_gapless()
        gap_locs[gapless] = np.NaN
        fig, ax = plt.subplots(figsize=(3.4,3.4))
        plot_phases(ax)
        my_cmap = copy.copy(cm.get_cmap('viridis'))
        my_cmap.set_bad('k')
        im = ax.pcolormesh(x,y,gap_locs,cmap=my_cmap)
        ti.colorbar(im)
        ax.set_aspect(1)
        add_axes_labels(ax)
        fig_path4 = f"{path_zq}/N{N}_iter{iterations}_res{points}_{zq_type}_energy_locs"
        fig.savefig(f"{fig_path4}.png", dpi=500, bbox_inches='tight')
        return
    
    for i in range(len(zq)):
        plot_phasediagram(zq_phases[:,:,i],zq[i])
        # plot_gapdiagram(small_energy[:,:,i],zq[i])
        # plot_gaplocs(small_energy_loc[:,:,i],zq[i])
        # plot_phase_gap(zq_phases[:,:,i],small_energy[:,:,i],zq[i])
    return

if __name__ == "__main__":
    # cProfile.run('main()', sort='cumtime')
    main()