#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import joblib
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rc
import copy
import matplotlib.cm as cm
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

def main():
    
    # TODO implement argparse
    #params = (a, b, l ,t)

    N = 16
    points = 100
    iterations = 4

    gapless = True

    max_x = 2
    min_x = 0
    max_y = 2
    min_y = 0
    x = np.linspace(min_x, max_x, num=points)
    y = np.linspace(min_y, max_y, num=points)

    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    rc('text.latex', preamble= r'\usepackage{amsfonts}')

    def load_gapless(filling):
        if filling == 'half':
            return np.zeros((points,points), dtype=bool)
        else:
            path = f"output/phasediagram/periodic/{filling}"
            gapless = joblib.load(f"{path}/res{points}_gapmask")
            return gapless

    def make_zq_filenames(filling):
        path_zq = f"output/zq/diagrams/{filling}"
        if not os.path.exists(path_zq):
            os.makedirs(path_zq)
        zq_phases_path = f'{path_zq}/zq_phases_N{N}_it{iterations}_res{points}'
        small_energy_path = f'{path_zq}/small_energy_N{N}_it{iterations}_res{points}'
        small_energy_loc_path = f'{path_zq}/smallen_loc_N{N}_it{iterations}_res{points}'
        return [path_zq,zq_phases_path,small_energy_path,small_energy_loc_path]

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
        cb.ax.xaxis.set_label_coords(-0.2,-0.5)
        return

    def add_axes_labels(ax, plot_x, plot_y):
        ax.set_xlim(min_x,max_x)
        ax.set_ylim(min_y,max_y)

        if plot_x == False:
            ax.xaxis.set_ticks([])
        else:
            ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
            ticks = ax.xaxis.get_major_ticks()
            ticks[-1].label1.set_horizontalalignment('right')
            ticks[0].label1.set_horizontalalignment('left')
        
        if plot_y == False:
            ax.yaxis.set_ticks([])
        else:
            ax.set_ylabel(r'$\alpha$')       
            ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
            ticks = ax.yaxis.get_major_ticks()
            ticks[-1].label1.set_verticalalignment('top')
            ticks[0].label1.set_verticalalignment('bottom')
        
        return

    def plot_gap(ax, filling, plot_x = False):
        path = f"output/phasediagram/periodic/{filling}"
        gap_path = f"{path}/res{points}_gap"
        gap = joblib.load(gap_path)
        # gap[gap < 1e-5] = 1e-5
        my_cmap = copy.copy(cm.get_cmap('inferno'))
        my_cmap.set_bad('k')
        im = plt.pcolormesh(x,y,gap, norm = colors.SymLogNorm(linthresh=1e-5), cmap=my_cmap, shading='auto')
        ax.grid(linestyle='--')
        add_axes_labels(ax, plot_x, plot_y = True)
        return im

    def plot_phases(ax, filling, zq_type, plot_x = False, xlabel = False):
        _, zq_path, _, _ = make_zq_filenames(filling)
        zq_phases_all = joblib.load(zq_path)
        cmap, v_min, v_max  = define_col_map('z6')
        if zq_type == 'z2':
            zq_phases = zq_phases_all[:,:,1]
            # zq_phases[zq_phases==3] = 1
        elif zq_type == 'z6':
            zq_phases = zq_phases_all[:,:,0]
        im = ax.pcolormesh(x,y,zq_phases,cmap=cmap,vmin=v_min,vmax=v_max, shading='auto')
        add_axes_labels(ax, plot_x, plot_y = False)
        if xlabel == True:
            ax.set_xlabel(r'$\lambda$')
        return im

    def make_figure():
        cbar_height = 0.1
        gap_height = 0.4
        w = 7.06
        one_w = w/3
        h = 2*one_w + cbar_height + gap_height
        fig = plt.figure(figsize=(w,h),dpi=500,constrained_layout=False)

        heights = [one_w,one_w, gap_height, cbar_height]
        main_grid = gs.GridSpec(ncols=3, nrows=4, figure=fig, height_ratios=heights, wspace=0.05,hspace=0.05)

        ax00 = fig.add_subplot(main_grid[0,0])
        im0 = plot_gap(ax00,'sixth')
        ax00.text(1,1,'(a)', c='w', ha='center', va='center')
        ax01 = fig.add_subplot(main_grid[0,1])
        im1 = plot_phases(ax01,'sixth','z2')
        ax01.text(1,1,'(b)', c='w', ha='center', va='center')
        ax02 = fig.add_subplot(main_grid[0,2])
        im2 = plot_phases(ax02,'sixth','z6')
        ax02.text(1,1,'(c)', c='w', ha='center', va='center')

        ax10 = fig.add_subplot(main_grid[1,0])
        _ = plot_gap(ax10, 'third', plot_x = True)
        ax10.text(1,1,'(d)', c='w', ha='center', va='center')
        ax11 = fig.add_subplot(main_grid[1,1])
        _ = plot_phases(ax11, 'third','z2', plot_x = True, xlabel = True)
        ax11.text(1,1,'(e)', c='w', ha='center', va='center')
        ax12 = fig.add_subplot(main_grid[1,2])
        _ = plot_phases(ax12, 'third', 'z6', plot_x = True)
        ax12.text(1,1,'(f)', c='w', ha='center', va='center')

        cbar_ax_0 = fig.add_subplot(main_grid[-1,0])
        fig.colorbar(im0, cax=cbar_ax_0,orientation='horizontal')
        cbar_ax_0.set_xlabel(r'$E$')
        cbar_ax_0.xaxis.set_label_coords(1.1,-0.5)

        cbar_ax_2 = fig.add_subplot(main_grid[-1,-1])
        cb2 = fig.colorbar(im2, cax=cbar_ax_2, orientation='horizontal')
        make_colbar_labels(cb2, 'z6')
        add_cbar_tag(cb2, 'z6')

        fig.savefig('output/lower_filling_zq_combo.png',bbox_inches='tight')
        return

    make_figure()
    return

if __name__ == "__main__":
    main()