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
import matplotlib.patches as patches

def format_func(value, tick_number):
    if value <= 1:
        return f'{np.round(value,3)}'
    else:
        v = np.round(2 - value, 3)
        part1 = r'$\frac{1}{'
        part2 = r'}$'
        return fr'{part1}{v}{part2}'

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def define_alph_hal(params):
    if params[0] ==1:
        alphname = "b"
        alph = params[1]
    else:
        alphname = "a"
        alph = params[0]

    if params[2] == 1:
        halname = "t"
        hal = params[3]
    else:
        halname = "l"
        hal = params[2]
    return [alphname,alph,halname,hal]

def make_kxky(points):
    r_vals = np.arange(points)
    r1,r2 = np.meshgrid(r_vals,r_vals)
    b1 = np.array([0,4*np.pi*np.sqrt(3)/9])
    b2 = (2*np.pi/9)*np.array([3,np.sqrt(3)])
    kx = (b1[0]*r1 + b2[0]*r2)/points
    ky = (b1[1]*r1 + b2[1]*r2)/points
    return [kx,ky]

def get_chernnos(F):
    chern = np.sum(F,axis=(1,2))/(2*np.pi)
    chernnos = np.round(np.real(chern))
    chernnos[chernnos == 0] = 0
    return chernnos

def define_filepaths(params0, chern_points):
    alphname0,alph0,halname0,hal0 = define_alph_hal(params0)
    chern_path0 = f"output/chern/{halname0}{hal0}_{alphname0}{alph0}_res{chern_points}"
    return chern_path0

def main():
    
    # TODO implement argparse
    #params = (a, b, l ,t)

    #CHERN
    params = [[1,0.5,0.2,1],
                [1,0.08,0.57,1],
                [1,0.25,1,0.25]
            ]

    N_ribbon = 100
    res_ribbon = 125
    chern_points = 500
    N_ribbon_zoom = np.array([200,100,800])
    res_ribbon_zoom = np.array([200,150,500])
    y_lim = np.array([0.1,0.1,0.03])
    label_pos = np.array([2,3,5])
    zoom = np.array([False, True, False])
    name = 'chern_phases'


    # #HOTI
    # params = [[1,0.2,0.1,1],
    #             [0.2,1,0.1,1],
    #             [0.2,1,1,0.4]
    #         ]

    # N_ribbon = 100
    # res_ribbon = 125
    # chern_points = 500
    # N_ribbon_zoom = np.array([200,200,800])
    # res_ribbon_zoom = np.array([200,200,500])
    # y_lim = np.array([0.1,0.1,0.03])
    

    # zoom = np.array([False, False, False])
    # name = 'hoti_phases'
    # label_pos = np.array([0,1,10])

    # #APPENDIX 1
    # params = [[1,0.38,1,1],
    #             [1,0.466,1,0.693],
    #             [1,0.5595,1,0.2]
    #         ]

    # N_ribbon = 100
    # res_ribbon = 125
    # chern_points = 500
    # N_ribbon_zoom = np.array([200,200,800])
    # res_ribbon_zoom = np.array([200,200,500])
    # y_lim = np.array([0.1,0.1,0.03])

    # zoom = np.array([True, True, True])
    # name = 'appendix_phases_1'
    # label_pos = np.array([4,6,7])

    # #APPENDIX 2
    # params = [[1,0.6,1,0.4],
    #             [0.6,1,1,0.4],
    #             [0.22,1,0.8,1]
    #         ]

    # N_ribbon = 100
    # res_ribbon = 125
    # chern_points = 500
    # N_ribbon_zoom = np.array([100,100,100])
    # res_ribbon_zoom = np.array([125, 125, 125])
    # y_lim = np.array([0.15,0.2,0.1])

    # zoom = np.array([True, True, True])
    # name = 'appendix_phases_2'
    # label_pos = np.array([8,9,11])

    rect_ws = 5*np.array([0.05,0.08,0.10,0.08,0.11,0.14,0.17,0.11,0.08,0.11,0.11,0.14])
    rect_x_offsets = 5*np.array([-0.004,-0.004,-0.004,-0.004,-0.004,-0.004,-0.004,-0.004,-0.004,-0.004,-0.004,-0.004])
    rect_y_offsets = 5*np.array([0.008,0.008,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.008,0.01])
    # x_texts = np.array([0.1,0.1,0.6,0.73,0.9,1.6,1.4,1.87,1.9,1.9,1.6,0.905])
    # y_texts = np.array([1.9,0.1,1,1.87,1.57,1.8,1.67,1.58,1.3,0.7,0.2,0.3])
    texts = ['I','II','IV','V','VI','VII','VIII','IX','X','XI','III','XII']

    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    rc('text.latex', preamble= r'\usepackage{amsfonts}')

    def find_lims(array_type):
        vmax = 0
        vmin = 0
        for i in range(3):
            path = define_filepaths(params[i], chern_points)
            array = joblib.load(f'{path}/{array_type}')
            array = np.real(array)
            if np.amax(array) > vmax:
                vmax = np.amax(array)
            if np.amin(array) < vmin:
                vmin = np.amin(array)
        return [vmin, vmax]

    def plot_energy_band(ax, eigensys, do_xticks = False, write_xlabel = False, yaxis = None,return_color = False, plot_no = ''):
        kx,ky = make_kxky(chern_points)
        vmin, _ = find_lims('eigensys')
        ky = ky + 4*np.pi*np.sqrt(3)/9 + 0.3
        im = ax.pcolormesh(kx,ky,np.real(eigensys),cmap='plasma', vmin = vmin, vmax=0, shading='auto')
        # ax.set_aspect('equal')
        if write_xlabel == True:
            ax.set_xlabel(r'$k_x$')
        if do_xticks == False:
            ax.xaxis.set_visible(False)
        if do_xticks == True:
            ax.set_xticks([0,np.pi/3,2*np.pi/3])
            ax.set_xticklabels([0,r'$\frac{\pi}{3}$',r'$\frac{2 \pi}{3}$'])
            ax.get_xaxis().majorTicks[0].label1.set_horizontalalignment('left')
            ax.get_xaxis().majorTicks[2].label1.set_horizontalalignment('right')
        ax.set_xlim((-0.1,2*np.pi/3+0.1))
        diff = 0.3
        ax.set_ylim((-0.1,2*np.pi/(np.sqrt(3))+0.1+4*np.pi*np.sqrt(3)/9 + diff))
        if yaxis == None:
            ax.yaxis.set_visible(False)
        elif yaxis == 'left':
            ax.set_yticks([4*np.pi*np.sqrt(3)/9 + diff,2*np.pi/(3*np.sqrt(3))+ 4*np.pi*np.sqrt(3)/9 + diff,4*np.pi/(3*np.sqrt(3))+ 4*np.pi*np.sqrt(3)/9 + diff,2*np.pi/(np.sqrt(3))+ 4*np.pi*np.sqrt(3)/9 + diff])
            ax.set_yticklabels([0,r'$\frac{2\pi}{3\sqrt{3}}$',r'$\frac{4\pi}{3\sqrt{3}}$',r'$\frac{2\pi}{\sqrt{3}}$'])
            ax.set_ylabel(r'$k_y$')
        elif yaxis == 'right':
            ax.yaxis.tick_right()
            ax.set_yticks([0,2*np.pi/(3*np.sqrt(3)),4*np.pi/(3*np.sqrt(3)),2*np.pi/(np.sqrt(3))])
            ax.set_yticklabels([0,r'$\frac{2\pi}{3\sqrt{3}}$',r'$\frac{4\pi}{3\sqrt{3}}$',r'$\frac{2\pi}{\sqrt{3}}$'])
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(r'$k_y$')
        ax.text(-0.1-1.5,2*np.pi/(np.sqrt(3))+4*np.pi*np.sqrt(3)/9 + diff ,plot_no, weight='bold')
        if return_color == False:
            return
        else:
            return im

    def plot_chern_band(ax, Fimag, i, set_title = None, return_color = False):
        kx,ky = make_kxky(chern_points)
        vmin, vmax = find_lims('chern')
        im = ax.pcolormesh(kx,ky,Fimag,cmap='seismic', norm = colors.SymLogNorm(linthresh = 1e-5, vmin=vmin, vmax=vmax), shading='auto')
        # ax.set_aspect('equal')
        chernno = np.sum(Fimag)/(2*np.pi)
        chernno=np.round(chernno,0)
        if chernno==0:
            chernno = 0
        chernno = int(chernno)
        ax.text(2*np.pi/3,0,rf'$c_{i}={chernno}$',ha='right',va='bottom')
        if set_title != None:
            ax.title.set_text(f'$n={set_title}$')
        if return_color == False:
            return
        else:
            return im

    def plot_ribbon_spectra(ax, params, xlabel=False, write_ylabel = False, plot_no = '', res = res_ribbon, N = N_ribbon, ylim = None):
        aorb_name, aorb, torl_name, torl = define_alph_hal(params)
        newpath = f"output/ribbon_spectra/res{res}_N{N}_{aorb_name}{aorb}_{torl_name}{torl}"
        newpath = newpath.replace('.','')
        energies = joblib.load(f"{newpath}/energies")
        mask_left = joblib.load(f"{newpath}/left")
        mask_right = joblib.load(f"{newpath}/right")
        mask_other = np.logical_not(np.logical_or(mask_left,mask_right))
        k = np.linspace(-np.pi,np.pi,num=res)
        _, k =np.meshgrid(np.zeros(6*N),k)
        y_max = np.amax(energies)
        
        ax.scatter(k[mask_left],energies[mask_left],c='b',s=0.75,linewidth=0.3)
        ax.scatter(k[mask_right],energies[mask_right],c='r',s=0.75,linewidth=0.3)
        ax.scatter(k[mask_other],energies[mask_other],c='black',s=0.3,linewidth=0.06)
        if write_ylabel == True:
            ax.set_ylabel(r'$E$')
        ax.set_xlim((-np.pi,np.pi))
        if ylim == None:
            ax.set_ylim((-y_max-0.1,y_max+0.1))
        else:
            ax.set_ylim((-ylim,ylim))
        if xlabel == False:
            ax.xaxis.set_visible(False)
        else:
            ax.set_xlabel(r'$k$')
            ax.set_xticks((-np.pi,-np.pi/2,0,np.pi/2,np.pi))
            ax.set_xticklabels((r'$-\pi$',r'$-\frac{\pi}{2}$',0,r'$\frac{\pi}{2}$',r'$\pi$'))
            ax.get_xaxis().majorTicks[0].label1.set_horizontalalignment('left')
            ax.get_xaxis().majorTicks[4].label1.set_horizontalalignment('right')
        ax.text(-np.pi-1.5,y_max-0.25,plot_no, weight='bold')
        return

    def make_figure():
        w = 7.06
        h = 8.3
        fig = plt.figure(figsize=(w,h),dpi=500,constrained_layout=False)

        A = 10*np.pi/(3*np.sqrt(3)) + 0.4
        B = 2*np.pi/3 + 0.2
        mid_text_width = 0.7
        right_text_width = 0.7
        cbar_width = 0.2

        # print(w - mid_text_width - B*h/A - right_text_width -cbar_width)
        # print(B*h/A)
        widths = [w - mid_text_width - B*h/A - right_text_width -cbar_width, mid_text_width, B*h/(3*A), B*h/(3*A), B*h/(3*A), right_text_width, cbar_width]
        main_grid = gs.GridSpec(ncols=7, nrows=3, figure=fig, width_ratios=widths, wspace=0,hspace=0.05)

        if zoom[0] == False:        
            rib_ax0 = fig.add_subplot(main_grid[0,0])
            plot_ribbon_spectra(rib_ax0, params[0], plot_no = '(a)')
        else:
            zoom_grid = gs.GridSpecFromSubplotSpec(ncols = 1, nrows = 2, subplot_spec=main_grid[0,0], hspace=0, height_ratios = [0.75,0.25])
            rib_ax1_0 = fig.add_subplot(zoom_grid[0,0])
            rib_ax1_1 = fig.add_subplot(zoom_grid[1,0])
            plot_ribbon_spectra(rib_ax1_0, params[0], plot_no = '(a)', res = res_ribbon)
            plot_ribbon_spectra(rib_ax1_1, params[0], ylim = y_lim[0], res = res_ribbon_zoom[0], N = N_ribbon_zoom[0])
        
        if zoom[1] == False:
            rib_ax1 = fig.add_subplot(main_grid[1,0])
            plot_ribbon_spectra(rib_ax1, params[1], write_ylabel=True, plot_no = '(c)')
        else:
            zoom_grid = gs.GridSpecFromSubplotSpec(ncols = 1, nrows = 2, subplot_spec=main_grid[1,0], hspace=0, height_ratios = [0.75,0.25])
            rib_ax1_0 = fig.add_subplot(zoom_grid[0,0])
            rib_ax1_1 = fig.add_subplot(zoom_grid[1,0])
            plot_ribbon_spectra(rib_ax1_0, params[1], plot_no = '(c)')
            plot_ribbon_spectra(rib_ax1_1, params[1], ylim = y_lim[1], N = N_ribbon_zoom[1], res= res_ribbon_zoom[1])
            # ticks10 = rib_ax1_0.yaxis.get_major_ticks()
            # ticks10[0].label1.set_verticalalignment('bottom')
            # ticks11 = rib_ax1_1.yaxis.get_major_ticks()
            # ticks11[0].label1.set_verticalalignment('top')
        
        if zoom[2] == False:
            rib_ax2 = fig.add_subplot(main_grid[2,0])
            plot_ribbon_spectra(rib_ax2, params[2],xlabel=True, plot_no = '(e)')
        else:
            zoom_grid = gs.GridSpecFromSubplotSpec(ncols = 1, nrows = 2, subplot_spec=main_grid[2,0], hspace=0, height_ratios = [0.7,0.3])
            rib_ax2_0 = fig.add_subplot(zoom_grid[0,0])
            rib_ax2_1 = fig.add_subplot(zoom_grid[1,0])
            plot_ribbon_spectra(rib_ax2_0, params[2], plot_no = '(e)')
            plot_ribbon_spectra(rib_ax2_1, params[2], ylim = y_lim[2], N = N_ribbon_zoom[2], res= res_ribbon_zoom[2])
            ticks20 = rib_ax2_0.yaxis.get_major_ticks()
            ticks20[0].label1.set_verticalalignment('bottom')
            ticks21 = rib_ax2_1.yaxis.get_major_ticks()
            ticks21[-1].label1.set_verticalalignment('top')

        chern_path0 = define_filepaths(params[0],chern_points)
        eigensys0 = joblib.load(f'{chern_path0}/eigensys')
        chern0 = joblib.load(f'{chern_path0}/chern')
        chern_ax0_0 = fig.add_subplot(main_grid[0,2])
        plot_energy_band(chern_ax0_0, eigensys0[:,:,0,0], yaxis='left', plot_no = '(b)')
        plot_chern_band(chern_ax0_0, chern0[0,:,:], 1, set_title = 1)
        chern_ax0_1 = fig.add_subplot(main_grid[0,3])
        plot_energy_band(chern_ax0_1, eigensys0[:,:,0,1])
        plot_chern_band(chern_ax0_1, chern0[1,:,:], 2, set_title = 2)
        chern_ax0_2 = fig.add_subplot(main_grid[0,4])
        plot_energy_band(chern_ax0_2, eigensys0[:,:,0,2],yaxis='right')
        plot_chern_band(chern_ax0_2, chern0[2,:,:], 3, set_title = 3)
        
        chern_path1 = define_filepaths(params[1],chern_points)
        eigensys1 = joblib.load(f'{chern_path1}/eigensys')
        chern1 = joblib.load(f'{chern_path1}/chern')
        chern_ax1_0 = fig.add_subplot(main_grid[1,2])
        plot_energy_band(chern_ax1_0, eigensys1[:,:,0,0], yaxis='left', plot_no = '(d)')
        plot_chern_band(chern_ax1_0, chern1[0,:,:], 1)
        chern_ax1_1 = fig.add_subplot(main_grid[1,3])
        plot_energy_band(chern_ax1_1, eigensys1[:,:,0,1], write_xlabel=True)
        plot_chern_band(chern_ax1_1, chern1[1,:,:], 2)
        chern_ax1_2 = fig.add_subplot(main_grid[1,4])
        plot_energy_band(chern_ax1_2, eigensys1[:,:,0,2],yaxis='right')
        plot_chern_band(chern_ax1_2, chern1[2,:,:], 3)

        chern_path2 = define_filepaths(params[2],chern_points)
        eigensys2 = joblib.load(f'{chern_path2}/eigensys')
        chern2 = joblib.load(f'{chern_path2}/chern')
        chern_ax2_0 = fig.add_subplot(main_grid[2,2])
        plot_energy_band(chern_ax2_0, eigensys2[:,:,0,0], do_xticks = True, yaxis='left', plot_no = '(f)')
        plot_chern_band(chern_ax2_0, chern2[0,:,:], 1)
        chern_ax2_1 = fig.add_subplot(main_grid[2,3])
        plot_energy_band(chern_ax2_1, eigensys2[:,:,0,1], do_xticks = True, write_xlabel=True)
        plot_chern_band(chern_ax2_1, chern2[1,:,:], 2)
        chern_ax2_2 = fig.add_subplot(main_grid[2,4])
        im0 = plot_energy_band(chern_ax2_2, eigensys2[:,:,0,2], do_xticks = True, yaxis='right', return_color=True)
        im1 = plot_chern_band(chern_ax2_2, chern2[2,:,:], 3, return_color=True)

        axs = [chern_ax0_0, chern_ax1_0, chern_ax2_0]
        for i in range(3):
            # define the rectangle size and the offset correction
            k = label_pos[i]
            rect_w = rect_ws[k]
            rect_h = 0.09*5
            rect_x_offset = rect_x_offsets[k]
            rect_y_offset = rect_y_offsets[k]

            # text coordinates and content
            x_text = -0.1-0.8
            y_text = (2*np.pi/(np.sqrt(3))+4*np.pi*np.sqrt(3)/9)/4
            text = texts[k]

            # place the text
            axs[i].text(x_text, y_text, text, ha="center", va="center", zorder=10)
            # create the rectangle (below the text, hence the smaller zorder)
            rect = patches.FancyBboxPatch((x_text-rect_w/2+rect_x_offset, y_text-rect_h/2+rect_y_offset),
                                    rect_w,rect_h,boxstyle=patches.BoxStyle("Round", pad=0.02),linewidth=1,edgecolor='orange',facecolor='wheat',zorder=9, clip_on=False)
            # add rectangle to plot
            axs[i].add_patch(rect)

        cbar_ax_0 = fig.add_subplot(main_grid[0,-1])
        fig.colorbar(im0, cax=cbar_ax_0)
        cbar_ax_0.title.set_text(r'$E$')
        # cbar_ax_0.set_ylabel(r'$E$',rotation=0, fontsize=12)
        # cbar_ax_0.yaxis.set_label_coords(-0.05,-0.5)
        cbar_ax_1 = fig.add_subplot(main_grid[-1,-1])
        cbar_ax_1.title.set_text(r'Im$(F_{12})$')
        # cbar_ax_1.set_ylabel(r'$F_{12}$',rotation=0, fontsize=12)
        # cbar_ax_1.yaxis.set_label_coords(1.06,-0.5)
        fig.colorbar(im1, cax=cbar_ax_1)
        fig.savefig(f'output/{name}.png',bbox_inches='tight')
        return

    make_figure()
    return

if __name__ == "__main__":
    main()