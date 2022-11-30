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

def main():
    
    # TODO implement argparse
    #params = (a, b, l ,t)

    N = 16

    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    rc('text.latex', preamble= r'\usepackage{amsfonts}')

    def make_figure():
        w = 7.06
        h = w
        fig, axs = plt.subplots(3,3,figsize=(w,h),dpi=500, constrained_layout=False,sharey='row')
        plt.subplots_adjust(hspace=0.05,wspace=0.05)

        lattice0 = ti.Lattice(PBC_i = False, PBC_j = False,cornertype = 'Five Sites', a = 0, b = 1, l = 0, t = 1, N=N)
        lattice0.colourcode = True
        lattice0.corner_p = 0.3
        lattice0.edge_p = 2    
        lattice0.energy_spectrum(indep='l',t=250,min_val=0,max_val=1, ax=axs[0,0])
        
        axs[0,0].xaxis.set_ticks([])
        axs[0,0].set_ylabel(r'$E$')

        lattice1 = ti.Lattice(PBC_i = False, PBC_j = False,cornertype = 'Five Sites', a = 0, b = 1, l = 1, t = 1, N=N)
        lattice1.colourcode = True
        lattice1.corner_p = 0.3
        lattice1.edge_p = 2
        lattice1.energy_spectrum(indep='t',t=250,min_val=0,max_val=1,ax=axs[0,1])
        axs[0,1].xaxis.set_ticks([])
        # axs[0,1].yaxis.set_ticks([])

        lattice2 = ti.Lattice(PBC_i = False, PBC_j = False,cornertype = 'Five Sites', a = 1, b = 1, l = 0, t = 1, N=N)
        lattice2.colourcode = True
        lattice2.corner_p = 0.3
        lattice2.edge_p = 2
        lattice2.energy_spectrum(indep='a',t=250,min_val=0,max_val=1,ax=axs[0,2])
        axs[0,2].xaxis.set_ticks([])
        # axs[0,1].yaxis.set_ticks([])

        _, y_top0 = axs[0,0].get_ylim()
        axs[0,0].text(0.05,y_top0*0.9,r'(a) $\alpha$ = 0',ha='left', va='top')
        axs[0,1].text(1.95,y_top0*0.9,r'$\alpha$ = 0 (b)',ha='right', va='top')
        axs[0,2].text(0.05,y_top0*0.9,r'(c) $\lambda = 0$',ha='left', va='top')

        lattice3 = ti.Lattice(PBC_i = False, PBC_j = False,cornertype = 'Five Sites', a = 0.1, b = 1, l = 0, t = 1, N=N)
        lattice3.colourcode = True
        lattice3.corner_p = 0.3
        lattice3.edge_p = 0.55    
        lattice3.energy_spectrum(indep='l',t=250,min_val=0,max_val=1, ax=axs[1,0])
        axs[1,0].xaxis.set_ticks([])
        axs[1,0].set_ylabel(r'$E$')

        lattice4 = ti.Lattice(PBC_i = False, PBC_j = False,cornertype = 'Five Sites', a = 0.1, b = 1, l = 1, t = 1, N=N)
        lattice4.colourcode = True
        lattice4.corner_p = 0.3
        lattice4.edge_p = 0.55
        lattice4.energy_spectrum(indep='t',t=250,min_val=0,max_val=1,ax=axs[1,1])
        axs[1,1].xaxis.set_ticks([])
        # axs[0,1].yaxis.set_ticks([])

        lattice5 = ti.Lattice(PBC_i = False, PBC_j = False,cornertype = 'Five Sites', a = 1, b = 1, l = 0.1, t = 1, N=N)
        lattice5.colourcode = True
        lattice5.corner_p = 0.3
        lattice5.edge_p = 0.55
        lattice5.energy_spectrum(indep='a',t=250,min_val=0,max_val=1,ax=axs[1,2])
        axs[1,2].xaxis.set_ticks([])
        # axs[0,1].yaxis.set_ticks([])

        _, y_top1 = axs[1,0].get_ylim()
        axs[1,0].text(0.05,y_top1*0.9,r'(d) $\alpha$ = 0.1',ha='left', va='top')
        axs[1,1].text(1.95,y_top1*0.9,r'$\alpha$ = 0.1 (e)',ha='right', va='top')
        axs[1,2].text(0.05,y_top1*0.9,r'(f) $\lambda = 0.1$',ha='left', va='top')

        lattice6 = ti.Lattice(PBC_i = False, PBC_j = False,cornertype = 'Five Sites', a = 0.4, b = 1, l = 0, t = 1, N=N)
        lattice6.colourcode = True
        lattice6.corner_p = 0.3
        lattice6.edge_p = 0.55    
        lattice6.energy_spectrum(indep='l',t=250,min_val=0,max_val=1, ax=axs[2,0])
        axs[2,0].xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        axs[2,0].set_xlabel(r'$\lambda$')
        axs[2,0].set_ylabel(r'$E$')
        ticks0 = axs[2,0].xaxis.get_major_ticks()
        ticks0[-1].label1.set_horizontalalignment('right')
        ticks0[0].label1.set_horizontalalignment('left')

        lattice7 = ti.Lattice(PBC_i = False, PBC_j = False,cornertype = 'Five Sites', a = 0.4, b = 1, l = 1, t = 1, N=N)
        lattice7.colourcode = True
        lattice7.corner_p = 0.3
        lattice7.edge_p = 0.55
        lattice7.energy_spectrum(indep='t',t=250,min_val=0,max_val=1,ax=axs[2,1])
        axs[2,1].xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        axs[2,1].set_xlabel(r'$\lambda$')
        ticks1 = axs[2,1].xaxis.get_major_ticks()
        ticks1[-1].label1.set_horizontalalignment('right')
        ticks1[0].label1.set_horizontalalignment('left')

        lattice8 = ti.Lattice(PBC_i = False, PBC_j = False,cornertype = 'Five Sites', a = 1, b = 1, l = 1, t = 0.8, N=N)
        lattice8.colourcode = True
        lattice8.corner_p = 0.3
        lattice8.edge_p = 0.55
        lattice8.energy_spectrum(indep='a',t=250,min_val=0,max_val=1,ax=axs[2,2])
        axs[2,2].xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        axs[2,2].set_xlabel(r'$\alpha$')
        ticks2 = axs[2,2].xaxis.get_major_ticks()
        ticks2[-1].label1.set_horizontalalignment('right')
        ticks2[0].label1.set_horizontalalignment('left')

        _, y_top2 = axs[2,0].get_ylim()
        axs[2,0].text(0.05,0.9*y_top2,r'(g) $\alpha$ = 0.4',ha='left', va='top')
        axs[2,1].text(1.95,0.9*y_top2,r'$\alpha$ = 0.4 (h)',ha='right', va='top')
        axs[2,2].text(0.05,0.9*y_top2, r'(i) $\lambda = \frac{1}{0.8}$',ha='left', va='top')

        fig.savefig('output/slice_spectra_combo.png',bbox_inches='tight')
        return

    make_figure()
    return

if __name__ == "__main__":
    main()