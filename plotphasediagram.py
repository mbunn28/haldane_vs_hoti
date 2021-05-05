#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ti
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rc
from scipy.signal import argrelextrema
import matplotlib.patches as patches

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
plt.rcParams['axes.axisbelow'] = True

N_or_res = "res"
N = 600
path = "output/phasediagram/periodic"

# gap = joblib.load(f"{path}/{N_or_res}{N}_gap")
# x = np.linspace(0,2,num=1201)
# # x = joblib.load(f"{path}/{N_or_res}{N}_x")

# # hal_val = joblib.load(f"{path}/{N_or_res}{N}_hal_val")
# # alph_val = joblib.load(f"{path}/{N_or_res}{N}_alph_val")

# # gap0 = joblib.load(f"{path}/N{N}_gap_v0")
# # hal_val0 = joblib.load(f"{path}/N{N}_hal_val_v0")
# # alph_val = joblib.load(f"{path}/N{N}_alph_val_v0")

# # gap1 = joblib.load(f"{path}/N{N}_gap_v1")
# # hal_val1 = joblib.load(f"{path}/N{N}_hal_val_v1")

# # gap2 = joblib.load(f"{path}/N{N}_gap_v2")
# # hal_val2 = joblib.load(f"{path}/N{N}_hal_val_v2")

# # gap3 = joblib.load(f"{path}/N{N}_gap_v3")
# # hal_val3 = joblib.load(f"{path}/N{N}_hal_val_v3")

# # gap4 = joblib.load(f"{path}/N{N}_gap_v4")
# # hal_val4 = joblib.load(f"{path}/N{N}_hal_val_v4")

# # gap = np.concatenate((gap4,gap3), axis=0)
# # gap = np.concatenate((gap,gap2), axis=0)
# # gap = np.concatenate((gap,gap1), axis=0)
# # gap = np.concatenate((gap,gap0), axis=0)

# # hal_val = np.concatenate((hal_val4,hal_val3), axis=0)
# # hal_val = np.concatenate((hal_val,hal_val2), axis=0)
# # hal_val = np.concatenate((hal_val,hal_val1), axis=0)
# # hal_val = np.concatenate((hal_val,hal_val0), axis=0)

# gap[gap>1e-2]= np.NaN
# gap_mask = np.zeros((2*N+1,2*N+1),dtype=bool)
# for i in range(2*N+1):
#     gap_mask_row = np.zeros((2*N+1),dtype=bool)
#     gap_mask_row[argrelextrema(gap[:,i], np.less)[0]] = True
#     gap_mask[:,i] = gap_mask_row
# gap[gap_mask == False] = np.NaN
   
# x_mesh, y_mesh = np.meshgrid(x,x)
# x_mesh[gap_mask == False] = np.NaN
# y_mesh[gap_mask == False] = np.NaN
# x_mesh[:,-8:] = np.NaN
# y_mesh[:,-8:] = np.NaN

# # x_mesh = joblib.load(f"{path}/{N_or_res}{N}_xmesh")
# # y_mesh = joblib.load(f"{path}/{N_or_res}{N}_ymesh")

# x = np.zeros((1,1))
# x[0,0]=np.NaN
# y = np.zeros((1,1))
# y[0,0]=np.NaN


# m = 1
# num = np.array([0])
# for i in range(1201):
#     x_vals = x_mesh[:,i]
#     y_vals = y_mesh[:,i]
    
#     x_vals = x_vals[~np.isnan(x_vals)]
#     y_vals = y_vals[~np.isnan(y_vals)]

#     n = np.shape(x_vals)[0]
#     if n == m:
#         x_row = np.empty((np.shape(x)[0],1))
#         x_row[:] = np.NaN
#         y_row = np.empty((np.shape(y)[0],1))
#         y_row[:] = np.NaN

#         for j in range(len(num)):
#             x_row[num[j],0] = x_vals[j]
#             y_row[num[j],0] = y_vals[j]
        
#         x = np.append(x, x_row, axis=1)
#         y = np.append(y, y_row, axis=1)
#         # print(np.shape(x))

#     if (n != m and n > 0):
#         max = num[-1]
#         for j in range(n):
#             if j == 0:
#                 num = np.array([max+1])
#             else:
#                 num = np.append(num, num[-1]+1)

#         m = n
        
#         x_col =  np.empty((m,np.shape(x)[1]))
#         x_col[:] = np.NaN
        
#         x = np.append(x, x_col, axis=0)
#         y = np.append(y, x_col, axis=0)
#         # print(np.shape(x))

#         x_row = np.empty((np.shape(x)[0],1))
#         x_row[:] = np.NaN
#         y_row = np.empty((np.shape(y)[0],1))
#         y_row[:] = np.NaN

#         for j in range(len(num)):
#             x_row[num[j],0] = x_vals[j]
#             y_row[num[j],0] = y_vals[j]

#         x = np.append(x, x_row, axis=1)
#         y = np.append(y, y_row, axis=1)
#         # print(np.shape(x))

# x_row = np.empty((np.shape(x)[0],1))
# x_row[:] = np.NaN
# y_row = np.empty((np.shape(x)[0],1))
# y_row[:,0] = y[:,-1]

# for j in range(5):
#     x_row[-1-j,0] = 2

# x = np.append(x, x_row, axis=1)
# y = np.append(y, y_row, axis=1)
# y[-3,-1] = 1
# y[-4,-1] = 1

x = joblib.load(f"{path}/{N_or_res}{N}_x_to_plot")
y = joblib.load(f"{path}/{N_or_res}{N}_y_to_plot")


fig, ax = plt.subplots(figsize=(3.4,3.4))
for i in range(np.shape(x)[0]):
    plt.plot(x[i,:],y[i,:],c='k',lw=0.75)
# plt.pcolormesh(x,x,gap, norm = colors.LogNorm(), cmap='inferno')
# plt.scatter((2-0.693),(2-0.466),linewidth=0.1,marker='x')
# plt.title("Phase Boundary Diagram")
ax.grid(linestyle='--')
# ax.set_xlim([0,0.4])
ax.set_aspect(1)

labels=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, r'$\frac{1}{0.8}$',r'$\frac{1}{0.6}$',r'$\frac{1}{0.4}$',r'$\frac{1}{0.2}$',r'$\infty$']
locs=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0,1.2,1.4,1.6,1.8,2.0]
ax.set_yticklabels(labels)
ax.set_yticks(locs)
ax.set_ylabel(r'$\alpha$')
ax.set_xlim((0,2))
ax.set_ylim((0,2))
# ax1 = ax.twinx()
# labels1 =[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
# locs1 =[2.0, 1.8, 1.6, 1.4, 1.2, 1.0]
# ax1.set_yticklabels(labels1)
# ax1.set_yticks(locs1)
# ax1.set_ylabel(r'$\beta$')
# # ax1.set_axisbelow(True)
# ax1.grid(linestyle='--', zorder=0)


ax.set_xticklabels(labels)
ax.set_xticks(locs)
ax.set_xlabel(r'$\lambda$')

rect_ws = np.array([0.05,0.08,0.11,0.10,0.08,0.11,0.14,0.17,0.11,0.08,0.11,0.14])
rect_x_offsets = np.array([-0.004,-0.004,-0.004,-0.004,-0.004,-0.004,-0.004,-0.004,-0.004,-0.004,-0.004,-0.004])
rect_y_offsets = np.array([0.008,0.008,0.008,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01])
x_texts = np.array([0.1,0.1,0.6,0.73,0.9,1.6,1.4,1.87,1.9,1.9,1.6,0.905])
y_texts = np.array([1.9,0.1,1,1.87,1.57,1.8,1.67,1.58,1.3,0.7,0.2,0.3])
texts = ['I','II','III','IV','V','VI','VII','VIII','IX','X','XI','XII']
for k in range(len(rect_ws)):
    # define the rectangle size and the offset correction
    rect_w = rect_ws[k]
    rect_h = 0.09
    rect_x_offset = rect_x_offsets[k]
    rect_y_offset = rect_y_offsets[k]

    # text coordinates and content
    x_text = x_texts[k]
    y_text = y_texts[k]
    text = texts[k]

    # place the text
    ax.text(x_text, y_text, text, ha="center", va="center", zorder=10)
    # create the rectangle (below the text, hence the smaller zorder)
    rect = patches.FancyBboxPatch((x_text-rect_w/2+rect_x_offset, y_text-rect_h/2+rect_y_offset),
                            rect_w,rect_h,boxstyle=patches.BoxStyle("Round", pad=0.02),linewidth=1,edgecolor='orange',facecolor='wheat',zorder=9)
    # add rectangle to plot
    ax.add_patch(rect)

plt.arrow(1.4,1.67,-0.05,-0.095,color='dimgray',head_width=0.015,zorder=8)
plt.arrow(1.87,1.58,0,-0.09,color='dimgray',head_width=0.015,zorder=8)

print(y[x == 1.8])
# ax2 = ax.twiny()
# ax2.set_xlim(ax.get_xlim())
# ax2.set_xticklabels(labels1)
# ax2.set_xticks(locs1)
# ax2.set_xlabel('t')
# # ax2.set_axisbelow(True)
# ax2.grid(linestyle='--', zorder=0)


# cbar = plt.colorbar(pad = 0.15)
# cbar.ax.get_yaxis().labelpad = 15
# cbar.ax.set_title('Energy Gap')

# plt.gcf().subplots_adjust(top=0.85)
# fig.tight_layout()
print(fig.get_size_inches())
fig.savefig(f"{path}/{N_or_res}{N}_diagram_yo.png", dpi=500,bbox_inches='tight')

# fig1 = plt.figure()
# plt.pcolormesh(x, x, gap, norm = colors.LogNorm(), cmap='inferno')
# fig1.savefig(f"{path}/periodic.png", dpi=500)

# joblib.dump(x_mesh, f"{path}/{N_or_res}{N}_xmesh")
# joblib.dump(y_mesh, f"{path}/{N_or_res}{N}_ymesh")
# joblib.dump(x, f"{path}/{N_or_res}{N}_x_to_plot")
# joblib.dump(y, f"{path}/{N_or_res}{N}_y_to_plot")

