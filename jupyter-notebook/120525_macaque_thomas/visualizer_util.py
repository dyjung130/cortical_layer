import os, sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import yaspy
notebook_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(notebook_dir, '../092325_cortical_gradients')))
from gradient_utils import *

# ====== COMPACT FUNCTIONS ======
def mapping_to_atlas(data_parc, atlas_data):
    mapped = np.full(len(atlas_data), np.nan)
    mapped[atlas_data != 0] = data_parc[atlas_data[atlas_data != 0] - 1]
    return mapped

def process_hemisphere(data_parc, atlas_data, surf_path, hemi, vmin=None, vmax=None, cmap='jet'):
    d = np.nanmean(data_parc, axis=1) if data_parc.ndim != 1 else data_parc
    mapped = mapping_to_atlas(d, atlas_data)
    vmin = np.round(np.nanmin(mapped), 2) if vmin is None else vmin
    vmax = np.round(np.nanmax(mapped), 2) if vmax is None else vmax
    p = yaspy.Plotter(surf_path, hemi=hemi)
    overlay = p.overlay(mapped, cmap=cmap, alpha=1, vmin=vmin, vmax=vmax)
    p.border(mapped, alpha=0)
    shots = [p.screenshot(v) for v in ("lateral", "medial")]
    return [[*shots, overlay]], vmin, vmax

def add_colorbar(ax, vmin, vmax, cmap):
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    plt.colorbar(sm, cax=make_axes_locatable(ax).append_axes("bottom", size="2.5%", pad=0.05), orientation="horizontal")

def plot_montage_with_colorbar(montage_img, vmin, vmax, cmap, title=None):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(montage_img)
    ax.axis('off')
    add_colorbar(ax, vmin, vmax, cmap)
    if title: ax.set_title(title, fontsize=16, fontweight='bold')
    fig.tight_layout()
    return fig, ax

def prepare_parc_data(thickness_types, atlas_lh):
    pd = {n: parcellate_data(t, atlas_lh).astype(float) for n, t in thickness_types}
    [arr.__setitem__(arr == 0, np.nan) for arr in pd.values()]
    pd['ratio'] = pd['supra'] / pd['total']
    pd['relative'] = pd['supra'] / pd['infra']
    return pd

def generate_figures_for_layers(layer_types, parc_data, atlas, surf):
    axes, titles = [], []
    for layer in layer_types:
        plotters, vmin, vmax = process_hemisphere(parc_data[layer], atlas, surf, 'lh')
        img =yaspy.montage([[p[0] for p in plotters], [p[1] for p in plotters]], pad=10, shareh=True, sharew=True)
        cmap = plotters[0][2].get_cmap()
        fig, ax = plot_montage_with_colorbar(img, vmin, vmax, cmap, title=layer.capitalize())
        axes.append((fig, ax))
        titles.append(layer.capitalize())
        plt.close(fig)
    return axes, titles

def plot_combined_row(axes_list, titles):
    fig, axs = plt.subplots(1, len(axes_list), figsize=(len(axes_list)*1.5, 2))
    axs = [axs] if len(axes_list)==1 else axs
    for i, (_, ax) in enumerate(axes_list):
        for im in ax.get_images(): axs[i].imshow(im.get_array())
        axs[i].set_axis_off()
        axs[i].set_title(titles[i], fontsize=12, fontweight='bold')
    fig.tight_layout()
    plt.show() 
