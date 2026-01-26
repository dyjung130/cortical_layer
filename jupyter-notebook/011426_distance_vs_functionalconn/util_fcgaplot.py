'''
Utility functions for plotting FCGA output (plot_fcga_output.ipynb)
'''
# --- Self-contained Utility Functions ---

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import importlib.util
def extract_mm_value(fname):
    """
    Extracts the integer 'mm' value from a filename (e.g., "subject_10mm.npy" -> 10).
    Returns float('inf') if no match is found.
    """
    m = re.search(r'(\d+)mm', fname)
    return int(m.group(1)) if m else float('inf')

def import_gradient_utils(import_path):
    """
    Import gradient_utils.py from a relative path robustly.
    Returns the loaded module.
    """
   
    module_file = os.path.abspath(
        os.path.join(os.getcwd(), import_path)
    )
    if not os.path.isfile(module_file):
        raise ImportError(f"gradient_utils.py not found at: {module_file}")
    spec = importlib.util.spec_from_file_location("gradient_utils", module_file)
    gradient_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gradient_utils)
    print(f"Successfully imported from: {module_file}")
    return gradient_utils

def fix_data_shape(data, nonzero_indices, n_vertices=32492):
    """
    Map the given data to a surface of `n_vertices`, filling zero elsewhere.
    Args:
        data: shape (N_nonzero_vertices, n_components)
        nonzero_indices: index locations of vertices corresponding to `data`
        n_vertices: full surface vertex count (default: 32492 for 32k fs_LR)
    Returns:
        data_fixed: shape (n_vertices, n_components)
    """
    data_fixed = np.zeros((n_vertices, data.shape[1]), dtype=data.dtype)
    data_fixed[nonzero_indices, :] = data
    return data_fixed

def plot_gradients(gradient_utils, grad_data, lh_surf, n_components=10, cmap='RdBu_r'):
    """
    Plot left hemisphere gradients using the gradient_utils plotting function.
    Args:
        gradient_utils: imported gradient_utils module
        grad_data: dict or np.ndarray for surface data
        lh_surf: path to left hemisphere surface file
        n_components: number of PCs/components
    Returns:
        fig: matplotlib.figure.Figure
    """
    color_ranges = [0.1] * n_components
    plotters = gradient_utils.create_hemisphere_plots(
        grad_data, lh_surf, hemi='lh', N_components_plot=n_components, color_ranges=color_ranges, cmap=cmap
    )

    num_pcas = n_components
    fig, axs = plt.subplots(2, num_pcas, figsize=(2 * num_pcas, 4))  # lateral & medial

    for i in range(num_pcas):
        ax_lateral = axs[0, i] if num_pcas > 1 else axs[0]
        ax_lateral.imshow(plotters[i][0])  # lateral
        ax_lateral.set_title(f'Component {i + 1}', fontsize=11)
        ax_lateral.axis('off')

        ax_medial = axs[1, i] if num_pcas > 1 else axs[1]
        ax_medial.imshow(plotters[i][1])  # medial
        ax_medial.axis('off')

    plt.tight_layout()
    return fig

def parcellate_data_matrix(data_fixed, atlas_data_lh, gradient_utils):
    """
    Parcellate each principal component in data_fixed onto atlas regions.
    Args:
        data_fixed: (n_vertices, n_components)
        atlas_data_lh: 1D array, LH atlas labels for each vertex
        gradient_utils: imported gradient_utils module (provides parcellate_data)
    Returns:
        parcellated_matrix: (n_parcels, n_components) -- mean value in each parcel for each component
        unique_labels: array of nonzero atlas parcel labels, shape (n_parcels,)
    """
    n_pcs = data_fixed.shape[1]
    unique_labels = np.unique(atlas_data_lh)
    unique_labels = unique_labels[unique_labels != 0]
    n_parcels = len(unique_labels)
    parcellated_matrix = np.zeros((n_parcels, n_pcs))
    for pc in range(n_pcs):
        # parcellate_data expects (values, labels), returns mean in each parcel
        values = gradient_utils.parcellate_data(data_fixed[:, pc], atlas_data_lh)
        parcellated_matrix[:, pc] = values
    return parcellated_matrix, unique_labels

def backproject_to_surface_from_parcels(parcellated_matrix, atlas_data_lh, unique_labels):
    """
    Back-project parcelwise values (each component) to a full-vertex surface array.
    Args:
        parcellated_matrix: (n_parcels, n_components)
        atlas_data_lh: (n_vertices,)
        unique_labels: (n_parcels,)
    Returns:
        reconstructed_data: (n_vertices, n_components)
    """
    n_vertices = atlas_data_lh.shape[0]
    n_components = parcellated_matrix.shape[1]
    reconstructed_data = np.zeros((n_vertices, n_components), dtype=parcellated_matrix.dtype)
    for pc in range(n_components):
        for lbl_idx, lbl in enumerate(unique_labels):
            reconstructed_data[atlas_data_lh == lbl, pc] = parcellated_matrix[lbl_idx, pc]
    return reconstructed_data

def plot_gradients_on_surface(gradient_utils, grad_this, lh_surf, N_components_plot=10, cmap='RdBu_r'):
    """
    Plot all components of the gradients for the left hemisphere using the surface plotting utility.
    Args:
        gradient_utils: imported gradient_utils module
        grad_this: array or dict for surface data
        lh_surf: path to left hemisphere surface file
        N_components_plot: number of gradients/components to plot
    Returns:
        fig: matplotlib.figure.Figure
    """
    color_ranges = [0.1]*N_components_plot
    plotters_lh = gradient_utils.create_hemisphere_plots(
        grad_this, lh_surf, hemi='lh', N_components_plot=N_components_plot, color_ranges=color_ranges, cmap=cmap
    )

    num_pcas = N_components_plot
    fig, axs = plt.subplots(2, num_pcas, figsize=(2*num_pcas, 4))  # two rows: lateral, medial

    for i in range(num_pcas):
        ax_lateral = axs[0, i] if num_pcas > 1 else axs[0]
        ax_lateral.imshow(plotters_lh[i][0])  # 0 is lateral
        ax_lateral.set_title(f'Component {i+1}', fontsize=11)
        ax_lateral.axis('off')

        ax_medial = axs[1, i] if num_pcas > 1 else axs[1]
        ax_medial.imshow(plotters_lh[i][1])  # 1 is medial
        ax_medial.axis('off')

    plt.tight_layout()
    return fig