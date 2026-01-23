# for registration data

# Standard library imports
import json
import os
import subprocess
import sys
import tempfile
import warnings
import threading
import concurrent.futures
import logging

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

import nibabel as nib

from PIL import Image

from scipy.spatial.distance import cdist
from scipy.linalg import orthogonal_procrustes

from brainspace.gradient import GradientMaps
from brainspace.utils.parcellation import map_to_labels

import yaspy

warnings.filterwarnings("ignore")

#NOTE: 
def clean_thickness_data(data_array):
    """Clean NaN and Inf values from array using masked arrays."""
    masked_array = np.ma.masked_invalid(data_array)
    return np.array(masked_array.filled(0))

#LOAD HCP THICKNESS DATA (ONLY TOTAL THICKNESS)
def load_thickness_hcp_data(base_dir, subject_dir, hemi, suffix=None):
    """Load thickness measurements for HCP data."""
    file_stub = f"{base_dir}{subject_dir}/{subject_dir}."
    if hemi == 'lh':
        file_path = file_stub + ("L.thickness.32k_" + suffix + "_fwhm_fs_LR.shape.gii" if suffix is not None else "L.thickness.32k_fs_LR.shape.gii")
    else:
        file_path = file_stub + ("R.thickness.32k_" + suffix + "_fwhm_fs_LR.shape.gii" if suffix is not None else "R.thickness.32k_fs_LR.shape.gii")

    return {'total': nib.load(file_path).darrays[0].data}

#LOAD EX VIVO THICKNESS DATA (TOTAL, INFRAGRANULAR, SUPRAGRANULAR THICKNESS)
def load_thickness_exvivo_data(base_dir, subject_dir, hemi, suffix=None):
    """Load thickness measurements for ex vivo data."""
    file_stub = f'{base_dir}{subject_dir}/{hemi}.thickness.'
        
    file_path_infra = file_stub + f'wm.inf.32k_{suffix}_fwhm_fs_LR.shape.gii' if suffix is not None else file_stub + 'wm.inf.32k_fs_LR.shape.gii'
    file_path_supra = file_stub + f'inf.pial.32k_{suffix}_fwhm_fs_LR.shape.gii' if suffix is not None else file_stub + 'inf.pial.32k_fs_LR.shape.gii'
    file_path_total = file_stub + f'32k_{suffix}_fwhm_fs_LR.shape.gii' if suffix is not None else file_stub + '32k_fs_LR.shape.gii'
   
    return {
        'infra': nib.load(file_path_infra).darrays[0].data,
        'supra': nib.load(file_path_supra).darrays[0].data,
        'total': nib.load(file_path_total).darrays[0].data
    }

def load_thickness_bigbrain_data(base_dir, hemi):
    """Load thickness measurements from BigBrain."""
    return {
        'infra': nib.load(f'{base_dir}/{hemi}.3-6.32k.shape.gii').darrays[0].data,  # layers 4,5,6 (ex vivo style)
        'supra': nib.load(f'{base_dir}/{hemi}.0-3.32k.shape.gii').darrays[0].data,  # layers 1,2,3
        'total': nib.load(f'{base_dir}/{hemi}.0-6.32k.shape.gii').darrays[0].data   # all layers
    }

def calculate_derived_measurements(thickness_data):
    """Calculate derived thickness measurements."""
    return {
        'relative': clean_thickness_data(np.divide(thickness_data['supra'], thickness_data['infra'],
                                         out=np.zeros_like(thickness_data['supra']), where=thickness_data['infra'] != 0)),
        'ratio_supra': clean_thickness_data(np.divide(thickness_data['supra'], thickness_data['infra'] + thickness_data['supra'],
                                            out=np.zeros_like(thickness_data['supra']), where=thickness_data['infra'] + thickness_data['supra'] != 0)),
        'ratio_infra': clean_thickness_data(np.divide(thickness_data['infra'], thickness_data['infra'] + thickness_data['supra'],
                                            out=np.zeros_like(thickness_data['infra']), where=thickness_data['infra'] + thickness_data['supra'] != 0)),
        'diff': clean_thickness_data((thickness_data['infra'] - thickness_data['supra']) / (thickness_data['infra'] + thickness_data['supra']))
    }

#NOTE parcellation of the data based on the given atlas (atlas_data).
def parcellate_data(data,atlas_data):
    """
    data: numpy array
    atlas_data: numpy array
    hemisphere: string
    """

    #parcellate data
    len_unique = len(np.unique(atlas_data[atlas_data != 0]))
    data_parc = np.zeros((len_unique))
  
    for i in range(len_unique):
        parcel_data = data[atlas_data == i+1]
        # First remove NaN values
        parcel_data = parcel_data[~np.isnan(parcel_data)]
        if len(parcel_data) == 0:
            data_parc[i] = 0
            continue
        # Calculate mean and std of non-NaN values
        parcel_data_mean = np.nanmean(parcel_data)
        parcel_data_std = np.nanstd(parcel_data)
        # Keep only values within 2 std of mean
        mask = np.abs(parcel_data - parcel_data_mean) < (2 * parcel_data_std)
        filtered_data = parcel_data[mask]
        # Calculate final mean of filtered data
        if len(filtered_data) > 0:
            data_parc[i] = np.nanmean(filtered_data)
        else:
            data_parc[i] = 0
    return data_parc


#NOTE: This is a function to compute the centroid distances of all structures to left and right thalamus in a parcellated image 
#using the Tian parcellation volumetric atlas
def compute_centroid_distances(parcellated_img_path, thalamus_lh_label, thalamus_rh_label):
    """
    Compute centroid distance of all structures to left and right thalamus in a parcellated image.

    Parameters
    ----------
    parcellated_img_path : str
        Path to a parcellated image (nifti file).
    thalamus_lh_label : list of int
        Label(s) that define left thalamus.
    thalamus_rh_label : list of int
        Label(s) that define right thalamus.

    Returns
    -------
    dict
        {
            'label': [float, ...],
            'dist_lh': [float, ...],  # distance to LH thalamus centroid
            'dist_rh': [float, ...],  # distance to RH thalamus centroid
        }
    """
    parc_img = nib.load(parcellated_img_path)
    parc_data = parc_img.get_fdata()
    parc_labels = np.unique(parc_data)

    thalamus_lh_coords = np.argwhere(np.isin(parc_data, thalamus_lh_label))
    thalamus_rh_coords = np.argwhere(np.isin(parc_data, thalamus_rh_label))
    thalamus_lh_centroid = np.mean(thalamus_lh_coords, axis=0) if thalamus_lh_coords.size else np.array([np.nan]*3)
    thalamus_rh_centroid = np.mean(thalamus_rh_coords, axis=0) if thalamus_rh_coords.size else np.array([np.nan]*3)

    dist_lh_arr = np.empty(len(parc_labels))
    dist_rh_arr = np.empty(len(parc_labels))

    for i, label in enumerate(parc_labels):
        if label != 0:
            coords = np.argwhere(parc_data == label)
            if coords.size == 0:
                centroid = np.array([np.nan]*3)
            else:
                centroid = np.mean(coords, axis=0)
            dist_lh = cdist(centroid.reshape(1, -1), thalamus_lh_centroid.reshape(1, -1))[0][0] if not np.isnan(thalamus_lh_centroid).any() else np.nan
            dist_rh = cdist(centroid.reshape(1, -1), thalamus_rh_centroid.reshape(1, -1))[0][0] if not np.isnan(thalamus_rh_centroid).any() else np.nan
            dist_lh_arr[i] = dist_lh
            dist_rh_arr[i] = dist_rh
        else:
            dist_lh_arr[i] = np.nan
            dist_rh_arr[i] = np.nan

    return {
        'label': [float(l) for l in parc_labels],
        'dist_lh': [float(d) for d in dist_lh_arr],
        'dist_rh': [float(d) for d in dist_rh_arr]
    }
 


def calculate_gradients_from_brainspace(data2plot, mask_indices, atlas_data, hemisphere_mask, n_components, g_dimension_reduction='pca', g_sparsity = 0.9, g_kernel = 'normalized_angle'):
    """Process gradient maps for one hemisphere"""
    grad_all = []
    # Fit gradient maps
    gm = GradientMaps(n_components, approach=g_dimension_reduction, kernel=g_kernel)
    
    if np.isnan(mask_indices).all():
        gm.fit(np.nan_to_num(data2plot, 0),sparsity = g_sparsity)#sparsity density is 0.9 by default
    else:
        mask = np.ones(data2plot.shape[0], dtype=bool)
        mask[mask_indices] = False
        gm.fit(data2plot, sparsity = g_sparsity)

    # Process gradients
    grad = []
    #for each gradient component..
    for j in range(n_components):
        data_len = len(data2plot)
        if np.isnan(mask_indices).all():
            tmp_gm = gm.gradients_[:,j]
        else:
            tmp_gm = np.full((data_len, 1), np.nan)
            tmp_gm[mask] = gm.gradients_[:,j].reshape(-1,1)
            tmp_gm = tmp_gm.ravel()
            
        atlas_slice = atlas_data
        min_val = np.min(atlas_slice[atlas_slice != 0])
        max_val = np.max(atlas_slice[atlas_slice != 0])
        grad.append(map_to_labels(tmp_gm, atlas_slice, mask=hemisphere_mask, 
                                fill=np.nan))#, source_lab=np.arange(min_val,max_val+1)))

    return gm, grad

   

def unmap_gradient(grad_all_aligned, mask_indices, atlas_data, hemisphere_mask, start_idx, end_idx):
    """Unmap gradients to atlas data"""
    grad_all_unmapped = []
    for i, gm in enumerate(gm_all):
        grad = []
        for j in range(N_components):
            tmp_gm = gm.gradients_[:,j] if np.isnan(mask_indices).all() else \
                    np.full(len(data2plot[i]), np.nan)
            
            # Apply mask if needed
            if not np.isnan(mask_indices).all():
                mask = np.ones(data2plot[i].shape[0], dtype=bool)
                mask[mask_indices] = False
                tmp_gm[mask] = gm.gradients_[:,j]
            
            # Map to atlas labels
            atlas_slice = atlas_data[start_idx:end_idx]
            nonzero = atlas_slice[atlas_slice != 0]
            grad.append(label_to_map(tmp_gm, atlas_slice, mask=hemisphere_mask,
                                    fill=np.nan, source_lab=np.arange(nonzero.min(), nonzero.max()+1)))
    return grad

def map_gradients(gm_all, data2plot, mask_indices, atlas_data, hemisphere_mask, start_idx, end_idx):
    """Map gradients to atlas data"""
    grad_all_aligned = []
    for i, gm in enumerate(gm_all):
        grad = []
        for j in range(N_components):
            # Get gradient data
            tmp_gm = gm.gradients_[:,j] if np.isnan(mask_indices).all() else \
                    np.full(len(data2plot[i]), np.nan)
            
            # Apply mask if needed
            if not np.isnan(mask_indices).all():
                mask = np.ones(data2plot[i].shape[0], dtype=bool)
                mask[mask_indices] = False
                tmp_gm[mask] = gm.gradients_[:,j]
            
            # Map to atlas labels
            atlas_slice = atlas_data[start_idx:end_idx]
            nonzero = atlas_slice[atlas_slice != 0]
            grad.append(map_to_labels(tmp_gm, atlas_slice, mask=hemisphere_mask,
                                    fill=np.nan, source_lab=np.arange(nonzero.min(), nonzero.max()+1)))
        grad_all_aligned.append(grad)
    return grad_all_aligned



#NOTE: Wrapper function for spin test (because the gradient analysis code and enigma use different python version 3.9 vs 3.11)
def run_enigma_spin_test(map1, map2, n_rot=1000):
    """
    Run ENIGMA spin test using lami environment
    """
    # Create temp files
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f1:
        map1_file = f1.name
        np.save(map1_file, map1)
    
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f2:
        map2_file = f2.name
        np.save(map2_file, map2)
    
    try:
        # Call lami environment's Python
        result = subprocess.run([
            'conda', 'run', '-n', 'lami', 'python',
            'enigma_spin_test.py',
            map1_file,
            map2_file,
            str(n_rot)
        ], capture_output=True, text=True)
        
        # Debug: print everything if there's an issue
        if result.returncode != 0:
            print("=== Script failed ===")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise RuntimeError(f"ENIGMA script failed with return code {result.returncode}")
        
        # Get stdout and find JSON
        stdout = result.stdout.strip()
        
        if not stdout:
            print("=== Empty output ===")
            print("STDERR:", result.stderr)
            raise RuntimeError("ENIGMA spin test returned empty output")
        
        # Find the line that starts with { and ends with }
        json_line = None
        for line in stdout.split('\n'):
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                json_line = line
                break
        
        if json_line is None:
            print("=== Could not find JSON in output ===")
            print("Full STDOUT:")
            print(stdout)
            print("\nSTDERR:")
            print(result.stderr)
            raise ValueError("No valid JSON found in output")
        
        # Parse the JSON
        try:
            output = json.loads(json_line)
        except json.JSONDecodeError as e:
            print("=== JSON decode error ===")
            print("Tried to parse:", json_line)
            print("Error:", str(e))
            raise
        
        # Check if the script reported an error
        if not output.get('success', True):
            raise RuntimeError(f"ENIGMA error: {output.get('error', 'Unknown error')}")
        
        return output['p_value'], np.array(output['null_dist'])
    
    except subprocess.CalledProcessError as e:
        print("=== Subprocess error ===")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise
    
    finally:
        # Cleanup
        try:
            os.remove(map1_file)
        except:
            pass
        try:
            os.remove(map2_file)
        except:
            pass




#this is the main funciton used for gradient alignment -092525 DJ
def align_gradients(X,Y,reflection=False,rotation=False):
    """Align source gradients to target gradients
        X: source gradients
        Y: target gradients
    """
    
    if len(X) != len(Y):
        raise ValueError("Lists must be same length")
    
    #set nan to 0
    X[np.isnan(X)] = 0
    Y[np.isnan(Y)] = 0

    #center the matrix
    X_centered = X - np.mean(X, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)
    R, _ = orthogonal_procrustes(X_centered, Y_centered)

    if reflection == True and rotation == False:
        sign_matrix = np.sign(np.diag(R))  
        sign_transform = np.diag(sign_matrix)
        X_sign_corrected = X_centered @ sign_transform + np.mean(Y, axis=0)
        return X_sign_corrected, sign_transform
    elif reflection == True and rotation == True:
        X = X_centered @ R + np.mean(Y, axis=0)
        return X, R
    else:
        return X, Y
   

def create_hemisphere_plots(grad_all_aligned, surf_file, hemi, N_components_plot,color_ranges,cmap='RdBu_r'):
    """Create plots for one hemisphere"""
    plotters = []

    for pc in range(N_components_plot):
        plotter = yaspy.Plotter(surf_file, hemi=hemi)
        #[0] is some radii parameter I used before.. 
        #m = np.max(np.abs(([grad_all_aligned[0][pc]])))
        # Use percentiles but have the color scale centered at zero
        data = grad_all_aligned[0][:,pc]
        vmax = np.percentile(np.abs(data), 95)
        vmin = -vmax
        color_ranges[pc] = vmax
        overlay = plotter.overlay(data, cmap=cmap, alpha=1, vmin=vmin, vmax=vmax)
        plotter.border(grad_all_aligned[0][:,pc], alpha=0)
        plotters.append([plotter.screenshot("lateral"), plotter.screenshot("medial"), overlay])
        
    return plotters




def stitch_images_in_folder(savepath, image_names, outname='stitched.png', direction='horizontal'):

    # Example usage:
    # png_files = [f"{gene}.png" for gene in GENE_NAMES]
    # stitch_images_in_folder(savepath, png_files, outname='gene_exp_stitched.png', direction='horizontal')



    """
    Stitch together PNG images in savepath, either horizontally or vertically, and save as one PNG.
    
    Args:
        savepath (str): Folder containing the images.
        image_names (sequence): Iterable of PNG file names (e.g., ["PVALB.png",...])
        outname (str): Output filename for the stitched PNG.
        direction ('horizontal' or 'vertical'): Direction to stitch the images.
    """
    # Load all images
    images = [Image.open(os.path.join(savepath, f)) for f in image_names if os.path.splitext(f)[-1] == ".png" and os.path.exists(os.path.join(savepath, f))]
    print(images)
    if not images:
        print("No PNG files were found to stitch.")
        return None

    if direction == 'horizontal':
        total_width = sum(im.width for im in images)
        max_height = max(im.height for im in images)
        stitched_image = Image.new('RGBA', (total_width, max_height), (255, 255, 255, 0))
        x_offset = 0
        for im in images:
            stitched_image.paste(im, (x_offset, 0))
            x_offset += im.width
    elif direction == 'vertical':
        total_height = sum(im.height for im in images)
        max_width = max(im.width for im in images)
        stitched_image = Image.new('RGBA', (max_width, total_height), (255,255,255,0))
        y_offset = 0
        for im in images:
            stitched_image.paste(im, (0, y_offset))
            y_offset += im.height
    else:
        raise ValueError("direction must be 'horizontal' or 'vertical'.")

    output_png = os.path.join(savepath, outname)
    stitched_image.save(output_png)
    print(f"Saved stitched image as {output_png}")
    return output_png




# ==== 5. COMPUTE SPEARMAN'S CORRELATION ====
def compute_spearman_matrix(X, Y):
    """
    X: shape (n_parcels, n_x_features)
    Y: shape (n_parcels, n_y_features)
    Returns: corr & p-value matrices: shape (n_x_features, n_y_features)
    """
    n_x, n_y = X.shape[1], Y.shape[1]
    corrs = np.zeros((n_x, n_y))
    ps = np.zeros_like(corrs)
    for i in range(n_x):
        for j in range(n_y):
            r, p = spearmanr(X[:, i], Y[:, j], nan_policy='omit')
            corrs[i, j] = r
            ps[i, j] = p
    return corrs, ps

# ==== 6. SPINTEST SPEARMAN'S CORRELATION ====
def compute_spintest_matrix(X, Y, hemi, max_workers=3):
    """
    X: shape (n_parcels, n_x_features)
    Y: shape (n_parcels, n_y_features)
    Returns: corr & p-value matrices: shape (n_x_features, n_y_features)
    """
    # Redundant import removed, already imported at module level
    n_x, n_y = X.shape[1], Y.shape[1]
    corrs = np.zeros((n_x, n_y))
    ps = np.zeros_like(corrs)

    def spintest_job(args):
        i, j = args
        if hemi == 'lh':
            map1 = np.concatenate((X[:, i], np.full(len(X[:, i]), np.nan)))
            map2 = np.concatenate((Y[:, j], np.full(len(Y[:, j]), np.nan)))
        else:
            map1 = np.concatenate((np.full(len(X[:, i]), np.nan), X[:, i]))
            map2 = np.concatenate((np.full(len(Y[:, j]), np.nan), Y[:, j]))

        p_spin, null_dist = run_enigma_spin_test(map1, map2, n_rot=1000)
        return (i, j, p_spin)

    jobs = [(i, j) for i in range(n_x) for j in range(n_y)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, j, p_spin in executor.map(spintest_job, jobs):
            ps[i, j] = p_spin
    return ps



def plot_component_correlation(
    data,
    n_components=3,
    sparsity_index=0,
    vmin=-0.8,
    vmax=0.8,
    figsize_per_component=(4, 4),
    colormap='RdBu_r',
    fontsize=14,
    show_values=True,
    show_significance=False,
    alpha=0.05,
    corr_method='pearson',
    do_spin_test=False,
    hemi='lh',
    savepath=None
):
    """
    Plot correlation matrix for each gradient component across metrics, showing only the lower triangle.

    Parameters
    ----------
    data : dict
        Dictionary of aligned gradients; keys are metrics, values are lists with arrays
        of shape (n_vertices, n_components).
    n_components : int
        Number of components to plot.
    sparsity_index : int
        Index into the list for each metric.
    vmin, vmax : float
        Color scale limits.
    figsize_per_component : tuple
        Size of each subplot (width, height).
    colormap : str
        Matplotlib colormap name.
    fontsize : int
        Font size for labels and titles.
    show_values : bool
        Whether to display correlation values on the heatmap.
    show_significance : bool
        Whether to add asterisks for significant correlations.
    alpha : float
        Significance threshold (only used if show_significance=True).
    corr_method : str
        'pearson' or 'spearman' correlation.

    Returns
    -------
    None
    """

    # Metric labels
    keys = list(data.keys())
    labels = [
        "HCP: total",
        "Ex: total",
        "Ex: supra",
        "Ex: infra",
        "Ex: supra/total",
        "Ex: supra/infra"
    ]

    n_metrics = len(keys)
    sample_shape = data[keys[0]][sparsity_index].shape

    print(f"Metrics: {labels}")
    print(f"Array shape: {sample_shape} (n_vertices x n_components)")

    # Create figure
    fig, axes = plt.subplots(
        1, n_components,
        figsize=(n_components * figsize_per_component[0], figsize_per_component[1]),
        squeeze=False
    )

    corrmat_list = []
    pmat_list = []
    pmat_spin_list = []

    for component in range(n_components):
        # Extract component from each metric
        try:
            comp_matrix = np.column_stack([
                data[key][sparsity_index][:, component] for key in keys
            ])
        except IndexError as e:
            print(f"Component {component} out of bounds for shape {sample_shape}.")
            raise e

        print(f"Component {component} matrix shape: {comp_matrix.shape}")

        # Compute correlation and p-values
        n = comp_matrix.shape[0]
        corrmat = np.zeros((n_metrics, n_metrics))
        pmat = np.ones((n_metrics, n_metrics))
        pmat_spin = np.ones((n_metrics, n_metrics))

        for i in range(n_metrics):
            for j in range(n_metrics):
                if i < j:
                    # Remove top half
                    corrmat[i, j] = np.nan
                    pmat[i, j] = np.nan
                    pmat_spin[i, j] = np.nan
                elif i == j:
                    corrmat[i, j] = np.nan
                    pmat[i, j] = np.nan
                    pmat_spin[i, j] = np.nan
                else:
                    x, y = comp_matrix[:, i], comp_matrix[:, j]
                    try:
                        if corr_method == 'spearman':
                            # Ensure x and y are (vertices, 1) by reshaping if (vertices,)
                            corr, p = spearmanr(x, y, nan_policy='omit')
                        else:
                            corr, p = pearsonr(x, y)

                        if do_spin_test:
                            try:
                                # Added print for debugging
                                print('x',x.shape,'y',y.shape)
                                print(f"Running spin test for ({i},{j}), hemi={hemi}")
                                if x.ndim == 1:
                                    x = x.reshape(-1, 1)
                                if y.ndim == 1:
                                    y = y.reshape(-1, 1)
                                p_spin = compute_spintest_matrix(x, y, hemi, max_workers=4)
                            except Exception as spin_e:
                                print(f"Spin test failed for ({i},{j}): {spin_e}")
                                p_spin = np.nan
                        else:
                            p_spin = np.nan
                    except Exception as e:
                        print(f"Correlation calculation failed for ({i},{j}): {e}")
                        corr, p = np.nan, np.nan
                        p_spin = np.nan
                    corrmat[i, j] = corr
                    pmat[i, j] = p
                    pmat_spin[i, j] = p_spin
        pmat_list.append(pmat)
        pmat_spin_list.append(pmat_spin)
        corrmat_list.append(corrmat)
        '''
        # For each layer group (column), do FDR correction across the features (rows)
        for i in range(spearman_ps_lh.shape[1]):
            pvals = spearman_ps_lh[:, i]
            rej, pvals_fdr, _, _ = multipletests(pvals, alpha=ALPHA, method='fdr_bh')
            spearman_ps_fdr_lh[:, i] = pvals_fdr
            spearman_signif_lh[:, i] = rej  # True for significant
        '''

        # Plot heatmap
        ax = axes[0, component]
        im = ax.imshow(corrmat, cmap=colormap, vmin=vmin, vmax=vmax)

        # Put x and y tick marks and labels
        ax.set_xticks(np.arange(n_metrics))
        ax.set_yticks(np.arange(n_metrics))
        ax.set_xticklabels(labels, rotation=90, ha='center', fontsize=fontsize)
        ax.set_yticklabels(labels, fontsize=fontsize)
        ax.set_title(f'G{component+1}', fontsize=fontsize+2, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        # Add tick marks with minor ticks for better visual clarity
        ax.tick_params(axis='x', which='minor', length=4, color='gray')
        ax.tick_params(axis='y', which='minor', length=4, color='gray')
        ax.set_xticks(np.arange(n_metrics)-0.5, minor=True)
        ax.set_yticks(np.arange(n_metrics)-0.5, minor=True)
        ax.grid(False)  # No grid, just ticks

        # Annotate values, only on lower triangle
        if show_values or show_significance:
            for i in range(n_metrics):
                for j in range(n_metrics):
                    if i <= j:
                        # Skip top triangle and diagonal
                        continue
                    val = corrmat[i, j]
                    if np.isnan(val):
                        continue

                    # Determine text color
                    text_color = "white" if np.abs(val) > 0.5 else "black"

                    # Build annotation text
                    if show_values:
                        annot = f"{val:.2f}"
                    else:
                        annot = ""

                    # Add significance markers
                    if show_significance and not np.isnan(pmat[i, j]):
                        if do_spin_test:
                            p_tmp = pmat_spin[i, j]
                        else:
                            p_tmp = pmat[i, j]
                        if not np.isnan(p_tmp) and p_tmp < alpha:
                            if p_tmp < 0.001:
                                star = "***"
                            elif p_tmp < 0.01:
                                star = "**"
                            else:
                                star = "*"

                            if show_values:
                                annot += f"\n{star}"
                            else:
                                annot = star

                    if annot:
                        ax.text(
                            j, i + 0.1, annot,
                            ha='center', va='center',
                            color=text_color,
                            fontsize=fontsize * 0.7
                        )

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Correlation", fontsize=fontsize-2, rotation=270, labelpad=15)
        cbar.ax.tick_params(labelsize=fontsize)

    plt.tight_layout()

    if savepath is not None:
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        plt.savefig(os.path.join(savepath, "component_correlation.png"))
        #save pspin and pmat to the savepath
        np.save(os.path.join(savepath, "pmat_spin.npy"), pmat_spin_list)
        np.save(os.path.join(savepath, "pmat.npy"), pmat_list)
        np.save(os.path.join(savepath, "corrmat.npy"), corrmat_list)
        return pmat_spin_list, pmat_list, corrmat_list, fig
    else:
        return pmat_spin_list, pmat_list, corrmat_list, fig
    plt.show()


# Plot spearman
def plot_spearman_bar(
    spearman_corrs, 
    spearman_ps_fdr, 
    feature_ylabels, 
    feature_xlabels,
    fontsize=10,
    colorize_ylabels=None,
    do_sort= True
):
    """
    Plot horizontal bar plot for Spearman correlations in 1xN columns.
    
    Args:
        spearman_corrs:     N_features x N_columns array of correlation values
        spearman_ps_fdr:    N_features x N_columns array of FDR p-values
        FEATURE_YLABELS:    list or array of ytick (feature) labels, length N_features
        layer_type_labels:  list of N_columns names for columns (titles)
        fontsize:           int or float, all font size for axes/ticks/titles
    """
    # Sorting order for first column
    if do_sort:
        sort_order = np.argsort(spearman_corrs[:, 0])
        ytick_labels = np.array(FEATURE_YLABELS)[sort_order]
        if colorize_ylabels is not None:
            colorize_ylabels = np.array(colorize_ylabels)[sort_order]
        sorted_spearman_corrs = spearman_corrs[sort_order]
        sorted_spearman_ps_fdr = spearman_ps_fdr[sort_order]
    else:
        sorted_spearman_corrs = spearman_corrs
        sorted_spearman_ps_fdr = spearman_ps_fdr
        ytick_labels = FEATURE_YLABELS
        colorize_ylabels = colorize_ylabels

    n_plots = sorted_spearman_corrs.shape[1]
    n_features = sorted_spearman_corrs.shape[0]
    nrows, ncols = 1, n_plots

    fig_width_per_col = 2.0
    fig_height_per_row = 0.22 * n_features

    from matplotlib import gridspec

    left_label_pad_inches = 1.15
    fig = plt.figure(
        figsize=(left_label_pad_inches + ncols * fig_width_per_col, fig_height_per_row)
    )
    gs = gridspec.GridSpec(
        nrows=1,
        ncols=ncols+1,
        width_ratios=[left_label_pad_inches/fig_width_per_col] + [1]*ncols,
        wspace=0.1
    )
    axes = []
    for ci in range(ncols):
        axes.append(fig.add_subplot(gs[0, ci+1]))
    axes = np.array(axes)
    
    # Define combined direction & p bins as in original
    p_bins = [0, 0.001, 0.01, 0.05, 1.0001]
    p_bin_labels = ['<0.001', '<0.01', '<0.05', 'n.s.']
    n_pbins = len(p_bins) - 1
    n_total_bins = n_pbins * 2

    from matplotlib.colors import ListedColormap
    cmap_base = plt.get_cmap('coolwarm_r', 512)
    neg_colors = [cmap_base(i) for i in np.linspace(1, 0.55, n_pbins)]
    pos_colors = [cmap_base(i) for i in np.linspace(0, 0.45, n_pbins)]
    all_colors = neg_colors + pos_colors
    combo_cmap = ListedColormap(all_colors)

    def get_dir_pbin(rvals, pvals, p_bins):
        pbin = np.digitize(pvals, p_bins) - 1
        pbin = np.clip(pbin, 0, n_pbins-1)
        sign = (rvals > 0).astype(int)
        combo_bin = pbin + sign * n_pbins
        combo_bin = np.where(rvals == 0, pbin, combo_bin)
        return combo_bin, pbin

    for idx in range(ncols):
        ax = axes[idx]
        vals = sorted_spearman_corrs[:, idx]
        pvals = sorted_spearman_ps_fdr[:, idx]

        combo_bins, pbins = get_dir_pbin(vals, pvals, p_bins)
        bar_colors = np.array([
            combo_cmap(cb) if not (np.isnan(v) or np.isnan(pv)) else (0.6,0.6,0.6,1.0)
            for cb, v, pv in zip(combo_bins, vals, pvals)
        ])

        ylocs = np.arange(n_features)
        bars = ax.barh(
            ylocs, vals, color=bar_colors,
            edgecolor=None,
            linewidth=0.4,
            height=0.4
        )
        for i, (bar, pbin, v, pv) in enumerate(zip(bars, pbins, vals, pvals)):
            if np.isnan(v) or np.isnan(pv):
                continue
            if p_bin_labels[pbin] != 'n.s.':
                n_aster = n_pbins - pbin
                asterisks = '*' * (n_aster-1)
                bar_width = bar.get_width()
                if bar_width >= 0:
                    xloc = bar.get_x() + bar_width + 0.04
                    align = 'left'
                else:
                    xloc = bar.get_x() + bar_width - 0.04
                    align = 'right'
                yloc = bar.get_y() + bar.get_height() / 4
                ax.text(
                    xloc, yloc,
                    asterisks,
                    va='center',
                    ha=align,
                    color='k',
                    fontsize=fontsize,
                    fontweight='bold'
                )
        ax.set_xlim([-1.5,1.5])
        ax.set_xticks([-1, 0, 1])
        ax.set_title(FEATURE_XLABELS[idx], fontsize=fontsize, fontweight='bold')
        ax.set_yticks(ylocs)
        if idx == 0:
            if colorize_ylabels is not None:
                # Set yticklabels individually with colors from colorize_ylabel
                for ytick, label, color in zip(ax.get_yticks(), ytick_labels, colorize_ylabels):
                    ax.annotate(
                        label,
                        xy=(0, ytick),
                        xycoords=("axes fraction", "data"),
                        xytext=(-10, 0),
                        textcoords="offset points",
                        ha='right', va='center',
                        color=color,
                        fontsize=fontsize
                    )
                ax.set_yticks(ax.get_yticks())  # Keep locations, but clear default text labels
                ax.set_yticklabels([''] * len(ytick_labels))
            else:
                ax.set_yticklabels(ytick_labels, fontsize=fontsize)
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis='y', direction='out', length=2, width=2, left=True, labelsize=fontsize)
        ax.tick_params(axis='x', direction='out', length=2, width=2, bottom=True, labelsize=fontsize)
        ax.axvline(0, color='k', lw=1, ls='-')
        ax.set_xlabel("Spatial Correlation", fontsize=fontsize)

    # Optionally, add y axis only to the first axes, hide spines on left pad axis
    pad_ax = fig.add_subplot(gs[0,0])
    pad_ax.axis('off')
    # [pad_ax.spines[side].set_visible(False) for side in ["top","bottom","left","right"]]

    plt.subplots_adjust(
        left=left_label_pad_inches/(left_label_pad_inches + ncols*fig_width_per_col), 
        right=1, wspace=0.08
    )
    plt.tight_layout()
    plt.show()