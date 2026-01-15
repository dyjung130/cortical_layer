# for registration data
import json
import os
import subprocess
import sys
import tempfile
import warnings
from PIL import Image
import nibabel as nib
import numpy as np
from scipy.spatial.distance import cdist
from brainspace.gradient import GradientMaps
from brainspace.utils.parcellation import map_to_labels
from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr
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
            print(data[atlas_data == i+1])
            print('no data',i)
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
 



def calculate_gradients_from_brainspace(data2plot, mask_indices, atlas_data, hemisphere_mask, n_components, g_dimension_reduction='pca', g_sparsity = 0.9, g_kernel='normalized_angle'):
    """Process gradient maps for one hemisphere"""
    grad_all = []
    
    
    # Fit gradient maps
    gm = GradientMaps(n_components, approach=g_dimension_reduction, kernel=g_kernel)
    
    if np.isnan(mask_indices).all():
        print('no mask')
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
        #print(f"Atlas range: {min_val}-{max_val}")

        grad.append(map_to_labels(tmp_gm, atlas_slice, mask=hemisphere_mask, 
                                fill=np.nan))#, source_lab=np.arange(min_val,max_val+1)))

    return gm, grad


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

    print('X',X.shape)
    print('Y',Y.shape)

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
