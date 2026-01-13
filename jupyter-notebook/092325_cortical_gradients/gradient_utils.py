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
