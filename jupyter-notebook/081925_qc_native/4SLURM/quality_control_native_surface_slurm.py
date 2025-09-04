"""
Quality control script for analyzing MRI cortical layer intensities
Usage: python quality_control_native_surface_slurm.py <subject_name> <base_path> [hemisphere]

This script is designed to be run as part of a SLURM job array.
It analyzes intensity differences at varying distances from cortical surfaces.

Required dependencies:
- numpy
- nibabel 
- scipy
- matplotlib
- os
- sys
"""

import sys
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates
import os
import matplotlib
# Set non-interactive backend since we're saving plots to files
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#params for the mainscript
params = {'sort_by_ap': True, 'spacing_mm': 0.12, 'dist_max_mm': 0.48, 'clim_max': 1, 'do_diff':True, 'fontsize': 9}

def sample_at_points(volume, points, affine):
    """Sample volume intensities at given points using trilinear interpolation"""
    inv_affine = np.linalg.inv(affine)
    homogeneous = np.column_stack([points, np.ones(len(points))])
    voxel_coords = (inv_affine @ homogeneous.T)[:3, :]
    return map_coordinates(volume.get_fdata(), voxel_coords, order=1, mode='constant', cval=0.0)

def generate_layer_intensity_profile(vol, layer_type, path_surf_norm, path_surf_coords, 
                                     save_path=None, sort_by_ap=True, spacing_mm=0.12,
                                     dist_max_mm=2, clim_max=3, cmap='RdBu_r', do_diff=False, fontsize = 9):
                                     
    """Generate and plot intensity profiles at varying distances from cortical surface"""
    
    # Load surface data
    surf_norm = nib.load(path_surf_norm)
    surf_coords = nib.load(path_surf_coords)
    
    # Extract coordinates and normals
    norm_xyz = np.array([surf_norm.darrays[i].data for i in range(3)])
    surf_xyz = np.array([surf_coords.darrays[i].data for i in range(3)])
    dist_array = np.flipud(np.concatenate([-np.arange(spacing_mm/2, dist_max_mm, spacing_mm)[::-1], 
                                          np.arange(spacing_mm/2, dist_max_mm, spacing_mm)]))
    all_points = [surf_xyz.T + norm_xyz.T * d for d in dist_array]
    all_values = np.array([sample_at_points(vol, p, vol.affine) for p in all_points])
    
    # Sort by AP direction
    if sort_by_ap:
        ap_order = np.argsort(surf_xyz[1])[::-1]
        all_values = all_values[:,ap_order]
        x_label_title = 'Vertices in AP direction'
    else:
        ap_order = np.arange(all_values.shape[1])
        x_label_title = 'Vertices (not ordered)'
    
    if do_diff:
        #added 082725 noarmzlize and calculate diff
        le_data = all_values
        #normalize the intensity values along columns
        #le_data = (le_data - np.mean(le_data, axis=0)) / np.std(le_data, axis=0)
        # Calculate the first-order gradient of the intensity values from top to bottom
        le_data = np.diff(le_data, axis=0)
        #y_extent = [-spacing_mm * le_data.shape[0]/2, spacing_mm * (le_data.shape[0]/2-1)]#previously I set to -1 but it's not correct
        y_extent = [-spacing_mm * le_data.shape[0]/2, spacing_mm * (le_data.shape[0]/2)]#
    else:
        le_data = all_values
        #normalize the intensity values along columns
        #le_data = (le_data - np.mean(le_data, axis=0)) / np.std(le_data, axis=0)
        y_extent = [-spacing_mm * le_data.shape[0]/2, spacing_mm * (le_data.shape[0]/2)]
    
    fig = plt.figure(figsize=(8, 1))
    im = plt.imshow(le_data, aspect='auto', cmap=cmap, 
                   extent=[0, le_data.shape[1], y_extent[0], y_extent[1]])
    
    # Add colorbar and formatting
    cbar = plt.colorbar(im, shrink=0.8)

    if do_diff:
        cbar.set_label('Intensity diff', fontsize=fontsize-1, rotation=270, labelpad=10)
    else:
        cbar.set_label('Intensity', fontsize=fontsize-1, rotation=270, labelpad=10)

    plt.clim(-clim_max, clim_max)
    cbar.set_ticks([-clim_max, 0, clim_max])
    cbar.ax.set_yticklabels([f'<-{clim_max}', '0', f'>{clim_max}'],fontsize=fontsize)
    
    # Configure axes
    y_ticks_pos = np.arange(0, y_extent[1], spacing_mm)
    y_ticks_neg = np.arange(0, y_extent[0], -spacing_mm)[1:]
    y_ticks = np.concatenate([y_ticks_neg, y_ticks_pos])
    plt.yticks(y_ticks,fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.ylabel(f'rel. {layer_type} (mm)',fontsize=fontsize)
    plt.xlabel(x_label_title,fontsize=fontsize)
    plt.axhline(y=0, color='black', linewidth=0.5, linestyle=':')
    
    # Add direction labels
    ax2 = plt.gca().twinx()
    ax2.set_ylim(y_extent)
    ax2.set_yticks([spacing_mm, spacing_mm*len(le_data)])
    ax2.set_yticklabels(['neg norm', 'pos norm'], rotation=-90, fontsize=fontsize-3)
    ax2.get_yticklabels()[0].set_color('#2166AC')
    ax2.get_yticklabels()[1].set_color('#B2182B')
    
    # Save plot if path provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        if do_diff:
            plt.savefig(f'{save_path}/{layer_type}.diff.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f'{save_path}/{layer_type}.raw.png', dpi=300, bbox_inches='tight')
        
    # Clean up figure
    plt.close(fig)
    
    return all_values, le_data, dist_array, ap_order

def main():
    # Parse command line arguments
    if len(sys.argv) < 3:
        print("Usage: python quality_control_native_surface_slurm.py <subject_name> <base_path> [hemisphere]")
        sys.exit(1)
    
    subject_name = sys.argv[1]
    base_path = sys.argv[2] 
    hemi = sys.argv[3] if len(sys.argv) > 3 else 'lh'
    
    print(f"Processing subject: {subject_name}")
    print(f"Base path: {base_path}")
    print(f"Hemisphere: {hemi}")
    
    # Set up paths
    save_path = './figures/'
    save_path_subject = os.path.join(save_path, subject_name, hemi)
    
    # Create directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path_subject, exist_ok=True)
    
    # Check if subject directory exists
    subject_path = os.path.join(base_path, subject_name)
    if not os.path.exists(subject_path):
        print(f"Error: Subject directory {subject_path} not found")
        sys.exit(1)
    
    # Handle compressed/uncompressed native.nii files
    vol_path = os.path.join(subject_path, 'mri', 'native.nii')
    vol_path_gz = os.path.join(subject_path, 'mri', 'native.nii.gz')
    
    if os.path.exists(vol_path):
        print(f"Loading {vol_path}")
        vol = nib.load(vol_path)
    elif os.path.exists(vol_path_gz):
        print(f"Loading {vol_path_gz}")
        vol = nib.load(vol_path_gz)
    else:
        print(f"Error: Neither {vol_path} nor {vol_path_gz} found")
        sys.exit(1)
    
    print(f"Loaded volume shape: {vol.shape}")
    
    # Main processing
    try:
        results = {}
       
        
        for layer_type in ['inf', 'white', 'pial']:
            files_path = os.path.join(base_path, subject_name, f'{hemi}.{layer_type}.32k_fs_LR')
            
            # Check if required surface files exist
            surf_norm_path = f'{files_path}.surfnorm.func.gii'
            surf_coord_path = f'{files_path}.coord.func.gii'
            
            if not os.path.exists(surf_norm_path):
                print(f"Warning: {surf_norm_path} not found, skipping {layer_type}")
                continue
            if not os.path.exists(surf_coord_path):
                print(f"Warning: {surf_coord_path} not found, skipping {layer_type}")
                continue
                
            print(f"Processing layer: {layer_type}")
            #calculate intensity profile
            raw, le_data, dist_array, ap_order = generate_layer_intensity_profile(
                vol, layer_type,
                surf_norm_path,
                surf_coord_path,
                save_path_subject, **params
            )
            results[f'{layer_type}_raw_intensity'] = raw
            key = f'{layer_type}_{"diff" if params["do_diff"] else "raw"}_intensity'
            results[key] = le_data
            results[f'{layer_type}_ap_order'] = ap_order
            
        
        # Save results
        if results:
            params['dist_array'] = dist_array
            results['params'] = params
            output_file = os.path.join(save_path_subject, f"intensity_{'diff' if params['do_diff'] else 'raw'}_{int(params['spacing_mm']*1000)}um_results.npz")
            np.savez(output_file, **results)
            print(f"Results saved to: {output_file}")
        else:
            print(f"No results to save for subject {subject_name}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error processing subject {subject_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"Successfully completed processing for subject {subject_name}")

if __name__ == "__main__":
    main()