import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates
from scipy.stats import zscore


def sample_at_points(volume, points, affine):
    """Sample volume intensities at given points using trilinear interpolation"""
    inv_affine = np.linalg.inv(affine)
    homogeneous = np.column_stack([points, np.ones(len(points))])
    voxel_coords = (inv_affine @ homogeneous.T)[:3, :]
    return map_coordinates(volume.get_fdata(), voxel_coords, order=1, mode='constant', cval=0.0)


def generate_layer_intensity_profile(vol, layer_type, path_surf_norm, path_surf_coords, 
                                     save_path=None, sort_by_ap=True, spacing_mm=0.12,
                                     dist_max_mm=2, clim_max=3, cmap='RdBu_r', do_diff=False, fontsize = 9, dist_method=0, params = None):
                                     
    """Generate and plot intensity profiles at varying distances from cortical surface"""
    
    # Initialize params if not provided
    if params is None:
        params = {'dist_method': dist_method}
    
    # Load surface data
    surf_norm = nib.load(path_surf_norm)
    surf_coords = nib.load(path_surf_coords)
    
    # Extract coordinates and normals
    norm_xyz = np.array([surf_norm.darrays[i].data for i in range(3)])
    surf_xyz = np.array([surf_coords.darrays[i].data for i in range(3)])

    #method 1 
    if params['dist_method'] == 0:
        #calculate half voxel up (spacing mm/2) and down (spacing mm/2) from the surface along the surfarce normal 
        dist_array = np.flipud(np.concatenate([-np.arange(spacing_mm/2, dist_max_mm, spacing_mm)[::-1], np.arange(spacing_mm/2, dist_max_mm, spacing_mm)]))
    elif params['dist_method'] == 1:
        #calculate full voxel length along the surface normal
        dist_array = np.flipud(np.concatenate([-np.arange(spacing_mm, dist_max_mm + spacing_mm, spacing_mm)[::-1], [0], 
                                        np.arange(spacing_mm, dist_max_mm + spacing_mm, spacing_mm)]))
                                          
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
    
 
    return all_values, dist_array, ap_order



#NOTE: this function is used to process the intensity data
def process_intensity_data(data_array, data_type, do_zscore, do_zscore_before_diff=False, do_abs_on_diff=False):
    """Process intensity data with optional differencing and z-scoring."""
    
    if data_type == 'diff':
        # Apply z-scoring before differencing if requested
        if do_zscore_before_diff:
            if do_zscore:
                data_array = zscore(data_array, axis=0)
            data_array = (np.diff(data_array, axis=0))
            if do_abs_on_diff:
                data_array = np.abs(data_array)
        elif do_zscore_before_diff == False:
            data_array = (np.diff(data_array, axis=0))
            if do_abs_on_diff:
                data_array = np.abs(data_array)
            if do_zscore:
                data_array = zscore(data_array, axis=0)
        
    
            
    elif data_type == 'raw':
        # Apply z-scoring for raw data if requested
        if do_zscore:
            data_array = zscore(data_array, axis=0)
            
    else:
        raise ValueError(f"Invalid data_type: {data_type}. Must be 'diff' or 'raw'")

    return data_array