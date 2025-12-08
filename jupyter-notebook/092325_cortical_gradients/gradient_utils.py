# for registration data
from scipy.spatial.distance import cdist
import nibabel as nib
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore")



#parcellation of the data based on the given atlas (atlas_data).
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
        # This keeps values that are strictly less than 2*std from the mean.
        # If the intent is to include values *within* 2 std (inclusive), then <= should be used instead of <.
        # Additionally, if the standard deviation is zero, this will remove all but the exact mean.
        # This is a correct method to identify and filter outliers by the classic definition, but may not work as expected if parcel_data_std == 0 or parcel_data has few elements.

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


#this is a function to compute the centroid distances of all structures to left and right thalamus in a parcellated image 
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
 
