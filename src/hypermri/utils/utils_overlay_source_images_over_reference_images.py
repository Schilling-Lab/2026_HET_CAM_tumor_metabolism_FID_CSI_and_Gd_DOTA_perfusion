###########################################################################################################################################################

# Overlay of source images over reference images

###########################################################################################################################################################


# =========================================================================================================================================================
# Import necessary packages
# =========================================================================================================================================================

import os, sys

import numpy as np

from hypermri import Study


# =========================================================================================================================================================
# Definition of necessary functions
# =========================================================================================================================================================

def calculate_coordinate_system_transformation_matrix(reference_COSY_base_vector_list, source_COSY_base_vector_list):
    """
    Generate the coordinate system transformation matrix. Pass in the list of basis vectors of the reference and the source coordinate system.

    Parameters
    ----------
    reference_COSY_base_vector_list:   list of 1D numpy arrays of shape (3,): [et1, et2, et3] 
                                       The reference coordinate system (reference) defines the coordinate system for the alignment process. 
    source_COSY_base_vector_list:      list of 1D numpy arrays of shape (3,): [es1, es2, es3] 
                                       The source coordinate system (sensed) is the coordinate system of the image that should be aligned in the reference coordinate system.
    Returns
    -------
    source_to_reference_transformation_matrix A: 2D numpy array of shape (3,3)
                                                 reference_vector = np.dot(source_vector, A.T)
                                                 (3,)          = np.dot((3,),          (3,3))
    """
    return np.array(reference_COSY_base_vector_list) @ np.array(source_COSY_base_vector_list).T


def grid_coordinate_system_transformation(source_grid, source_to_reference_transformation_matrix):
    """
    Parameters
    ----------
    source_grid:   4D numpy array of shape (3,N1,N2,N3): 
                   Grid described in source (sensed) coordinate system of the image that should be aligned in the reference coordinate system.
    source_to_reference_transformation_matrix A: 2D numpy array of shape (3,3)
                                              reference_vector = np.dot(source_vector, A.T)
                                              (3,)          = np.dot((3,),          (3,3))                   
    Returns
    -------
    source_rps_grid_mm_transformed:   4D numpy array of shape (3,N1,N2,N3): 
                                   Grid described in reference (reference) coordinate system for the alignment process. 
    """
    return tuple(np.dot(np.array(source_grid).transpose(1,2,3,0), source_to_reference_transformation_matrix.T).transpose(3,0,1,2))
    

def determine_image_stack_rps_endpoint_grid_mm(BrukerExp):
    """
    Parameters
    ----------
    BrukerExp:   Bruker experiment object
                 Contains image stack.

    Returns
    -------
    rps_endpoint_grid_mm:   4D numpy array of shape (3,2,2,2): 
                            Grid describing extent endpoints of BrukerExp.
    """
    rps_extent_mm = BrukerExp.rps_extent_mm

    rps_endpoint_read_axis_mm  = np.array([rps_extent_mm[0], rps_extent_mm[1]])
    rps_endpoint_phase_axis_mm = np.array([rps_extent_mm[2], rps_extent_mm[3]])
    rps_endpoint_slice_axis_mm = np.array([rps_extent_mm[4], rps_extent_mm[5]])
    
    rps_endpoint_read_grid_mm, rps_endpoint_phase_grid_mm, rps_endpoint_slice_grid_mm = np.meshgrid(rps_endpoint_read_axis_mm, rps_endpoint_phase_axis_mm, rps_endpoint_slice_axis_mm, indexing='ij')
    rps_endpoint_grid_mm                                                              = (rps_endpoint_read_grid_mm, rps_endpoint_phase_grid_mm, rps_endpoint_slice_grid_mm)
    
    return rps_endpoint_grid_mm


def overlay_source_images_over_reference_images(reference_BrukerExp, source_BrukerExp):
    '''
    Parameters
    ----------
    reference_BrukerExp:   Bruker experiment object
                           Contains reference image stack.
    source_BrukerExp:      Bruker experiment object
                           Contains source image stack.

    Returns
    -------
    source transformation: dict {tuple, tuple, array}, shapes {}
                           Containing transformed (endpoint) grid and transformed reco_data numpy array.
    

    Examples
    -------
 
    '''
    
    reference_COSY_base_vector_list = list(reference_BrukerExp.acquisition_matrix)
    source_COSY_base_vector_list    = list(source_BrukerExp.acquisition_matrix)
    
    source_to_reference_transformation_matrix = calculate_coordinate_system_transformation_matrix(
                                                                                                  reference_COSY_base_vector_list=reference_COSY_base_vector_list, 
                                                                                                  source_COSY_base_vector_list=source_COSY_base_vector_list,
                                                                                                  )
    # print(source_to_reference_transformation_matrix)
                                                                                                                                                                                                    
    permutation_tuple        = source_to_reference_transformation_matrix    
    source_transpose_array   = np.zeros(3, dtype=int)
    
    for i, v in enumerate(permutation_tuple):
        for j, k in enumerate(v):
            if np.abs(np.abs(k) - 1) < 0.25:
                source_transpose_array[i] = j

    # print(source_transpose_array)

    source_rps_grid_mm                                 = source_BrukerExp.rps_grid_mm
    source_rps_endpoint_grid_mm                        = determine_image_stack_rps_endpoint_grid_mm(source_BrukerExp)                                                                                               
                                                       
    source_rps_grid_mm_transformed                     = grid_coordinate_system_transformation(source_grid=source_rps_grid_mm, source_to_reference_transformation_matrix=source_to_reference_transformation_matrix)
    source_rps_endpoint_grid_mm_transformed            = grid_coordinate_system_transformation(source_grid=source_rps_endpoint_grid_mm, source_to_reference_transformation_matrix=source_to_reference_transformation_matrix)
    
    source_rps_grid_mm_transformed_rearranged          = tuple(np.array(source_rps_grid_mm_transformed).transpose(0, source_transpose_array[0]+1, source_transpose_array[1]+1, source_transpose_array[2]+1))
    source_rps_endpoint_grid_mm_transformed_rearranged = tuple(np.array(source_rps_endpoint_grid_mm_transformed).transpose(0, source_transpose_array[0]+1, source_transpose_array[1]+1, source_transpose_array[2]+1))

   
    source_data_rearranged                             = source_BrukerExp.data_array.copy().to_numpy().transpose(0, 2, 1, 3, 4, 5, 6, 7, 8, 9).transpose(source_transpose_array[0], source_transpose_array[1], source_transpose_array[2], 3, 4, 5, 6, 7, 8, 9)
    # source_data_rearranged                             = source_BrukerExp.reco_array.copy().to_numpy().transpose(0, source_transpose_array[0]+1, source_transpose_array[1]+1, source_transpose_array[2]+1, 4, 5)
    
    source_rps_extent_mm_rearranged                    = [
                                                           source_rps_endpoint_grid_mm_transformed_rearranged[0][0,0,0], source_rps_endpoint_grid_mm_transformed_rearranged[0][1,0,0],
                                                           source_rps_endpoint_grid_mm_transformed_rearranged[1][0,0,0], source_rps_endpoint_grid_mm_transformed_rearranged[1][0,1,0],
                                                           source_rps_endpoint_grid_mm_transformed_rearranged[2][0,0,0], source_rps_endpoint_grid_mm_transformed_rearranged[2][0,0,1],
                                                         ]

    return {
        'source_transpose_array':                              source_transpose_array,                                 
        'source_rps_grid_mm_transformed_rearranged':           source_rps_grid_mm_transformed_rearranged,
        'source_rps_extent_mm_rearranged':                     source_rps_extent_mm_rearranged,
        'source_data_rearranged':                              source_data_rearranged,
    
    }