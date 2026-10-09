###########################################################################################################################################################

# Overlay of CSI fit results over high-resolution images

###########################################################################################################################################################


# =========================================================================================================================================================
# Import necessary packages
# =========================================================================================================================================================

import os, sys

import numpy as np
import pandas as pd

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Circle

from hypermri import Study

from hypermri.utils.plot.utils_plot import add_scalebar
from hypermri.utils.image.interpolation import img_interp
from hypermri.utils.plot.utils_plot import extent_to_slice


# =========================================================================================================================================================
# Definition of necessary functions
# =========================================================================================================================================================

def plot_fit_results_overview(
    CSI_BrukerExp,
    fit_spectra=None,
    fit_chemical_shift_axis_ppm=None,
    fit_amps=None,  
    rep=None,
    CSI_data=None,
    marked_voxel=None,
    anatomical_BrukerExp=None,
    image_stack=None,
    anatomical_image_numbers=None,
    mask=None,
    mask_interpolation=None, 
    mask_image_number=None,    
    plot_params={},
    fit_params={},
):
    '''
    Plot an overview of the CSI fit results
    Parameters
    ----------
    CSI_BrukerExp: Bruker experiment object
        Contains CSI data
    fit_spectra: nd numpy array
        Array of fitted spectra
    fit_chemical_shift_axis_ppm: 1d numpy array
        Array of chemical shifts sampled in fitted_spectra
    fit_amps: nd numpy array
        Array of fitted amplitudes
    rep: int/range
        Repetition to be plotted
    CSI_data: nd numpy array
        The CSI data that should be plotted in overlay with fit_spectra
    marked_voxel: [i,j] tuple
        Indices of marked spectrum in overlay
    anatomical_BrukerExp: Bruker experiment object
        Contains anatomical reference images
    anatomical_image_numbers: list(int)
        Number of slices chosen for overlay (indexing starts at 0)
    plot_params: dict
        Plot parameters

    Returns
    -------
    A multisubplot figure

    Examples
    -------

    '''
    
    if CSI_data is None and fit_spectra is None:
        raise ValueError('Pass either CSI_data or/and fit_spectra!')

    chemical_shift_axis_ppm = CSI_BrukerExp.reco_data.attrs['freq_range_ppm']

    if fit_spectra is None:
        fit_spectra_passed                      = False
        fit_spectra                             = np.zeros_like(CSI_data)
        fit_chemical_shift_axis_ppm             = np.zeros_like(chemical_shift_axis_ppm)
    
        linewidth_fit                           = plot_params.get('linewidth_fit', 0)
        plot_params['linewidth_fit']            = linewidth_fit
        alpha_fit                               = plot_params.get('alpha_fit', 0)
        plot_params['alpha_fit']                = alpha_fit
        linewidth_measurement                   = plot_params.get('linewidth_measurement', 0.5)
        plot_params['linewidth_measurement']    = linewidth_measurement
        alpha_measurement                       = plot_params.get('alpha_measurement', 1)
        plot_params['alpha_measurement']        = alpha_measurement
    else:
        fit_spectra_passed = True

    if CSI_data.shape[1:3] != fit_spectra.shape[1:3]:
        raise ValueError(
            f'CSI_data and fit_spectra need to have same x-y dimensions but have'
            f'CSI_data: {CSI_data.shape[1:3]} and '
            f'fit_spectra: {fit_spectra.shape[1:3]}'
        )
    if anatomical_BrukerExp is None:
        raise ValueError('Please supply anatomical_BrukerExp')

    # check if data is complex, if yes apply ||
    if np.issubdtype(CSI_data.dtype, np.complexfloating):
        CSI_data = np.abs(CSI_data)

    if np.issubdtype(fit_spectra.dtype, np.complexfloating):
        fit_spectra = np.abs(fit_spectra)

    # assume that the data is real/imaginary part in case it goes below zero:
    if np.min(CSI_data) < 0 or np.min(fit_spectra) < 0:
        plot_min = min(np.min(CSI_data), np.min(fit_spectra))
    else:
        plot_min = 0

    # decide which repetitions to plot:
    if rep is None:
        rep = range(0, 1)
    elif type(rep) == int:
        rep = range(rep, rep + 1)

    # extract plot parameters:
    if plot_params is None:
        plot_params = {}

    sample_ID                           = plot_params.get('sample_ID', '')
    figsize                             = plot_params.get('figsize', (8, 6.5))
    plot_title                          = plot_params.get('plot_title', None)
    
    t_start_CSI_after_inj_s             = plot_params.get('t_start_CSI_after_inj_s', 0)
    
    scaling_factor                      = plot_params.get('scaling_factor', 1)
    ylim                                = plot_params.get('ylim', (0, 1e7))        
    xlim                                = plot_params.get('xlim', (190, 165))

    linewidth_measurement               = plot_params.get('linewidth_measurement', 0.5)
    linewidth_fit                       = plot_params.get('linewidth_fit', 0.25)

    linecolor_measurement               = plot_params.get('linecolor_measurement', 'y')
    linecolor_fit                       = plot_params.get('linecolor_fit', 'r')

    alpha_measurement                   = plot_params.get('alpha_measurement', 1)
    alpha_fit                           = plot_params.get('alpha_fit', 1)

    alpha_CSI_amps                      = plot_params.get('alpha_CSI_amps', [0.5, 0.5, 0.5])
    
    if marked_voxel is None:
        marked_voxel = plot_params.get('marked_voxel', None)
        if marked_voxel is None:
            marked_voxel = (0, 0)
            plot_params['marked_voxel'] = marked_voxel
            
        if not isinstance(marked_voxel, tuple):
            try:
                marked_voxel = tuple(marked_voxel)
            except:
                pass
    
    show_tqdm                           = plot_params.get('show_tqdm', True)
    
    show_mask                           = plot_params.get('show_mask', False) 
    show_mask_map                       = plot_params.get('show_mask_map', False) 
    show_mask_contour                   = plot_params.get('show_mask_contour', False) 
    show_mask_interpolation             = plot_params.get('show_mask_interpolation', False) 

    # vmin_anatomical_image               = plot_params.get('vmin_anatomical_image', None)
    # vmax_anatomical_image               = plot_params.get('vmax_anatomical_image', None)
    vmin_anatomical_percentile          = plot_params.get('vmin_anatomical_percentile', 0)
    vmax_anatomical_percentile          = plot_params.get('vmax_anatomical_percentile', 100)
    
    vmin_CSI_amps                       = plot_params.get('vmin_lac_map', [None, None])
    vmax_CSI_amps                       = plot_params.get('vmin_lac_map', [None, None])
    vmin_CSI_amp_ratio                  = plot_params.get('vmin_CSI_amp_ratio', 0)
    vmax_CSI_amp_ratio                  = plot_params.get('vmax_CSI_amp_ratio', 1)
    
    CSI_extent                          = plot_params.get('CSI_extent', [CSI_BrukerExp.rps_extent_mm[2], CSI_BrukerExp.rps_extent_mm[3], CSI_BrukerExp.rps_extent_mm[0], CSI_BrukerExp.rps_extent_mm[1]])
    anatomical_image_extent             = plot_params.get('anatomical_image_extent', CSI_extent)
    mask_extent                         = plot_params.get('mask_extent', anatomical_image_extent)

    thresholds                          = plot_params.get('thresholds', [5, 1])
    use_percentile_thresholds           = plot_params.get('use_percentile_thresholds', False)

    metabolite_colors                   = plot_params.get('metabolite_colors', ['black', 'red'])
    colormap_list                       = plot_params.get('colormap_list', ['jet', 'jet', 'jet'])
    interpolation_factor                = plot_params.get('interpolation_factor', 2)
    
    # vmin_anatomical_image_percentile    = plot_params.get('vmin_anatomical_image_percentile', 0)
    # vmax_anatomical_image_percentile    = plot_params.get('vmax_anatomical_image_percentile', 100)
    vmin_percentiles                    = plot_params.get('vmin_percentiles', [0, 0, 0])
    vmax_percentiles                    = plot_params.get('vmax_percentiles', [100, 100, 100])
    
    save_fig                            = plot_params.get('save_fig', False)
    save_dpi                            = plot_params.get('save_dpi', 500)
    fig_transparent                     = plot_params.get('fig_transparent', True)
    save_path                           = plot_params.get('save_path', CSI_BrukerExp.path)
    save_format                         = plot_params.get('save_format', '.png')

    x_axis_mm                           = np.linspace(CSI_extent[0] + 1/2 * (CSI_extent[1] - CSI_extent[0])/CSI_data.shape[2], 
                                                      CSI_extent[1] - 1/2 * (CSI_extent[1] - CSI_extent[0])/CSI_data.shape[2], 
                                                      CSI_data.shape[2])
    y_axis_mm                           = np.linspace(CSI_extent[2] + 1/2 * (CSI_extent[3] - CSI_extent[2])/CSI_data.shape[1], 
                                                      CSI_extent[3] - 1/2 * (CSI_extent[3] - CSI_extent[2])/CSI_data.shape[1], 
                                                      CSI_data.shape[1])
                                                      
    x                                   = marked_voxel[1]
    y                                   = marked_voxel[0]
        
    if np.any(image_stack):
        image_stack = image_stack
    else:
        image_stack = np.squeeze(np.abs(anatomical_BrukerExp.data_array)) 
        
    for r in tqdm(rep, disable=not show_tqdm):
        fig, ax = plt.subplots(figsize=figsize)
        ax.clear()
    
        for anatomical_image_number in anatomical_image_numbers:

            if anatomical_image_extent[1] > anatomical_image_extent[0] and anatomical_image_extent[3] > anatomical_image_extent[2]: 
                CSI_slice = extent_to_slice(image_stack[:, :, anatomical_image_number].shape, anatomical_image_extent, CSI_extent, origin='lower')
                vmin_anatomical_image = np.nanpercentile(image_stack[:, :, anatomical_image_number][CSI_slice], vmin_anatomical_percentile)
                vmax_anatomical_image = np.nanpercentile(image_stack[:, :, anatomical_image_number][CSI_slice], vmax_anatomical_percentile)
            else:
                vmin_anatomical_image = np.nanpercentile(image_stack[:, :, anatomical_image_number], vmin_anatomical_percentile)
                vmax_anatomical_image = np.nanpercentile(image_stack[:, :, anatomical_image_number], vmax_anatomical_percentile)
            
            fig, ax = plt.subplots(4, 3, figsize=figsize, gridspec_kw={'width_ratios': [1, 1, 0.1], 'height_ratios': [1, 1, 1, 1]})            
                
            for i in tqdm(range(4), disable=not show_tqdm):
                for j in tqdm(range(3), disable=not show_tqdm):
                    
                    ax[i][j].clear()
                    
                    circle = Circle(((CSI_extent[1] + CSI_extent[0])/2, (CSI_extent[3] + CSI_extent[2])/2), 12, transform=ax[i][j].transData)
                    
                    if i==0 and j==0:
                    
                        anatomical_image = image_stack[:, :, anatomical_image_number]
                        
                        subplot = ax[i][j].imshow(anatomical_image, 
                                                  cmap='bone', 
                                                  vmin=vmin_anatomical_image, 
                                                  vmax=vmax_anatomical_image, 
                                                  extent=anatomical_image_extent,
                                                  origin='lower')
                                                                          
                        if show_mask_contour:
                            mask_contour = ax[i][j].contour(
                                                    mask[:,:,mask_image_number], 
                                                    levels=[0.5], 
                                                    colors='magenta', 
                                                    linewidths=1, 
                                                    extent=mask_extent, 
                                                    origin='lower'
                                                    )                        
                        
                        # line1    = ax[i][j].axvline(x_axis_mm[x], c='magenta', linestyle='dashed', linewidth=1)
                        # line2    = ax[i][j].axhline(y_axis_mm[y], c='magenta', linestyle='dashed', linewidth=1)
                        
                        # line1.set_clip_path(circle)
                        # line2.set_clip_path(circle)
                        subplot.set_clip_path(circle)
                        
                        ax[i][j].set_title(r'$^{1}$H $T_{2}$-w. image')
                        ax[i][j].axis('off')
                        
                        ax[i][j].set_xlim(CSI_extent[0], CSI_extent[1])
                        ax[i][j].set_ylim(CSI_extent[2], CSI_extent[3])
                                            
                        divider = make_axes_locatable(ax[i][j])  
                        cb_ax   = divider.append_axes('bottom', size='5%', pad=0.05)  
                        cb_ax.set_visible(False)
                        
                        # iax     = inset_axes(cb_ax, width='75%', height='100%', loc='center', borderpad=0)    
                        # cbar    = fig.colorbar(subplot, cax=iax, orientation='horizontal', ticks=[vmin, vmid, vmax], format='%.1e')
                        
                        # image_start_time = '\n' + r'$t_{\mathrm{after~inj}}$ $\in$ = ' + '{0:.0f} s'.format(r * CSI_BrukerExp.frame_time_s[0] + t_start_CSI_after_inj_s)
                        # cbar.set_label(r'$^{13}$C FID amplitude' + '\n[a. u.]')
                        # cbar.ax.ticklabel_format(style='sci', scilimits=(-3, 3))
            
                        add_scalebar(
                            px=1,
                            ax=ax[i][j],
                            units='mm',
                            fixed_value=5,
                            color='white',
                            box_alpha=0.0,
                            box_color='w',
                            location='lower center',
                            frameon=True,
                            pad=0.075,
                            length_fraction=None,
                            border_pad=0.075,
                            fontsize=8,
                        )
                        
                        ################################################################################################################################################### 
                        sp_ax   = divider.append_axes('left', size='110%', pad=0.8)                        
                        sp_ax.set_visible(False)
                        
                        spiax   = inset_axes(sp_ax, width='100%', height='100%', loc='center', borderpad=20)
                        spiax.set_visible(False)
                                         
                    elif j==0:
                    
                        if i==3:                        
                            reference = img_interp(np.squeeze(np.abs(fit_amps[:, :, 0, r, 0, 0])) * scaling_factor,
                                               interp_factor=interpolation_factor,
                                               interp_method='bilinear',
                                               threshold=0)                                                  
                            if use_percentile_thresholds:
                                threshold = np.nanpercentile(reference, vmin_percentiles[0])
                            else:
                                threshold = thresholds[0]                        
                            image_interpolation = img_interp(np.squeeze(np.abs(fit_amps[:, :, 0, r, 0, 1]/fit_amps[:, :, 0, r, 0, 0])),
                                                             interp_factor=interpolation_factor,
                                                             interp_method='bilinear',
                                                             threshold=0)  
                            image= np.where(reference > threshold, image_interpolation, np.nan)                
                                            
                            ax[i][j].set_title(r'HP [1-$^{13}$C] lac/pyr')
                            
                            subplot1 = ax[i][j].imshow(anatomical_image,
                                                    cmap='bone',
                                                    extent=anatomical_image_extent, 
                                                    vmin=vmin_anatomical_image, 
                                                    vmax=vmax_anatomical_image,
                                                    origin='lower')
                            subplot2 = ax[i][j].imshow(image,
                                                    cmap=colormap_list[i],
                                                    alpha=alpha_CSI_amps[i-1],
                                                    extent=CSI_extent,
                                                    vmin=0,
                                                    vmax=0.8,# np.nanmax(image),
                                                    origin='lower')                            
                        
                        else:    
                            reference = img_interp(np.squeeze(np.abs(fit_amps[:, :, 0, r, 0, i-1])) * scaling_factor,
                                               interp_factor=interpolation_factor,
                                               interp_method='bilinear',
                                               threshold=0)                                                  
                            if use_percentile_thresholds:
                                threshold = np.nanpercentile(reference, vmin_percentiles[i-1])
                            else:
                                threshold = thresholds[i-1]
                        
                            image = img_interp(np.squeeze(np.abs(fit_amps[:, :, 0, r, 0, i-1])) * scaling_factor,
                                            interp_factor=interpolation_factor,
                                            interp_method='bilinear',
                                            threshold=threshold)
                            ax[i][j].set_title(r'HP ' + fit_params.attrs['metabs'][i-1])
                        
                            subplot1 = ax[i][j].imshow(anatomical_image,
                                                    cmap='bone',
                                                    extent=anatomical_image_extent, 
                                                    vmin=vmin_anatomical_image, 
                                                    vmax=vmax_anatomical_image,
                                                    origin='lower')
                            subplot2 = ax[i][j].imshow(image,
                                                    cmap=colormap_list[i],
                                                    alpha=alpha_CSI_amps[i-1],
                                                    vmin=np.nanmin(image),
                                                    vmax=np.nanmax(image),
                                                    extent=CSI_extent,
                                                    origin='lower')

                        if show_mask_contour:
                            mask_contour = ax[i][j].contour(
                                                    mask[:,:,mask_image_number], 
                                                    levels=[0.5], 
                                                    colors='magenta', 
                                                    linewidths=1, 
                                                    extent=mask_extent, 
                                                    origin='lower'
                                                    )
                        
                        # line1    = ax[i][j].axvline(x_axis_mm[x], c='magenta', linestyle='dashed', linewidth=1)
                        # line2    = ax[i][j].axhline(y_axis_mm[y], c='magenta', linestyle='dashed', linewidth=1)

                        subplot1.set_clip_path(circle)
                        subplot2.set_clip_path(circle)                        
                        # line1.set_clip_path(circle)
                        # line2.set_clip_path(circle)                        
                        
                        ax[i][j].axis('off')
                        
                        ax[i][j].set_xlim(CSI_extent[0], CSI_extent[1])
                        ax[i][j].set_ylim(CSI_extent[2], CSI_extent[3])
                        
                        if i==3:
                            vmin = 0 # np.nanpercentile(image, 0)
                            vmax = 0.8 # np.nanmax(image) # np.nanpercentile(image, 100)
                        else:
                            vmin = np.nanmin(image) # np.nanpercentile(image, 0)
                            vmax = np.nanmax(image) # np.nanpercentile(image, 100)
                        vmid = (vmax +  vmin) / 2
                    
                        divider = make_axes_locatable(ax[i][j])  
                        cb_ax   = divider.append_axes('bottom', size='5%', pad=0.05)  
                        cb_ax.set_visible(False)
                        
                        iax     = inset_axes(cb_ax, width='75%', height='100%', loc='center', borderpad=0)    
                        if i==3:
                            cbar    = fig.colorbar(subplot2, cax=iax, orientation='horizontal', ticks=[vmin, vmid, vmax], format='%.2f')
                            # cbar    = fig.colorbar(subplot2, cax=iax, orientation='horizontal', ticks=[vmin_CSI_amp_ratio, (vmin_CSI_amp_ratio + vmax_CSI_amp_ratio)/2, vmax_CSI_amp_ratio], format='%.2f')
                        else:
                            cbar    = fig.colorbar(subplot2, cax=iax, orientation='horizontal', ticks=[vmin, vmid, vmax], format='%.2f')
                        
                        image_start_time = '\n' + r'$t_{\mathrm{after~inj}}$ $\in$ = ' + '{0:.0f} s'.format(r * CSI_BrukerExp.frame_time_s[0] + t_start_CSI_after_inj_s)
                        if i==3:
                            pass
                            # cbar.set_label('[a. u.]')
                            # cbar.ax.ticklabel_format(style='sci', scilimits=(-3, 3))
                        else:
                            cbar.set_label(r'$^{13}$C FID amplitude' + '\n[a. u.]')
                            # cbar.ax.ticklabel_format(style='sci', scilimits=(-3, 3))
                        
                        # add_scalebar(
                            # px=1,
                            # ax=ax[i][j],
                            # units='mm',
                            # fixed_value=5,
                            # color='white',
                            # box_alpha=0.0,
                            # box_color='w',
                            # location='lower center',
                            # frameon=True,
                            # pad=0.075,
                            # length_fraction=None,
                            # border_pad=0.075,
                            # fontsize=8,
                        # )
                        
                        ################################################################################################################################################### 
                        sp_ax   = divider.append_axes('left', size='110%', pad=0.8) 
                        sp_ax.set_visible(False)
                        
                        spiax   = inset_axes(sp_ax, width='100%', height='100%', loc='center', borderpad=20)
                        spiax.set_visible(False)
                        
                    elif j==1:
                    
                        if i==0:
                            image = img_interp(np.squeeze(np.abs(np.mean(CSI_data[:, :, :, 0, r, :], axis=0))) * scaling_factor,                
                                               interp_factor=interpolation_factor,
                                               interp_method='bilinear',
                                               threshold=0)
                            ax[i][j].set_title(r'$^{13}$C density image')

                            subplot = ax[i][j].imshow(image,
                                                    cmap=colormap_list[i],
                                                    extent=CSI_extent,
                                                    origin='lower')
                        
                        elif i==3:                        
                            image = img_interp(np.squeeze(np.abs(fit_amps[:, :, 0, r, 0, 1]/fit_amps[:, :, 0, r, 0, 0])),
                                            interp_factor=interpolation_factor,
                                            interp_method='bilinear',
                                            threshold=0)
                            ax[i][j].set_title(r'HP [1-$^{13}$C] lac/pyr')

                            subplot = ax[i][j].imshow(image,
                                                      cmap=colormap_list[i],
                                                      extent=CSI_extent,
                                                      vmin=0,
                                                      vmax=2.0, # np.nanmax(image),                                                      
                                                      origin='lower')
                            
                        else:                        
                            image = img_interp(np.squeeze(np.abs(fit_amps[:, :, 0, r, 0, i-1])) * scaling_factor,
                                               interp_factor=interpolation_factor,
                                               interp_method='bilinear',
                                               threshold=0)                           
                            ax[i][j].set_title(r'HP ' + fit_params.attrs['metabs'][i-1])

                            subplot = ax[i][j].imshow(image,
                                                    cmap=colormap_list[i],
                                                    extent=CSI_extent,
                                                    origin='lower')
                                                    
                        if show_mask_contour:
                            mask_contour = ax[i][j].contour(
                                                    mask[:,:,mask_image_number], 
                                                    levels=[0.5], 
                                                    colors='magenta', 
                                                    linewidths=1, 
                                                    extent=mask_extent, 
                                                    origin='lower'
                                                    )
                                                                                                      
                        # line1    = ax[i][j].axvline(x_axis_mm[x], c='magenta', linestyle='dashed', linewidth=1)
                        # line2    = ax[i][j].axhline(y_axis_mm[y], c='magenta', linestyle='dashed', linewidth=1)

                        subplot.set_clip_path(circle)
                        # line1.set_clip_path(circle)
                        # line2.set_clip_path(circle)                        
                                                  
                        ax[i][j].axis('off')
                       
                        ax[i][j].set_xlim(CSI_extent[0], CSI_extent[1])
                        ax[i][j].set_ylim(CSI_extent[2], CSI_extent[3])

                        if i==3:
                            vmin = 0 # np.nanpercentile(image, 0)
                            vmax = 2.0 # np.nanpercentile(image, 100)
                        else:
                            vmin = np.nanpercentile(image, 0)
                            vmax = np.nanpercentile(image, 100)
                        vmid = (vmax +  vmin) / 2
                    
                        divider = make_axes_locatable(ax[i][j])  
                        cb_ax   = divider.append_axes('bottom', size='5%', pad=0.05)  
                        cb_ax.set_visible(False)
                        
                        iax     = inset_axes(cb_ax, width='75%', height='100%', loc='center', borderpad=0)    
                        
                        if i==3: 
                            # cbar    = fig.colorbar(subplot, cax=iax, orientation='horizontal', ticks=[vmin_CSI_amp_ratio, (vmin_CSI_amp_ratio + vmax_CSI_amp_ratio)/2, vmax_CSI_amp_ratio], format='%.2f')
                            cbar    = fig.colorbar(subplot, cax=iax, orientation='horizontal', ticks=[vmin, vmid, vmax], format='%.2f')
                            # cbar.ax.ticklabel_format(style='sci', scilimits=(-3, 3))
                        else: 
                            cbar    = fig.colorbar(subplot, cax=iax, orientation='horizontal', ticks=[vmin, vmid, vmax], format='%.2f')
                            # cbar.ax.ticklabel_format(style='sci', scilimits=(-3, 3))
                        
                        image_start_time = '\n' + r'$t_{\mathrm{after~inj}}$ $\in$ = ' + '{0:.0f} s'.format(r * CSI_BrukerExp.frame_time_s[0] + t_start_CSI_after_inj_s)
                        if i==0:
                            cbar.set_label(r'$\langle$ $^{13}$C spectral signal $\rangle$' + '\n[a. u.]')
                        elif i==3:
                            pass
                            # cbar.set_label('[a. u.]')
                        else:
                            cbar.set_label(r'$^{13}$C FID amplitude' + '\n[a. u.]')
            
                        ################################################################################################################################################### 
                        sp_ax   = divider.append_axes('right', size='110%', pad=0.8)                        
                        sp_ax.set_visible(False)
                        
                        spiax   = inset_axes(sp_ax, width='100%', height='100%', loc='center', borderpad=20)
                        
                        if i==0:
                            x1    = chemical_shift_axis_ppm
                            x2    = fit_chemical_shift_axis_ppm
                            xmin1 = xlim[0]
                            xmax1 = xlim[1]
                        
                            y11 = np.squeeze(np.abs(CSI_data[:, y, x, 0, r, :])) * scaling_factor
                            y12 = np.squeeze(np.abs(fit_spectra[:, y, x, 0, r, :])) * scaling_factor  
                            ymin1 = ylim[0]
                            ymax1 = ylim[1]
                                
                            spiax.set_ylabel(r'$^{13}$C signal magnitude' + '\n[a. u.]')
                            spiax.set_xlabel(r'$^{13}$C chemical shift'+ '\n[ppm]')
                            
                            spiax.invert_xaxis()
                        
                            spiax.plot(x1, y11, color=linecolor_measurement, lw=linewidth_measurement, linestyle='-', label=r'measurement', alpha=alpha_measurement)
                            spiax.plot(x2, y12, color=linecolor_fit, lw=linewidth_fit, linestyle='-', label=r'fit', alpha=alpha_fit)
                            
                            spiax.set_xlim(xmin1, xmax1)
                            if ylim == 'max':
                                ymax1 = max(y11.max(), y12.max())
                                spiax.set_ylim(- 0.1 * ymax1, ymax1 * 1.1)
                            else:
                                spiax.set_ylim(ymin1, ymax1)
        
                            spiax.tick_params(direction='in', right=True, top=True)
                                        
                            leg = spiax.legend(framealpha=1, title=r'$t_{\mathrm{after~injection}}$ = ' + '{0:.0f} s'.format(r * CSI_BrukerExp.frame_time_s[0] + t_start_CSI_after_inj_s), loc='upper right', bbox_to_anchor=(0.95, 1.8))        
                            leg.get_frame().set_edgecolor('black')
                            
                        elif i==3:
                            x3    = np.arange(CSI_data.shape[-2]) * CSI_BrukerExp.frame_time_s[0] + t_start_CSI_after_inj_s
                            xmin3 = 0
                            xmax3 = 200
                
                            y3    = np.abs(fit_amps[y, x, 0, :, 0, 1]/fit_amps[y, x, 0, :, 0, 0])
                            
                            if show_mask:
                                y4    = np.abs((np.repeat(mask_interpolation[:, :, :, np.newaxis], fit_amps.shape[3], 3) / mask_interpolation.sum() * fit_amps[:, :, :, :, 0, 1].to_numpy() / fit_amps[:, :, :, :, 0, 0].to_numpy()).sum((0,1,2)))
                                                        
                            ymin3 = 0
                            ymax3 = 1.5 # 1.1 * np.max(y3)
                        
                            spiax.set_ylabel(r'HP [1-$^{13}$C] lac/pyr' + '\n[a. u.]')
                            spiax.set_xlabel(r'$t_{\mathrm{after~injection}}$ [s]')
                                    
                            spiax.plot(x3, y3, color='magenta', lw=1.0, linestyle='dashed')
                            
                            if show_mask:
                                spiax.plot(x3, y4, color='magenta', lw=1.0, linestyle='-')
                
                            spiax.axvline(x3[r], color='k', linestyle='dashed', linewidth=1)
                            # spiax.axvspan(x3[r], x3[r], alpha=0.2, color='gray', label='time span used\n for averaging')
                    
                            # spiax.set_xlim(xmin3, xmax3)
                            spiax.set_ylim(ymin3, ymax3)
                            
                            spiax.set_xticks([25, 50, 75, 100])
                            spiax.set_yticks([0.0, 0.5, 1.0, 1.5])
                            spiax.tick_params(direction='in', right=True, top=True)
                        
                            # leg = spiax.legend(framealpha=1, loc='upper right', bbox_to_anchor=(0.8, 1.4))         
                            # leg.get_frame().set_edgecolor('black')   
                        
                        else:
                            x3    = np.arange(CSI_data.shape[-2]) * CSI_BrukerExp.frame_time_s[0] + t_start_CSI_after_inj_s
                            xmin3 = 0
                            xmax3 = 200
                
                            y3    = np.abs(fit_amps[y, x, 0, :, 0, i-1]) * scaling_factor
                            ymin3 = 0
                            ymax3 = 1.1 * np.max(y3)
                        
                            spiax.set_ylabel(r'$^{13}$C FID amplitude' + '\n[a. u.]')
                            spiax.set_xlabel(r'$t_{\mathrm{after~injection}}$ [s]')
                                    
                            spiax.plot(x3, y3, color=metabolite_colors[i-1], lw=1.0, linestyle='-')
                
                            spiax.axvline(x3[r], color='k', linestyle='dashed', linewidth=1)
                            # spiax.axvspan(x3[r], x3[r], alpha=0.2, color='gray', label='time span used\n for averaging')
                    
                            # spiax.set_xlim(xmin3, xmax3)
                            spiax.set_ylim(ymin3, ymax3)
                            
                            spiax.set_xticks([25, 50, 75, 100])
                            # spiax.set_yticks([0.0, 0.5, 1.0, 1.5])
                            spiax.tick_params(direction='in', right=True, top=True)
                        
                            # leg = spiax.legend(framealpha=1, loc='upper right', bbox_to_anchor=(0.8, 1.4))         
                            # leg.get_frame().set_edgecolor('black')                             
                        
                        ###################################################################################################################################################
                        
                    elif j==2:
                        
                        ax[i][j].set_visible(False)
            
                    # plt.tight_layout()

                    if save_fig:
            
                        save_name = plot_params.get('save_name', '{0}_fit_results_rep_{1}_image_{2}'.format(sample_ID, r, anatomical_image_number))
            
                        plt.savefig(
                            os.path.join(save_path, save_name + save_format),
                            transparent=fig_transparent,
                            dpi=save_dpi,
                        ) 

        plt.close()