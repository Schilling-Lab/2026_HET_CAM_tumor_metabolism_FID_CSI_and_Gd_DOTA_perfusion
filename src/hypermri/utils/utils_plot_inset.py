###########################################################################################################################################################

# Inset in high-resolution images

###########################################################################################################################################################


# =========================================================================================================================================================
# Import necessary packages
# =========================================================================================================================================================

import os, sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from hypermri import Study

from hypermri.utils.plot.utils_plot import add_scalebar
from hypermri.utils.plot.utils_plot import add_patch
from hypermri.utils.plot.utils_plot import extent_to_slice


# =========================================================================================================================================================
# Definition of necessary functions
# =========================================================================================================================================================

def plot_inset(
    CSI_BrukerExp,
    fit_spectra=None,
    fit_chemical_shift_axis_ppm=None,
    fit_lac_amp=None,
    fit_pyr_amp=None,    
    rep=None,
    CSI_data=None,
    anatomical_BrukerExp=None,
    image_stack=None,
    anatomical_image_number=None,
    mask=None,
    mask_image_number=None,
    mask_interpolation=None,
    plot_params={},
):
    '''
    Plot an overlay of the measured and the fit spectra
    Parameters
    ----------
    CSI_BrukerExp: Bruker experiment object
        Contains CSI data
    fit_spectra: nd numpy array
        Array of fitted spectra
    fit_chemical_shift_axis_ppm: 1d numpy array
        Array of chemical shifts sampled in fitted_spectra
    fit_lac_amp: nd numpy array
        Array of fitted lactate amplitudes
    fit_pyr_amp: nd numpy array
        Array of fitted pyruvate amplitudes
    rep: int/range
        Repetition to be plotted
    CSI_data: nd numpy array
        The CSI data that should be plotted in overlay with fit_spectra
    anatomical_BrukerExp: Bruker experiment object
        Contains anatomical reference images
    anatomical_image_number: int
        Number of slices chosen for overlay (indexing starts at 0)
    plot_params: dict
        Plot parameters

    Returns
    -------
    A figure with inset

    Examples
    -------
    plot_params = {}
    
    plot_params['figsize']                            = (7,3)                            
    
    plot_params['alpha_single_freq_map']              = 0.5
    plot_params['alpha_R_lac_pyr_map']                = 0.5
    
    plot_params['show_tqdm']                          = False 
    
    plot_params['vmin_anatomical_image']              = RARE_vmin
    plot_params['vmax_anatomical_image']              = RARE_vmax
    plot_params['vmin_R_lac_pyr_map']                 = 0
    plot_params['vmax_R_lac_pyr_map']                 = 0.2 
    
    plot_params['CSI_extent']                         = [CSI_RARE_rps_extent_mm_rearranged[2], CSI_RARE_rps_extent_mm_rearranged[3], CSI_RARE_rps_extent_mm_rearranged[0], CSI_RARE_rps_extent_mm_rearranged[1]]
    plot_params['anatomical_image_extent']            = [RARE_rps_extent_mm[2], RARE_rps_extent_mm[3], RARE_rps_extent_mm[0], RARE_rps_extent_mm[1]]
    
    plot_params['show_ticks']                         = False
    plot_params['show_inset']                         = True
    plot_params['show_single_freq_map']               = False
    plot_params['show_R_lac_pyr_map']                 = False
    plot_params['single_freq_index']                  = 27
    plot_params['single_freq_map_treshold']           = 1e6
    
    plot_params['save_fig']                           = True
    plot_params['save_dpi']                           = 200
    plot_params['fig_transparent']                    = False
    
    RARE_inset_path                                   = os.path.join(specific_processing_path, '06_CSI_anatomical_images_with_inset')
    plot_params['save_path']                          = RARE_inset_path
    os.makedirs(RARE_inset_path, exist_ok=True)
    
    plot_params['save_name']                          = '{0}_RARE_inset_s_{1}_fit'.format(egg_MRI_ID, RARE_selected_slice_idx)
    
    plot_inset(
        CSI_BrukerExp=CSI,
        fit_spectra=None,
        fit_chemical_shift_axis_ppm=None,
        fit_lac_amp=None,
        fit_pyr_amp=None,
        rep=0,
        CSI_data=np.abs(fftnd(CSI_RARE_reco_data_rearranged, axes=[0])),
        anatomical_BrukerExp=RARE,
        image_stack=None,
        anatomical_image_number=RARE_selected_slice_idx,
        plot_params=plot_params,
    )
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
    figsize                             = plot_params.get('figsize', (15, 5))
    plot_title                          = plot_params.get('plot_title', None)

    alpha_single_freq_map               = plot_params.get('alpha_single_freq_map', 0.5)
    alpha_R_lac_pyr_map                 = plot_params.get('alpha_R_lac_pyr_map', 0.5)

    show_tqdm                           = plot_params.get('show_tqdm', True)

    vmin_anatomical_image               = plot_params.get('vmin_anatomical_image', None)
    vmax_anatomical_image               = plot_params.get('vmax_anatomical_image', None)
    vmin_anatomical_percentile          = plot_params.get('vmin_anatomical_percentile', 0)
    vmax_anatomical_percentile          = plot_params.get('vmax_anatomical_percentile', 100)
    
    vmin_single_frequency_map           = plot_params.get('vmin_single_frequency_map', None)
    vmax_single_frequency_map           = plot_params.get('vmax_single_frequency_map', None)
    vmin_R_lac_pyr_map                  = plot_params.get('vmin_R_lac_pyr_map', None)
    vmax_R_lac_pyr_map                  = plot_params.get('vmax_R_lac_pyr_map', None)
    
    CSI_extent                          = plot_params.get('CSI_extent', [CSI_BrukerExp.rps_extent_mm[2], CSI_BrukerExp.rps_extent_mm[3], CSI_BrukerExp.rps_extent_mm[0], CSI_BrukerExp.rps_extent_mm[1]])
    anatomical_image_extent             = plot_params.get('anatomical_image_extent', CSI_extent)
    mask_extent                         = plot_params.get('mask_extent', CSI_extent)
    mask_interpolation_extent           = plot_params.get('mask_interpolation_extent', CSI_extent)

    show_ticks                          = plot_params.get('show_ticks', False) 
    show_inset                          = plot_params.get('show_inset', True) 
    show_single_freq_map                = plot_params.get('show_single_freq_map', False) 
    show_R_lac_pyr_map                  = plot_params.get('show_R_lac_pyr_map', False) 
    show_mask                           = plot_params.get('show_mask', False) 
    show_mask_map                       = plot_params.get('show_mask_map', False) 
    show_mask_contour                   = plot_params.get('show_mask_contour', False) 
    show_mask_interpolation             = plot_params.get('show_mask_interpolation', False) 
    single_freq_index                   = plot_params.get('single_freq_index', 27) 
    single_freq_map_treshold            = plot_params.get('single_freq_map_treshold', 1e6) 
    
    save_fig                            = plot_params.get('save_fig', False)
    save_dpi                            = plot_params.get('save_dpi', None)
    fig_transparent                     = plot_params.get('fig_transparent', True)
    save_path                           = plot_params.get('save_path', CSI_BrukerExp.path)
    save_format                         = plot_params.get('save_format', '.png')
    
    y_dim = fit_spectra.shape[2]
    x_dim = fit_spectra.shape[3]

    if np.any(image_stack):
        image_stack = image_stack
    else:
        image_stack = np.squeeze(np.abs(anatomical_BrukerExp.data_array))

    # if anatomical_image_extent[1] > anatomical_image_extent[0] and anatomical_image_extent[3] > anatomical_image_extent[2]: 
        # CSI_slice = extent_to_slice(image_stack[:, :, anatomical_image_number].shape, anatomical_image_extent, CSI_extent, origin='lower')
        # vmin_anatomical_image = np.nanpercentile(image_stack[:, :, anatomical_image_number][CSI_slice], vmin_anatomical_percentile)
        # vmax_anatomical_image = np.nanpercentile(image_stack[:, :, anatomical_image_number][CSI_slice], vmax_anatomical_percentile)
    # else:
        # vmin_anatomical_image = np.nanpercentile(image_stack[:, :, anatomical_image_number], vmin_anatomical_percentile)
        # vmax_anatomical_image = np.nanpercentile(image_stack[:, :, anatomical_image_number], vmax_anatomical_percentile)
        
    for r in rep:
        fig, ax = plt.subplots(figsize=figsize)
        ax.clear()

        anatomical_image = ax.imshow(
                                     image_stack[:, :, anatomical_image_number],
                                     cmap='bone',
                                     vmin=vmin_anatomical_image, vmax=vmax_anatomical_image,
                                     extent=anatomical_image_extent,
                                     origin='lower',
                                    )
        
        if show_mask:
        
            if show_mask_map:                                 
                mask_map = ax.imshow(
                                    np.where(mask[:,:,mask_image_number]>0.001, mask[:,:,mask_image_number], np.nan), 
                                    cmap='jet_r', 
                                    alpha=0.5, 
                                    extent=mask_extent, 
                                    origin='lower'
                                    )

            if show_mask_contour:                      
                mask_contour = ax.contour(
                                        mask[:,:,mask_image_number], 
                                        levels=[0.5], 
                                        colors='magenta', 
                                        linewidths=1, 
                                        extent=mask_extent, 
                                        origin='lower'
                                        )

        if show_single_freq_map:
            single_frequency_map = ax.imshow(
                                             np.where(np.squeeze(CSI_data[single_freq_index, :, :, 0, r]) > single_freq_map_treshold, np.squeeze(CSI_data[single_freq_index, :, :, 0, r]), np.nan),
                                             cmap='jet',
                                             vmin=vmin_single_frequency_map, vmax=vmax_single_frequency_map,
                                             extent=CSI_extent,
                                             alpha=alpha_single_freq_map,
                                             origin='lower',
                                            )

            vmid_single_frequency_map = (vmax_single_frequency_map + vmin_single_frequency_map) / 2

            try:
                cbar.remove()
            except:
                pass
            
            divider = make_axes_locatable(ax)  
            cb_ax   = divider.append_axes('bottom', size='5%', pad=0.05)  
            cb_ax.clear()
            cb_ax.set_visible(False)
                
            iax     = inset_axes(cb_ax, width='75%', height='50%', loc='center', borderpad=0) 
            iax.clear()
            cbar    = fig.colorbar(single_frequency_map, cax=iax, orientation='horizontal', ticks=[vmin_single_frequency_map, vmid_single_frequency_map, vmax_single_frequency_map], 
                                   label=r'$^{13}$C signal [a. u.]' + '\nof chemical shift ' + '{:.1f} ppm'.format(chemical_shift_axis_ppm[single_freq_index]))
            cbar.ax.ticklabel_format(style='sci', scilimits=(-3, 3))
            cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))

        if show_R_lac_pyr_map:
            R_lac_pyr_map = ax.imshow(
                                      np.where(np.squeeze(CSI_data[single_freq_index, :, :, 0, r]) > single_freq_map_treshold, np.abs(np.squeeze(fit_lac_amp[:, :, 0, r]) / np.squeeze(fit_pyr_amp[:, :, 0, r])), np.nan),
                                      cmap='jet',
                                      vmin=vmin_R_lac_pyr_map, vmax=vmax_R_lac_pyr_map,
                                      extent=CSI_extent,
                                      alpha=alpha_R_lac_pyr_map,
                                      origin='lower',
                                     )

            vmid_R_lac_pyr_map = (vmax_R_lac_pyr_map + vmin_R_lac_pyr_map) / 2

            try:
                cbar.remove()
            except:
                pass
            
            divider = make_axes_locatable(ax)  
            cb_ax   = divider.append_axes('bottom', size='5%', pad=0.05)  
            cb_ax.clear()
            cb_ax.set_visible(False)
                
            iax     = inset_axes(cb_ax, width='75%', height='50%', loc='center', borderpad=0) 
            iax.clear()
            cbar    = fig.colorbar(R_lac_pyr_map, cax=iax, orientation='horizontal', ticks=[vmin_R_lac_pyr_map, vmid_R_lac_pyr_map, vmax_R_lac_pyr_map], 
                                   label=r'[1-$^{13}$C] lactate/pyruvate signal ratio')
            cbar.ax.ticklabel_format(style='sci', scilimits=(-3, 3))
            cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))        
        
        if not show_ticks:
            ax.axis('off')
            
        ax.set_title(plot_title, color='white')
        
        add_scalebar(
            px=1e-1, 
            ax=ax,
            units='cm',
            fixed_value=1,
            color='w',
            box_alpha=0.0,
            box_color='w',
            location='lower right',
            frameon=True,
            pad=0.075,
            length_fraction=None,
            border_pad=0.075,
            fontsize=8,
        )
        
        ax.set_ylim(anatomical_image_extent[2], anatomical_image_extent[3])
        ax.set_xlim(anatomical_image_extent[0], anatomical_image_extent[1])

        width  = CSI_extent[1] - CSI_extent[0] 
        height = CSI_extent[3] - CSI_extent[2]
        
        width_center  = (CSI_extent[1] + CSI_extent[0]) / 2 
        height_center = (CSI_extent[3] + CSI_extent[2]) / 2
                
        patchboxlist    = [(height, width)]
        patchcenterlist = [(height_center, width_center)]

        if show_inset:
            insetslicelist = []
            colorlist = ['r']
    
            for i, patchcenter in enumerate(patchcenterlist):
                insetslicelist.append(add_patch(
                image=image_stack[:, :, anatomical_image_number],
                patchcenter=patchcenter,
                patchbox=patchboxlist[i],
                ax=None,
                sl=None,
                color=colorlist[i],
                lw=1,
                fc='none',
                out=True,
                ))
        
                # axin = ax.inset_axes([1.05, 0.0, (1.0*image_stack.shape[0]*patchboxlist[0][1]/patchboxlist[0][0])/image_stack.shape[1], 1.0])
                axin = ax.inset_axes([1.01, 0.0, 1.4, 1.0])
                # axin = ax.inset_axes([-1.4, 0.0, 1.4, 1.0])
        
                inset = axin.imshow(image_stack[:, :, anatomical_image_number],
                                    vmin=vmin_anatomical_image, vmax=vmax_anatomical_image,
                                    cmap='bone',
                                    extent=anatomical_image_extent,
                                    origin='lower')
        
                axin.set_xlim([insetslicelist[0][1].start, insetslicelist[0][1].stop])
                axin.set_ylim([insetslicelist[0][0].start, insetslicelist[0][0].stop])
                        
                axin.set_xticks([])
                axin.set_yticks([])
        
                axin.spines['bottom'].set_color(colorlist[0])
                axin.spines['top'].set_color(colorlist[0])
                axin.spines['left'].set_color(colorlist[0])
                axin.spines['right'].set_color(colorlist[0])
        
                add_scalebar(
                px=1,
                ax=axin,
                units='mm',
                fixed_value=5,
                color='w',
                box_alpha=0.0,
                box_color='w',
                location='lower right',
                frameon=True,
                pad=0.075,
                length_fraction=None,
                border_pad=0.075,
                fontsize=8
                )
                
                if show_mask:

                    if show_mask_map:                                 
                        maskin_map = axin.imshow(
                                                np.where(mask[:,:,mask_image_number]>0.001, mask[:,:,mask_image_number], np.nan), 
                                                cmap='jet_r', 
                                                alpha=0.5, 
                                                extent=mask_extent, 
                                                origin='lower'
                                                )

                    if show_mask_contour:                            
                        maskin_contour = axin.contour(
                                                    mask[:,:,mask_image_number], 
                                                    levels=[0.5], 
                                                    colors='magenta', 
                                                    linewidths=1, 
                                                    extent=mask_extent, 
                                                    origin='lower'
                                                    )
                                                    
                    if show_mask_interpolation:
                        maskin_map = axin.imshow(
                                                mask_interpolation, 
                                                cmap='jet', 
                                                alpha=0.8, 
                                                extent=mask_interpolation_extent, 
                                                origin='lower'
                                                )
                    
                
        # plt.tight_layout()
            
        # divider = make_axes_locatable(ax)  
        # cb_ax   = divider.append_axes('bottom', size='5%', pad=0.05)  
        # cb_ax.clear()
        # cb_ax.set_visible(False)
            
        # iax     = inset_axes(cb_ax, width='50%', height='100%', loc='center', borderpad=0) 
        # iax.clear()
        # cbar    = fig.colorbar(anatomical_image, cax=iax, orientation='horizontal', ticks=[vmin_anatomical_image, vmax_anatomical_image], 
                               # label=r'$^{1}$H signal [a. u.]')
        # cbar.ax.ticklabel_format(style='sci', scilimits=(-3, 3))
        # cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))         
        # for spine in cbar.ax.spines.values():
            # spine.set_edgecolor('white')
     
        # cbar.set_ticklabels(['0', '1'])
     
        if save_fig:

            save_name = plot_params.get('save_name',  '{0}_inset'.format(sample_ID))

            plt.savefig(
                os.path.join(save_path, save_name + save_format),
                transparent=fig_transparent,
                dpi=save_dpi,
                facecolor="black"
            )
        
        plt.close()
            