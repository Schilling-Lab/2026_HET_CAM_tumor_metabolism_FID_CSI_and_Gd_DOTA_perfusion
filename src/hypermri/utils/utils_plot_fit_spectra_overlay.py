###########################################################################################################################################################

# Overlay of CSI spectra over high-resolution images

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

from hypermri import Study

from hypermri.utils.plot.utils_plot import add_scalebar
from hypermri.utils.plot.utils_plot import extent_to_slice


# =========================================================================================================================================================
# Definition of necessary functions
# =========================================================================================================================================================

def plot_fit_spectra_overlay(
    CSI_BrukerExp,
    fit_spectra=None,
    fit_chemical_shift_axis_ppm=None,
    fit_lac_amp=None,
    fit_pyr_amp=None,    
    rep=None,
    CSI_data=None,
    marked_voxel=None,
    anatomical_BrukerExp=None,
    image_stack=None,
    anatomical_image_number=None,
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
    marked_voxel: [i,j] tuple
        Indices of marked spectrum in overlay
    anatomical_BrukerExp: Bruker experiment object
        Contains anatomical reference images
    anatomical_image_number: int
        Number of slices chosen for overlay (indexing starts at 0)
    plot_params: dict
        Plot parameters

    Returns
    -------
    A multisubplot figure

    Examples
    -------
    plot_params = {}
    
    plot_params['figsize']                            = (7,4)                            
                                    
    plot_params['ylim']                               = 'max' 
    plot_params['xlim']                               = (190, 165) 
    
    plot_params['linewidth_measurement']              = 0.3
    plot_params['linewidth_fit']                      = 0.3
    plot_params['marked_voxel_linewidth_measurement'] = 1    
    plot_params['marked_voxel_linewidth_fit']         = 1                                
    
    plot_params['linecolor_measurement']              = 'yellow'
    plot_params['linecolor_fit']                      = 'red'
    plot_params['linecolor_marked_voxel_measurement'] = 'black'
    plot_params['linecolor_marked_voxel_fit']         = 'red'
    
    plot_params['alpha_measurement']                  = 1
    plot_params['alpha_fit']                          = 1
    plot_params['alpha_marked_voxel_measurement']     = 1
    plot_params['alpha_marked_voxel_fit']             = 1
    
    plot_params['alpha_single_freq_map']              = 0.5
    plot_params['alpha_R_lac_pyr_map']                = 0.5
    
    plot_params['marked_voxel']                       = (12, 12)                            
    marked_voxel                                      = plot_params['marked_voxel']        
    plot_params['title_marked_voxel']                 = f'tumor voxel {marked_voxel}\n spectrum'
    plot_params['ylim_marked_voxel']                  = (0, 1e6)                          
    
    plot_params['show_tqdm']                          = False 
    
    # plot_params['vmin_anatomical_image']            = 0
    # plot_params['vmax_anatomical_image']            = 6e5
    plot_params['vmin_R_lac_pyr_map']                 = 0
    plot_params['vmax_R_lac_pyr_map']                 = 0.2 
    
    plot_params['CSI_extent']                         = [CSI_rps_endpoint_grid_mm_transformed_rearranged[1][0,0,0], CSI_rps_endpoint_grid_mm_transformed_rearranged[1][0,1,0], CSI_rps_endpoint_grid_mm_transformed_rearranged[0][0,0,0], CSI_rps_endpoint_grid_mm_transformed_rearranged[0][1,0,0]]
    plot_params['anatomical_image_extent']            = [RARE_extent[2], RARE_extent[3], RARE_extent[0], RARE_extent[1]]
    
    plot_params['show_ticks']                         = False
    plot_params['show_single_freq_map']               = False
    plot_params['show_R_lac_pyr_map']                 = True
    plot_params['single_freq_index']                  = 27
    plot_params['single_freq_map_treshold']           = 1e6
    
    plot_params['save_fig']                           = True
    plot_params['save_dpi']                           = 500
    plot_params['fig_transparent']                    = False
    plot_params['save_path']                          = ''
    plot_params['save_name']                          = '{0}_spectra_overlayed_slice_index_{1}_fit'.format('Egg_106_EDD16', RARE_slice_selected_idx)
    
    plot_fit_spectra_overlay(
        CSI_BrukerExp=CSI,
        fit_spectra=fit_spectra.transpose(6, transpose_array[0], transpose_array[1], transpose_array[2], 3, 4, 5)[::flip_array[0],::flip_array[1],::flip_array[2]].sum(-1),
        fit_chemical_shift_axis_ppm=fit_chemical_shift_axis_ppm,
        fit_lac_amp=fit_amps[...,1],
        fit_pyr_amp=fit_amps[...,0],
        rep=0,
        CSI_data=np.abs(fftnd(CSI_reco_data_rearranged, axes=[0])),
        marked_voxel=None,
        anatomical_BrukerExp=RARE,
        image_stack=None,
        anatomical_image_number=RARE_slice_selected_idx,
        plot_params=plot_params,
    )
    
    plt.show()
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

    ylim                                = plot_params.get('ylim', (None, None))
    if ylim == 'max':
        ylim = (0, np.max(CSI_data))
    xlim                                = plot_params.get('xlim', (None, None))

    linewidth_measurement               = plot_params.get('linewidth_measurement', 0.5)
    linewidth_fit                       = plot_params.get('linewidth_fit', 0.25)
    marked_voxel_linewidth_measurement  = plot_params.get('marked_voxel_linewidth_measurement', 1)
    marked_voxel_linewidth_fit          = plot_params.get('marked_voxel_linewidth_fit', 1)

    linecolor_measurement               = plot_params.get('linecolor_measurement', 'y')
    linecolor_fit                       = plot_params.get('linecolor_fit', 'r')
    linecolor_marked_voxel_measurement  = plot_params.get('linecolor_marked_voxel_measurement', 'y')
    linecolor_marked_voxel_fit          = plot_params.get('linecolor_marked_voxel_fit', 'r')

    alpha_measurement                   = plot_params.get('alpha_measurement', 1)
    alpha_fit                           = plot_params.get('alpha_fit', 1)
    alpha_marked_voxel_measurement      = plot_params.get('alpha_marked_voxel_measurement', 1)
    alpha_marked_voxel_fit              = plot_params.get('alpha_marked_voxel_fit', 1)

    alpha_single_freq_map               = plot_params.get('alpha_single_freq_map', 0.5)
    alpha_R_lac_pyr_map                 = plot_params.get('alpha_R_lac_pyr_map', 0.5)
    
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

    mvt                                 = f'Spectrum from voxel {marked_voxel}'
    title_marked_voxel                  = plot_params.get('title_marked_voxel', mvt)
    ylim_marked_voxel                   = plot_params.get('ylim_marked_voxel', ylim)

    show_tqdm                           = plot_params.get('show_tqdm', True)

    # vmin_anatomical_image               = plot_params.get('vmin_anatomical_image', None)
    # vmax_anatomical_image               = plot_params.get('vmax_anatomical_image', None)
    vmin_anatomical_percentile          = plot_params.get('vmin_anatomical_percentile', 0)
    vmax_anatomical_percentile          = plot_params.get('vmax_anatomical_percentile', 100)
            
    vmin_single_frequency_map           = plot_params.get('vmin_single_frequency_map', None)
    vmax_single_frequency_map           = plot_params.get('vmax_single_frequency_map', None)
    vmin_R_lac_pyr_map                  = plot_params.get('vmin_R_lac_pyr_map', None)
    vmax_R_lac_pyr_map                  = plot_params.get('vmax_R_lac_pyr_map', None)
    vmin_percentile                     = plot_params.get('vmin_percentile', 0)
    vmax_percentile                     = plot_params.get('vmax_percentile', 100)
    
    CSI_extent                          = plot_params.get('CSI_extent', [CSI_BrukerExp.rps_extent_mm[2], CSI_BrukerExp.rps_extent_mm[3], CSI_BrukerExp.rps_extent_mm[0], CSI_BrukerExp.rps_extent_mm[1]])
    anatomical_image_extent             = plot_params.get('anatomical_image_extent', CSI_extent)

    show_ticks                          = plot_params.get('show_ticks', False) 
    show_single_freq_map                = plot_params.get('show_single_freq_map', False) 
    show_R_lac_pyr_map                  = plot_params.get('show_R_lac_pyr_map', False) 
    single_freq_index                   = plot_params.get('single_freq_index', 27) 
    single_freq_map_treshold            = plot_params.get('single_freq_map_treshold', 1e6) 
    fit_pyr_amp_treshold                = plot_params.get('fit_pyr_amp_treshold', 0) 
    use_percentile_threshold            = plot_params.get('use_percentile_threshold', False)

    save_fig                            = plot_params.get('save_fig', False)
    save_dpi                            = plot_params.get('save_dpi', None)
    fig_transparent                     = plot_params.get('fig_transparent', True)
    save_path                           = plot_params.get('save_path', CSI_BrukerExp.path)
    
    y_dim = fit_spectra.shape[2]
    x_dim = fit_spectra.shape[3]

    if np.any(image_stack):
        image_stack = image_stack
    else:
        image_stack = np.squeeze(np.abs(anatomical_BrukerExp.data_array))
        
    if anatomical_image_extent[1] > anatomical_image_extent[0] and anatomical_image_extent[3] > anatomical_image_extent[2]: 
        CSI_slice = extent_to_slice(image_stack[:, :, anatomical_image_number].shape, anatomical_image_extent, CSI_extent, origin='lower')
        vmin_anatomical_image = np.nanpercentile(image_stack[:, :, anatomical_image_number][CSI_slice], vmin_anatomical_percentile)
        vmax_anatomical_image = np.nanpercentile(image_stack[:, :, anatomical_image_number][CSI_slice], vmax_anatomical_percentile)
    else:
        vmin_anatomical_image = np.nanpercentile(image_stack[:, :, anatomical_image_number], vmin_anatomical_percentile)
        vmax_anatomical_image = np.nanpercentile(image_stack[:, :, anatomical_image_number], vmax_anatomical_percentile)

    for r in rep:
        fig, ax = plt.subplots(figsize=figsize)
        ax.clear()

        ax.imshow(
            image_stack[:, :, anatomical_image_number],
            cmap='bone',
            vmin=vmin_anatomical_image, vmax=vmax_anatomical_image,
            extent=anatomical_image_extent,
            origin='lower',
        )

        if show_single_freq_map:
        
            if use_percentile_threshold:
                threshold = np.nanpercentile(np.squeeze(CSI_data[single_freq_index, :, :, 0, r]), vmin_percentile)
            else:
                threshold = single_freq_map_treshold
                
            single_frequency_map = ax.imshow(
                                             np.where(np.squeeze(CSI_data[single_freq_index, :, :, 0, r]) > threshold, np.squeeze(CSI_data[single_freq_index, :, :, 0, r]), np.nan),
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
        
            if use_percentile_threshold:
                threshold = np.nanpercentile(np.squeeze(np.abs(fit_pyr_amp[:, :, 0, r])), vmin_percentile)
            else:
                threshold = fit_pyr_amp_treshold
                
            R_lac_pyr_map = ax.imshow(
                                      np.where(np.squeeze(np.abs(fit_pyr_amp[:, :, 0, r])) > threshold, np.abs(np.squeeze(fit_lac_amp[:, :, 0, r]) / np.squeeze(fit_pyr_amp[:, :, 0, r])), np.nan),
                                      cmap='jet',
                                      vmin=vmin_R_lac_pyr_map, 
                                      vmax=vmax_R_lac_pyr_map,
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
            
        ax.set_title(plot_title)
        
        add_scalebar(
            px=1, 
            ax=ax,
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
            fontsize=8,
        )
        
        ax.set_ylim(CSI_extent[2], CSI_extent[3])
        ax.set_xlim(CSI_extent[0], CSI_extent[1])

        colorlist = ['dodgerblue', 'magenta']            

        height = (CSI_extent[3] - CSI_extent[2]) / CSI_data.shape[1]
        irange = np.arange(
            CSI_extent[2],
            CSI_extent[3],
            height,
        )
        width = (CSI_extent[1] - CSI_extent[0]) / CSI_data.shape[2]
        jrange = np.arange(
            CSI_extent[0],
            CSI_extent[1],
            width,
        )

        for i in tqdm(range(CSI_data.shape[1]), disable=not show_tqdm):
            for j in tqdm(range(CSI_data.shape[2]), disable=not show_tqdm):
                
                bottom_anchor = irange[i]
                left_anchor   = jrange[j]

                axin = ax.inset_axes(
                    [left_anchor, bottom_anchor, width, height],
                    transform=ax.transData,
                ) 
                axin.clear()

                axin.plot(
                    chemical_shift_axis_ppm,
                    np.squeeze(CSI_data[:, i, j, 0, r]),
                    color=linecolor_measurement,
                    linewidth=linewidth_measurement,
                    alpha=alpha_measurement,
                )
                if fit_spectra_passed:
                    axin.plot(
                        fit_chemical_shift_axis_ppm,
                        np.squeeze(fit_spectra[:, i, j, 0, r]),
                        color=linecolor_fit,
                        linewidth=linewidth_fit,
                        alpha=alpha_fit,
                    )

                axin.invert_xaxis()

                axin.set_ylim(ylim)
                axin.set_xlim(xlim)

                axin.set_yticks([])
                axin.set_xticks([])

                axin.patch.set_alpha(0)

                if (i, j) == marked_voxel:
                    axin.spines['bottom'].set_color(colorlist[1])
                    axin.spines['top'].set_color(colorlist[1])
                    axin.spines['left'].set_color(colorlist[1])
                    axin.spines['right'].set_color(colorlist[1])

                    axin.spines['bottom'].set_linewidth(1.5)
                    axin.spines['top'].set_linewidth(1.5)
                    axin.spines['left'].set_linewidth(1.5)
                    axin.spines['right'].set_linewidth(1.5)

                else:
                    axin.spines['bottom'].set_color(colorlist[0])
                    axin.spines['top'].set_color(colorlist[0])
                    axin.spines['left'].set_color(colorlist[0])
                    axin.spines['right'].set_color(colorlist[0])

                    axin.spines['bottom'].set_linewidth(0.5)
                    axin.spines['top'].set_linewidth(0.5)
                    axin.spines['left'].set_linewidth(0.5)
                    axin.spines['right'].set_linewidth(0.5)

        axin = ax.inset_axes([1.2, 0.4, 0.4, 0.4])

        axin.plot(
            chemical_shift_axis_ppm,
            np.squeeze(CSI_data[:, marked_voxel[0], marked_voxel[1], 0, r]),
            color=linecolor_marked_voxel_measurement,
            linewidth=marked_voxel_linewidth_measurement,
            label=title_marked_voxel,
            alpha=alpha_marked_voxel_measurement,
        )
        if fit_spectra_passed:
            axin.plot(
                fit_chemical_shift_axis_ppm,
                np.squeeze(fit_spectra[:, marked_voxel[0], marked_voxel[1], 0, r]),
                color=linecolor_marked_voxel_fit,
                linewidth=marked_voxel_linewidth_fit,
                alpha=alpha_marked_voxel_fit,
                label='fit',
            )
            axin.legend()

        axin.invert_xaxis()

        if ylim_marked_voxel == 'auto':
            ymax_measured = np.max(
                np.squeeze(CSI_data[:, marked_voxel[0], marked_voxel[1], 0, r]),
            )
            if fit_spectra_passed:
                ymax_fit = np.max(
                    np.squeeze(fit_spectra[:, marked_voxel[0], marked_voxel[1], 0, r]),
                )
            else:
                ymax_fit = 0
            ymax = 1.1 * max(ymax_measured, ymax_fit)
            ymin = -0.1 * max(ymax_measured, ymax_fit)
            axin.set_ylim((ymin, ymax))
        else:
            axin.set_ylim(ylim_marked_voxel)
        axin.set_xlim(xlim)

        axin.set_ylabel(r'$^{13}$C signal magnitude' + '\n[a. u.]')
        axin.set_xlabel(r'$^{13}$C chemical shift' + '\n[ppm]')

        axin.spines['bottom'].set_color(colorlist[1])
        axin.spines['top'].set_color(colorlist[1])
        axin.spines['left'].set_color(colorlist[1])
        axin.spines['right'].set_color(colorlist[1])
        
        axin.tick_params(direction='in', right=True, top=True)
            
        leg = axin.legend(bbox_to_anchor=(0.8, -0.6), edgecolor='k')

        if save_fig:

            save_name = plot_params.get('save_name',  '{0}_spectra_overlayed_rep_{1}'.format(sample_ID, r))

            # plt.savefig(
                # os.path.join(save_path, save_name + '.png'),
                # transparent=fig_transparent,
                # dpi=save_dpi,
            # )
            plt.savefig(
                os.path.join(save_path, save_name + '.svg'),
                transparent=fig_transparent,
                dpi=save_dpi,
            )
            
        plt.close()