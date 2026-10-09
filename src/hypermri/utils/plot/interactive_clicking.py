# function for interactive clicking on pixels and then displaying the csi
# spectrum of that pixel

plt.rcParams["figure.figsize"] = (12, 5)


def show_spectrum(event):
    """
    gets called by fig.canvas.mpl_connect('button_press_event',show_spec) for plotting interactively
    :param event:
    :return:
    """
    x_coord = int(np.round(event.ydata, 0))
    y_coord = int(np.round(event.xdata, 0))
    # clear figure if we hit click
    ax3.cla()
    spec_disp = (
        spec_csi[:, y_coord, x_coord]
        - np.mean(spec_csi[bg[0] : bg[1], y_coord, x_coord])
    ) / np.std(spec_csi[bg[0] : bg[1], y_coord, x_coord])
    ax3.plot(ppm_csi, spec_disp, linewidth=0.5, color="k")
    ax3.set_title("[" + str(y_coord) + "," + str(x_coord) + "]")
    ax3.set_xlabel(r"$\sigma$[ppm]")
    ax3.set_ylabel("SNR")
    ax3.set_xlim([np.max(ppm_csi), np.min(ppm_csi)])
    # ax3.set_xticks([195,185,175,165])
    if show_peak_lines is True:
        ax3.vlines(171.2, 0, np.max(spec_disp), linestyle="dashed", color="g")
        ax3.vlines(183.4, 0, np.max(spec_disp), linestyle="dashed", color="g")
    else:
        pass
    print(
        "[" + str(y_coord) + "," + str(x_coord) + "]",
        np.max(
            (
                spec_csi[:, y_coord, x_coord]
                - np.mean(spec_csi[bg[0] : bg[1], y_coord, x_coord])
            )
            / np.std(spec_csi[bg[0] : bg[1], y_coord, x_coord])
        ),
    )


# make the plot
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.set_title("CSI")
if interpolate is True:
    im = ax1.imshow(
        csi_snr_map, cmap="plasma", picker=True, alpha=1.0, interpolation="bicubic"
    )
else:
    im = ax1.imshow(csi_snr_map, cmap="plasma", picker=True, alpha=1.0)
ax1.set_xlabel("Voxel")
ax1.set_ylabel("Voxel")


# define the fov extent, see function in utils_anatomical
fov = Define_Extent(anat_ref_img)
ax2.set_title("T2W Anat")
# show anatomical reference image
ax2.imshow(
    (anat_ref_img.seq2d[:, :, 0]).T, cmap="gray", picker=True, alpha=1.0, extent=fov
)
ax2.set_xlabel("mm")
ax2.set_ylabel("mm")
fig.canvas.mpl_connect("button_press_event", show_spectrum)

ax3.set_title("Click on CSI")
plt.tight_layout()
