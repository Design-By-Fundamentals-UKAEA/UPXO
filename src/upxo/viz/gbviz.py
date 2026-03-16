import numpy as np
import matplotlib.pyplot as plt

def see_all_gbsegs(gbsegImage, figsize=(5, 5), dpi=75, cmap='nipy_spectral',
            title="GB segments", xlabel="X-axis, um", ylabel="Y-axis, um",
            cbarLabel="Segment ID (float)"):
    """
    Visualize the GB segments in the given image.

    Parameters
    ----------
    gbsegImage : 2D array-like
        The image containing the GB segments, where each segment is represented by a unique ID (float).
    
    Import
    ------
    import upxo.viz.gbviz as gbViz
    Use as: gbViz.see_all_gbsegs
    """
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(gbsegImage, cmap=cmap, origin='lower')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(label=cbarLabel)
    plt.show()

def see_gbsegs_fid(gbsegImage, fid, figsize=(5, 5), dpi=75, cmap='nipy_spectral',
            title="GB segments", xlabel="X-axis, um", ylabel="Y-axis, um",
            cbarLabel="Segment ID (float)"):
    """
    import upxo.viz.gbviz as gbViz
    Use as: gbViz.see_gbsegs_fid
    """
    scaled_mask = gbsegImage*np.asarray(gbsegImage >= fid, dtype=float)*np.asarray(gbsegImage < fid+1, dtype=float)
    scaled_mask[scaled_mask<fid]= np.nan

    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(scaled_mask, cmap=cmap, origin='lower')
    plt.title(title+f". FID={fid}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(label=cbarLabel)
    plt.show()

def see_gbsegs_jp_by_jpo(gbsegs, jps_by_order, style_by_order=None, default_style=None,
            figsize=(5, 5), dpi=80, legend_anchor=(1.02, 1.0), ms2=4,
            ms3=4, ms4=4, ms5=4, legend_loc='upper left', legend_title='Junction point data',
            legend_frameon=True, hide_axis=True, cmap='rainbow'):
    """
    Import
    import upxo.viz.gbviz as gbViz
    Use as: gbViz.plot_gbsegs_by_order
    """
    gbsegs[gbsegs == 0.0] = np.nan
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(gbsegs, cmap=cmap)
    if style_by_order is None:
        style_by_order = {2: dict(fmt='o', color='white', ms=ms2, label='2-junction'),
                          3: dict(fmt='o', color='black', ms=ms3, label='3-junction'),
                          4: dict(fmt='s', color='red', ms=ms4, label='4-junction'),
                          5: dict(fmt='^', color='green', ms=ms5, label='5-junction'),}
    if default_style is None:
        default_style = dict(fmt='s', color='k', ms=6, label='>=5-junction')
    used_labels = set()
    for order, groups in sorted(jps_by_order.items()):
        st = style_by_order.get(order, default_style)
        for pts in groups.values():
            if pts.size == 0:
                continue
            label = st['label'] if st['label'] not in used_labels else None
            ax.plot(pts[:, 1], pts[:, 0], st['fmt'], color=st['color'], 
                    markersize=st['ms'], markeredgecolor='black', label=label)
            if label is not None:
                used_labels.add(label)
    if used_labels:
        ax.legend(loc=legend_loc, bbox_to_anchor=legend_anchor, title=legend_title,
                  frameon=legend_frameon, borderaxespad=0.0)
    if hide_axis:
        ax.axis('off')
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04, label='Segment ID (float)')
    return fig, ax