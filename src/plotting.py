"""Generic figure helpers shared by every stage of the pipeline.

Nothing here knows about sunspots. It holds the idioms that were re-typed in
`sunspot_analysis.py` and in the notebooks: saving a figure to a region's `plots/`
directory, laying a boolean mask over an image, and picking a symmetric colour range
for a magnetogram.

Region-specific figures live in `analysis`, `spectra`, `oscillation` and `comparison`.
"""

from __future__ import annotations

import pathlib

import numpy as np

# One colour per named region, so "green is always the umbra" holds across the project.
# These are the colours the FFT figures have always used; new figures follow them. The
# older per-region plots (`plot_area` in particular) predate this and still carry their own
# choices — left alone so existing figures do not silently change.
REGION_COLORS = {
    'umbra': 'green',
    'penumbra': 'purple',
    'both': 'steelblue',
    'quiet': 'darkorange',
    'sun_spot': 'purple',
    'hot_spot': 'green',
}

REGION_CMAPS = {
    'umbra': 'Greens',
    'penumbra': 'Purples',
    'both': 'Oranges',
    'sun_spot': 'Purples',
    'hot_spot': 'Greens',
    'quiet': 'Blues',
}

# Distinct outline colours for drawing several boxes on one full-disk map.
BOX_COLORS = ('red', 'cyan', 'yellow', 'lime', 'magenta', 'orange', 'deepskyblue', 'white')


def save_figure(fig, plots_dir, filename: str, save: bool = True, dpi: int = 150) -> pathlib.Path | None:
    """Write `fig` into `plots_dir`, creating the directory. No-op unless `save`.

    Returns the path written, or None. Callers used to spell this as
    `_savefig(fig, plots_dir if save else None, name)`; the `save` flag is a parameter
    here so that idiom does not have to be repeated at every call site.
    """
    if not save or plots_dir is None:
        return None
    plots_dir = pathlib.Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    out = plots_dir / filename
    fig.savefig(out, dpi=dpi, bbox_inches='tight')
    print(f'Figure saved → {out}')
    return out


def overlay_mask(ax, mask, region: str | None = None, *, cmap: str | None = None,
                 alpha: float = 0.5, origin: str = 'lower'):
    """Draw a boolean mask as a translucent wash over whatever `ax` already shows.

    `np.nan` outside the mask is what makes the rest of the image show through, and
    `vmin=0, vmax=1` pins the single value 1.0 to the top of the colormap so the wash
    has a constant colour regardless of how many pixels are set.
    """
    if cmap is None:
        cmap = REGION_CMAPS.get(region or '', 'Greys')
    return ax.imshow(np.where(mask, 1.0, np.nan), cmap=cmap, alpha=alpha,
                     origin=origin, vmin=0, vmax=1)


def symmetric_limits(frame, percentile: float = 99) -> float:
    """A symmetric colour limit for signed data: the given percentile of |values|.

    A plain min/max lets one bad pixel wash a magnetogram out, and an asymmetric range
    puts zero field somewhere other than the middle of a diverging colormap.
    """
    finite = np.asarray(frame)[np.isfinite(frame)]
    if finite.size == 0:
        return 1.0
    vmax = float(np.nanpercentile(np.abs(finite), percentile))
    return vmax if vmax > 0 else 1.0


def add_colorbar(fig, mappable, ax, label: str):
    """Colorbar sized to sit flush against a square image axis."""
    return fig.colorbar(mappable, ax=ax, label=label, fraction=0.046, pad=0.04)


def plot_regions_on_map(hmi_map_rot, regions, cmap=None, norm=None, colors=BOX_COLORS,
                        title: str | None = None, show: bool = True):
    """Draw a list of HPC box regions as labelled quadrangles on a rotated HMI map.

    Works for both continuum and magnetogram maps:
      - continuum   (BUNIT != Gauss): gray colormap with percentile clipping
      - magnetogram (BUNIT == Gauss): RdBu_r with a +/-500 G symmetric norm

    Parameters
    ----------
    hmi_map_rot : north-up `sunpy.map.Map` (already rotated)
    regions     : list of (bottom_left, top_right) SkyCoord pairs
    cmap, norm  : override the auto-detected colormap / norm
    colors      : outline colours, cycled over the regions
    title       : override the auto-generated title
    """
    import astropy.units as u
    import matplotlib.pyplot as plt

    is_magnetogram = 'gauss' in hmi_map_rot.meta.get('bunit', '').lower()

    if cmap is None:
        cmap = 'RdBu_r' if is_magnetogram else 'gray'
    if norm is None and is_magnetogram:
        norm = plt.Normalize(vmin=-500, vmax=500)
    if title is None:
        kind = 'magnetogram' if is_magnetogram else 'continuum'
        title = f'HMI {kind} — all regions'

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection=hmi_map_rot)

    if norm is not None:
        hmi_map_rot.plot(axes=ax, cmap=cmap, norm=norm)
    else:
        hmi_map_rot.plot(axes=ax, cmap=cmap, clip_interval=(1, 99.9) * u.percent)

    hmi_map_rot.draw_grid(axes=ax, color='white', alpha=0.3, lw=0.5)
    for i, (bl, tr) in enumerate(regions):
        hmi_map_rot.draw_quadrangle(bl, top_right=tr,
                                    edgecolor=colors[i % len(colors)],
                                    linewidth=2, label=f'Region {i + 1}')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(title)
    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax
