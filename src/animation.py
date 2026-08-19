"""HTML animations of a region's cubes with its masks overlaid.

Its own module so `matplotlib.animation` — which pulls in writers and can be slow to
import — is only loaded when an animation is actually wanted.
"""

from __future__ import annotations

import pathlib

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy import ndimage
from matplotlib.animation import FuncAnimation, HTMLWriter

from .plotting import save_figure, symmetric_limits  # noqa: F401  (save_figure: parity)

def save_animation(
    data: dict,
    metrics: dict,
    save_path: str | pathlib.Path,
    step: int = 50,
    fps: int = 5,
    embed_limit_mb: float = 50.0,
    mag_symmetric_cbar: bool = True,
    dop_symmetric_cbar: bool = True,
    embed_frames: bool = False,
) -> None:
    """
    Save a 3-channel (continuum / magnetogram / dopplergram) animation as HTML.

    Parameters
    ----------
    data               : dict from load_ds0n_region()
    metrics            : dict from compute_metrics()
    save_path          : output .html file path (parent dirs created if needed)
    step               : subsample every Nth frame to keep file size manageable
    fps                : frames per second
    embed_limit_mb     : matplotlib animation size limit in MB
    embed_frames       : if True (default) the frames are base64'd into the .html, so it
                         is a single self-contained file that can be moved or shared.
                         False writes them to a sibling ``<name>_frames/`` directory
                         instead — smaller, but the .html breaks if it is moved without
                         that directory. Raise ``step`` if an embedded file gets too big.
    mag_symmetric_cbar : if True (the default), centre the magnetogram colorbar on zero:
                         vmin = −vmax, with vmax the 99th percentile of |B| over the
                         sampled frames. False keeps the [2nd, 98th] percentile limits.
    dop_symmetric_cbar : the same for the dopplergram, and on by default for the same
                         reason. `RdBu_r` is a diverging colormap, so an asymmetric range
                         puts zero velocity somewhere other than the white midpoint — a
                         blueshift then reads as a different size from a redshift of equal
                         magnitude, and the panel cannot be compared frame to frame or
                         against the magnetogram beside it.
    """
    save_path = pathlib.Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams['animation.embed_limit'] = embed_limit_mb

    cube_cont    = data['cube_cont']
    cube_mag     = data['cube_mag']
    cube_dop     = data['cube_dop']
    umbra        = data['umbra']
    penumbra     = data['penumbra']
    n_t          = data['n_t']
    cadence_s    = data['cadence_s']
    mean_mag_umb = metrics['mean_mag_umb']
    mean_mag_pen = metrics['mean_mag_pen']

    frames_idx = np.arange(0, n_t, step)
    cubes_ch   = [cube_cont,        cube_mag,           cube_dop]
    cmaps_ch   = ['gray',           'bwr',              'RdBu_r']
    labels_ch  = ['Continuum (DN)', 'Magnetogram (G)',  'Dopplergram (m/s)']
    _sample    = np.arange(0, n_t, max(1, n_t // 50))
    clims      = [np.nanpercentile(c[_sample], [2, 98]) for c in cubes_ch]

    # Both signed channels go on diverging colormaps, where the midpoint colour *is* the
    # zero. `symmetric_limits` is the same rule the still figures in `analysis` use, so a
    # frame grabbed from the animation and one plotted directly share a scale — and it
    # falls back to a usable limit on an all-NaN sample, which a bare percentile does not.
    for channel, symmetric in ((1, mag_symmetric_cbar), (2, dop_symmetric_cbar)):
        if symmetric:
            vmax = symmetric_limits(cubes_ch[channel][_sample], percentile=99)
            clims[channel] = np.array([-vmax, vmax])

    _LEGEND_HANDLES = [
        mpatches.Patch(color=(0.0, 0.85, 0.0, 0.75), label='Umbra'),
        mpatches.Patch(color=(0.55, 0.0, 1.0, 0.75), label='Penumbra'),
    ]

    def _make_filled_rgba(umb, pen):
        """Filled colour overlay for the continuum panel."""
        h, w = umb.shape
        rgba = np.zeros((h, w, 4), dtype=float)
        rgba[pen, 0] = 0.55; rgba[pen, 2] = 1.0;  rgba[pen, 3] = 0.40
        rgba[umb, 0] = 0.0;  rgba[umb, 1] = 0.85; rgba[umb, 3] = 0.45
        return rgba

    def _make_outline_rgba(umb, pen, thickness=2):
        """Border-only overlay for magnetogram and dopplergram panels."""
        h, w = umb.shape
        rgba = np.zeros((h, w, 4), dtype=float)
        pen_border = ndimage.binary_dilation(pen, iterations=thickness) & ~pen
        umb_border = ndimage.binary_dilation(umb, iterations=thickness) & ~umb
        rgba[pen_border, 0] = 0.55; rgba[pen_border, 2] = 1.0;  rgba[pen_border, 3] = 0.95
        rgba[umb_border, 0] = 0.0;  rgba[umb_border, 1] = 0.85; rgba[umb_border, 3] = 0.95
        return rgba

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    fig.subplots_adjust(wspace=0.3, top=0.88, left=0.06, right=0.97)
    ims = []; overlays = []

    for idx, (ax, cube, cmap, clim, lbl) in enumerate(
            zip(axes, cubes_ch, cmaps_ch, clims, labels_ch)):
        im = ax.imshow(cube[0], origin='lower', cmap=cmap,
                       vmin=clim[0], vmax=clim[1], interpolation='nearest')
        fig.colorbar(im, ax=ax, label=lbl, fraction=0.046, pad=0.06)
        ov_data = (_make_filled_rgba if idx == 0 else _make_outline_rgba)(umbra[0], penumbra[0])
        ov = ax.imshow(ov_data, origin='lower', interpolation='nearest')
        ax.set_xlabel('X (px)')
        ax.set_ylabel('Y (px)')
        ax.text(0.02, 0.02, f'Cadence: {cadence_s:.0f} s', transform=ax.transAxes,
                color='white', fontsize=9, va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6))
        ax.legend(handles=_LEGEND_HANDLES, loc='upper right', fontsize=7)
        ims.append(im); overlays.append(ov)

    suptitle = fig.suptitle('', fontsize=11)

    def _update(i):
        t = frames_idx[i]
        for j, (im, ov, cube) in enumerate(zip(ims, overlays, cubes_ch)):
            im.set_data(cube[t])
            ov_data = (_make_filled_rgba if j == 0 else _make_outline_rgba)(umbra[t], penumbra[t])
            ov.set_data(ov_data)
        elapsed_h = data['time_h'][t]
        b_u = f'{mean_mag_umb[t]:.0f}' if np.isfinite(mean_mag_umb[t]) else 'N/A'
        b_p = f'{mean_mag_pen[t]:.0f}' if np.isfinite(mean_mag_pen[t]) else 'N/A'
        suptitle.set_text(
            f'SDO/HMI  |  frame {t:04d}  |  t = {elapsed_h:.2f} h  |  '
            f'<B> umb={b_u} G   pen={b_p} G'
        )
        return ims + overlays

    anim = FuncAnimation(fig, _update, frames=len(frames_idx),
                         interval=1000 // fps, blit=False)
    plt.close(fig)
    # HTMLWriter defaults to embed_frames=False, which scatters the frames into a sibling
    # directory the .html then depends on. Pass the writer explicitly so the default here
    # is a single portable file, as embed_limit_mb always implied.
    anim.save(str(save_path),
              writer=HTMLWriter(fps=fps, embed_frames=embed_frames))
    size_mb = save_path.stat().st_size / 1e6
    print(f'Animation saved → {save_path}  ({len(frames_idx)} frames, {size_mb:.1f} MB'
          f'{"" if embed_frames else ", frames in a sibling directory"})')
    return save_path
