"""Turning continuum frames into umbra / penumbra / hot-spot / quiet-sun masks.

Two threshold families live here side by side, because the two datasets were segmented
differently and forcing them together would silently change published numbers:

* `build_regions` (line A, NOAA) cuts at *fractions* of each frame's own quiet-sun
  intensity. An absolute cut makes mask areas drift as the region rotates, and for a
  near-limb region the whole frame can fall under a fixed penumbra cut.
* `masks_from_cubes` (line B, DS0N) cuts at absolute DN, as the delivered cubes were
  originally segmented.

Both then hand the combined umbra|penumbra footprint to `select_spot_cluster`, which is
what restricts the masks to ONE connected sunspot.

Despite the commit messages, none of this is k-means: it is intensity thresholding plus
connected-component labelling. `sklearn` is not a dependency.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from astropy.io import fits
from scipy import ndimage

from .config import DEFAULT_HOTSPOT_G, MIN_CLUSTER_PX  # noqa: F401  (re-exported)

def _keep_central_cluster(binary_img: np.ndarray, mode: str) -> np.ndarray:
    if mode not in ('largest', 'central'):
        raise ValueError(f"cluster_mode must be 'largest' or 'central', got {mode!r}")
    labeled, n = ndimage.label(binary_img)  # type: ignore[misc]
    if n == 0:
        return binary_img
    if mode == 'largest':
        sizes = ndimage.sum(binary_img, labeled, range(1, n + 1))
        keep = int(np.argmax(sizes)) + 1
    else:
        cy, cx = np.array(binary_img.shape) / 2
        centroids = ndimage.center_of_mass(binary_img, labeled, range(1, n + 1))
        dists = [np.hypot(y - cy, x - cx) for y, x in centroids]
        keep = int(np.argmin(dists)) + 1
    return (labeled == keep).astype(binary_img.dtype)


def select_spot_cluster(
    footprint: np.ndarray,
    mode: str = 'largest',
    track: bool = True,
    connectivity: int = 2,
    min_area: int = MIN_CLUSTER_PX,
) -> tuple[np.ndarray, dict]:
    """Keep one connected sunspot per frame out of a whole cube's thresholded footprint.

    Thresholding a continuum frame selects *every* dark pixel in the box — the target spot,
    the other members of the group, pores, and bad pixels. The area of that mask then moves
    for reasons that have nothing to do with the spot being measured: NOAA 11117's umbra
    area drifts 98% across its window, and every mean taken over the mask inherits that.
    Labelling the footprint into connected components and keeping one of them measures a
    sunspot instead of a box.

    Why the *combined* footprint (umbra | penumbra) rather than each mask separately, which
    is what `_keep_central_cluster` is used for in `masks_from_cubes`: a sunspot is one
    connected dark region with its umbra nested inside its penumbra, so the combined mask is
    the thing that has one blob per spot. Labelling umbra and penumbra independently can
    pick the largest umbra from one spot and the largest penumbra from another. Intersect
    afterwards instead — ``umbra & spot``, ``penumbra & spot``.

    Not k-means, deliberately: k-means clusters pixels in some feature space and has no
    notion of spatial connectedness, so it will happily merge two separate spots into one
    cluster and split one spot in half. Connected-component labelling is the operation that
    means "these pixels are the same spot".

    Parameters
    ----------
    footprint : ndarray, (n_t, ny, nx) bool
        The combined umbra|penumbra mask for every frame.
    mode : {'largest', 'central'}
        How the target is chosen in the first frame, and whenever tracking loses it:
        by area, or by centroid distance to the frame centre. 'central' is meaningful
        because `locate_ar_window` centres the cutout on the catalogue centroid.
    track : bool
        Follow the same spot from frame to frame by maximum pixel overlap with the previous
        selection, rather than re-running `mode` independently each time. Two comparable
        spots make plain 'largest' flip between them mid-window, which puts a step in every
        series — the exact artifact this function exists to remove.
    connectivity : {1, 2}
        1 = 4-connectivity, 2 = 8-connectivity (default), so diagonally touching penumbral
        pixels stay one blob.
    min_area : int
        Discard components below this many pixels before choosing — see `MIN_CLUSTER_PX`
        for why this is not optional in practice. 0 disables it.

    Returns
    -------
    (spot, info)
        `spot` is a bool cube of the same shape, True only inside the selected component,
        so ``spot`` is a subset of ``footprint`` by construction.
        `info` holds per-frame `n_clusters` (after the `min_area` cut), `area`, `fraction`
        (selected / total footprint), `centroid` (n_t, 2) and `switched`. **Look at
        `switched` and `fraction`.** Tracking losing the spot is the one way this makes
        things worse, and it is invisible in the masks themselves.

    Notes
    -----
    `fraction` is deliberately measured against the *whole* footprint, speckle included, so
    it stays an honest "how much of the dark area is this spot" rather than flattering
    itself by excluding what `min_area` already threw away.

    A `switched` frame is not automatically a bug. On NOAA 11536 the spot decays from 473
    to 18 px across a 4-day window, and the tracker re-picks once at the very end when
    there is essentially nothing left to track. Check *when* it happened before treating it
    as one.
    """
    if mode not in ('largest', 'central'):
        raise ValueError(f"mode must be 'largest' or 'central', got {mode!r}")

    footprint = np.asarray(footprint, dtype=bool)
    n_t = footprint.shape[0]
    structure = ndimage.generate_binary_structure(2, connectivity)
    centre = np.array(footprint.shape[1:]) / 2

    spot = np.zeros_like(footprint)
    info = dict(
        n_clusters=np.zeros(n_t, dtype=int),
        area=np.zeros(n_t, dtype=int),
        fraction=np.full(n_t, np.nan),
        centroid=np.full((n_t, 2), np.nan),
        switched=np.zeros(n_t, dtype=bool),
    )

    # The tracking reference is the last frame in which something was actually selected,
    # not literally t-1: a NaN gap frame selects nothing and must not break the chain.
    previous = None

    for t in range(n_t):
        frame = footprint[t]
        if not frame.any():
            continue

        labeled, n = ndimage.label(frame, structure=structure)
        if n == 0:
            continue
        labels = np.arange(1, n + 1)
        sizes = ndimage.sum_labels(frame, labeled, labels)

        if min_area:
            big = sizes >= min_area
            if not big.any():
                continue                        # nothing here but speckle
            labels, sizes = labels[big], sizes[big]
            # Blank the discarded components so overlap and centroids ignore them too.
            labeled = np.where(np.isin(labeled, labels), labeled, 0)
        info['n_clusters'][t] = len(labels)

        keep = None
        if track and previous is not None:
            # Overlap of every label with the previous selection, in one pass.
            overlap = np.bincount(labeled[previous].ravel(), minlength=n + 1)
            overlap[0] = 0                      # label 0 is background
            if overlap.max() > 0:
                keep = int(overlap.argmax())
            else:
                info['switched'][t] = True      # lost it — fall through to `mode`

        if keep is None:
            if mode == 'largest':
                keep = int(labels[np.argmax(sizes)])
            else:
                centroids = ndimage.center_of_mass(frame, labeled, labels)
                dists = [np.hypot(y - centre[0], x - centre[1]) for y, x in centroids]
                keep = int(labels[np.argmin(dists)])

        selected = labeled == keep
        spot[t] = selected
        info['area'][t] = int(selected.sum())
        info['fraction'][t] = info['area'][t] / frame.sum()
        info['centroid'][t] = ndimage.center_of_mass(selected)
        previous = selected

    return spot, info


def masks_from_cubes(
    cube_cont: np.ndarray,
    cube_mag: np.ndarray,
    cube_dop: np.ndarray,
    cube_dop_qsun: np.ndarray,
    cube_mag_qsun: np.ndarray,
    umbra_thresh: float = 30_000,
    penumbra_thresh: float = 50_000,
    cluster_mode: str = 'largest',
    cadence_s: float = 720.0,
    filter_mask: bool = True,
    filter_both: bool = False,
    mag_filter: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict:
    """
    Build region masks and the ``load_ds0n_region``-style data dict from
    already-loaded (n_t, ny, nx) cubes, instead of reading FITS files from
    disk. Used by ``load_ds0n_region`` internally, and directly by callers that
    have derived/reconstructed cubes (e.g. from a coefficient-inversion
    pipeline) rather than the on-disk ones.

    Parameters mirror ``load_ds0n_region`` — see its docstring for details.
    """
    n_t = cube_cont.shape[0]

    _cad = np.asarray(cadence_s, dtype=float)
    if _cad.ndim == 0:
        time_h = np.arange(n_t) * float(_cad) / 3600
        median_cadence = float(_cad)
    else:
        time_h = np.concatenate([[0.0], np.cumsum(_cad)]) / 3600
        median_cadence = float(np.median(_cad))

    umbra_raw    = (cube_cont < umbra_thresh)    & np.isfinite(cube_cont)
    penumbra_raw = (cube_cont < penumbra_thresh) & np.isfinite(cube_cont) & ~umbra_raw

    if filter_mask:
        umbra    = np.array([_keep_central_cluster(umbra_raw[t],    cluster_mode) for t in range(n_t)], dtype=bool)
        penumbra = np.array([_keep_central_cluster(penumbra_raw[t], cluster_mode) for t in range(n_t)], dtype=bool)
        del umbra_raw, penumbra_raw
    else:
        umbra = umbra_raw
        penumbra = penumbra_raw

    both = umbra | penumbra if filter_both else (cube_cont < penumbra_thresh) & np.isfinite(cube_cont)

    hot_spot = None
    if mag_filter is not None:
        hot_spot = mag_filter(cube_mag) & (umbra | penumbra)

    return dict(
        cube_cont=cube_cont, cube_mag=cube_mag, cube_dop=cube_dop,
        cube_dop_qsun=cube_dop_qsun, cube_mag_qsun=cube_mag_qsun,
        umbra=umbra, penumbra=penumbra, both=both, hot_spot=hot_spot,
        time_h=time_h, n_t=n_t, cadence_s=median_cadence,
        umbra_thresh=umbra_thresh, penumbra_thresh=penumbra_thresh,
    )


def build_regions(
    cube_cont: np.ndarray,
    cube_mag: np.ndarray | None = None,
    umbra_frac: float = 0.60,
    penumbra_frac: float = 0.90,
    qsun_percentile: int = 80,
    cluster_mode: str | None = 'largest',
    cluster_track: bool = True,
    cluster_min_area: int = MIN_CLUSTER_PX,
    mag_filter: Callable[[np.ndarray], np.ndarray] | None = None,
    hotspot_gauss: float = DEFAULT_HOTSPOT_G,
    custom_valid_region: np.ndarray | None = None,
) -> dict:
    """Segment a continuum cube into umbra / penumbra / hot spot / quiet sun.

    This is the analysis-side counterpart to the corrections in
    ``notebooks/02A_data_procesing.ipynb``: 02A produces the three corrected cubes and stops,
    and everything here is re-derivable from them at any time. That split exists so that
    retuning a threshold or a tracker means re-running 03A only — 02A costs ~90 s per region
    because the limb-darkening and Doppler corrections re-read every per-frame FITS header,
    and none of that work depends on where the umbra boundary is drawn.

    Thresholds are fractions of each frame's **own** quiet-sun intensity, not absolute DN
    (which is what `masks_from_cubes` takes, for the DS0X path). An absolute cut makes the
    mask areas drift as the region rotates, and for a near-limb region the whole frame can
    fall under a fixed penumbra cut.

    Parameters
    ----------
    cube_cont : ndarray, (n_t, ny, nx)
        Limb-darkening-corrected continuum, as 02A writes it.
    cube_mag : ndarray, optional
        Plane-corrected magnetogram. Needed only for the hot spot; without it `hot_spot`
        comes back None.
    umbra_frac, penumbra_frac : float
        ``I < frac * I_qs``. Penumbra additionally excludes umbra.
    qsun_percentile : int
        Percentile of the frame's finite pixels used as its ``I_qs``.
    cluster_mode : {'largest', 'central', None}
        Restrict umbra/penumbra to one connected sunspot — see `select_spot_cluster`. None
        keeps every dark pixel in the box, other spots and pores included.
    mag_filter : callable, optional
        ``B -> bool`` for the hot spot. Defaults to ``|B| > hotspot_gauss``, on ``|B|``
        rather than signed B because getting the sign wrong yields an *empty* mask rather
        than an error.

    Returns
    -------
    dict
        ``umbra``, ``penumbra``, ``both``, ``hot_spot``, ``qsun`` bool cubes; ``i_qs``;
        ``cluster_info`` (or None); ``raw_area_px`` for the footprint before the cluster
        selection; and ``umbra_thresh`` / ``penumbra_thresh``, the median absolute DN the
        fractional cuts worked out to, which the plot legends quote.

    Notes
    -----
    Every step is NaN-safe, because a gap frame on 02A's uniform time grid is entirely NaN:
    the percentile guards on there being finite pixels, and a comparison against a NaN
    threshold is False, so a gap frame simply gets empty masks.

    **Quiet sun is the complement of the raw footprint, deliberately** — computed before the
    cluster selection narrows things down. Otherwise every dark pixel the selection
    discarded, a second sunspot's umbra included, would land in the quiet-sun mask and
    contaminate the quiet-sun reference that everything downstream subtracts.
    """
    cube_cont = np.asarray(cube_cont)
    n_t = cube_cont.shape[0]

    finite = np.isfinite(cube_cont)
    i_qs = np.array([np.percentile(cube_cont[t][finite[t]], qsun_percentile)
                     if finite[t].any() else np.nan for t in range(n_t)])

    with np.errstate(invalid='ignore'):
        raw_umbra = (cube_cont < (umbra_frac * i_qs)[:, None, None]) & finite
        raw_pen   = (cube_cont < (penumbra_frac * i_qs)[:, None, None]) & finite & ~raw_umbra
    raw_both = raw_umbra | raw_pen

    if custom_valid_region is not None:
        custom_valid_region = np.asarray(custom_valid_region, dtype=bool)
        if custom_valid_region.shape != cube_cont.shape[1:]:
            raise ValueError(
                f'custom_valid_region shape {custom_valid_region.shape} '
                f'does not match cube_cont shape {cube_cont.shape[1:]}'
            )
        raw_umbra &= custom_valid_region
        raw_pen   &= custom_valid_region
        raw_both  &= custom_valid_region

    qsun = finite & ~raw_both

    cluster_info = None
    if cluster_mode:
        spot, cluster_info = select_spot_cluster(
            raw_both, mode=cluster_mode, track=cluster_track, min_area=cluster_min_area)
        umbra, penumbra, both = raw_umbra & spot, raw_pen & spot, spot
    else:
        umbra, penumbra, both = raw_umbra, raw_pen, raw_both

    hot_spot = None
    if cube_mag is not None:
        if mag_filter is None:
            def mag_filter(b):
                return np.abs(b) > hotspot_gauss
        with np.errstate(invalid='ignore'):
            hot_spot = mag_filter(cube_mag) & both

    return dict(
        umbra=umbra, penumbra=penumbra, both=both, hot_spot=hot_spot, qsun=qsun,
        i_qs=i_qs, cluster_info=cluster_info,
        raw_area_px={'umbra': raw_umbra.reshape(n_t, -1).sum(axis=1),
                     'penumbra': raw_pen.reshape(n_t, -1).sum(axis=1),
                     'both': raw_both.reshape(n_t, -1).sum(axis=1)},
        umbra_thresh=float(np.nanmedian(i_qs) * umbra_frac),
        penumbra_thresh=float(np.nanmedian(i_qs) * penumbra_frac),
        cluster_mode=cluster_mode, umbra_frac=umbra_frac, penumbra_frac=penumbra_frac,
    )


#: Bit values of the mask cube `write_masks_cube` produces, also written as BIT_* keywords.
BIT_UMBRA, BIT_PENUMBRA, BIT_HOTSPOT = 1, 2, 4


def write_masks_cube(data: dict, path, header=None, history=None):
    """Write the regions in `data` as one ``uint8`` bitmask cube, for DS9.

    A single integer cube rather than three float ones: DS9 renders integers far more
    cleanly, and one file keeps the overlapping hot spot alongside the regions it sits
    inside. The bit values and threshold fractions go in as keywords rather than as a
    convention, so a reader gets what was actually used instead of assuming defaults.

    This is an **export**, not an input. `load_noaa_region` rebuilds the masks from the
    cubes every time rather than reading this file, so it can never go stale against the
    thresholds currently set in the notebook.
    """
    from .utilities import write_cube

    masks = (data['umbra'].astype(np.uint8) * BIT_UMBRA
             | data['penumbra'].astype(np.uint8) * BIT_PENUMBRA)
    if data.get('hot_spot') is not None:
        masks |= data['hot_spot'].astype(np.uint8) * BIT_HOTSPOT

    header = fits.Header() if header is None else header.copy()
    header['BUNIT']    = ('', 'bit flags, see BIT_* keywords')
    header['BIT_UMB']  = (BIT_UMBRA, 'bit value for umbra')
    header['BIT_PEN']  = (BIT_PENUMBRA, 'bit value for penumbra')
    header['BIT_HOT']  = (BIT_HOTSPOT, 'bit value for hot spot')
    header['UMB_FRAC'] = (data.get('umbra_frac', np.nan), 'umbra threshold / I_qs')
    header['PEN_FRAC'] = (data.get('penumbra_frac', np.nan), 'penumbra threshold / I_qs')
    header['CLUSTER']  = (data.get('cluster_mode') or 'none', 'connected-component selection')

    return write_cube(masks, path, header=header, timestamps=data.get('timestamps'),
                      history=list(history or []) + [
                          '03A: 0 = quiet sun; bits overlap (hot spot is inside the spot)',
                          '03A: 0 also covers dark pixels outside the selected spot - '
                          'they are NOT quiet sun',
                          '03A: a gap frame has every bit 0'])
