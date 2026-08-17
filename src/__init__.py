"""Sunspot pulsation analysis.

Import the module you need rather than the package: `from src import loaders, spectra`.
This file deliberately re-exports nothing — the previous version pulled `matplotlib`,
`scipy` and `pandas` in the moment anything under `src` was touched, and no notebook ever
used the names it exported.

Pipeline stages, in order:

    config          paths, dataset registries, per-region parameters
    utilities       cube I/O, the uniform time grid, per-frame correction driver
    plotting        generic figure helpers
    download        stage 1 — HEK/JSOC acquisition                        (01A)
    processing      stage 2 — corrections, on top of:                     (02A)
      doppler_calibration, limb_darkening, post_processing
    segmentation    umbra / penumbra / hot-spot masks
    loaders         stage 3 entry — load_noaa_region (A) | load_ds0n_region (B)
    analysis        per-region metrics, time series, mu diagnostics       (03A / 03B)
    spectra         FFT / PSD, peaks, detrending, notch filter
    oscillation     amplitude by fitting
    animation       HTML animations
    comparison      stage 4 — across regions and datasets                 (04)
"""
