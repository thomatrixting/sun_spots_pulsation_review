# sun_spots_pulsation_review

Measuring oscillations in the umbra (and penumbra) of sunspots from SDO/HMI dopplergrams,
and testing whether the fitted amplitude relates to the Wilson depression.

Two datasets feed the same analysis:

- **NOAA** — active regions queried from the HEK, downloaded as tracked JSOC cutouts and
  corrected here. These are the regions Löptien et al. measured Wilson depressions for with
  Hinode, so there is an independent depression to compare a fitted amplitude against.
- **DS0N** — the `sebastian_sun_spots` datasets, delivered already processed.

They converge on the same in-memory structure, so every analysis function is shared. Only
the loaders differ.

## Layout

```
notebooks/
├── 01A_download_data.ipynb        A · NOAA · HEK query, JSOC cutouts, cube building
├── 02A_data_processing.ipynb      A · NOAA · corrections, uniform time grid
├── 03A_data_analysis.ipynb        A · NOAA · per-region analysis
├── 03B_data_analysis.ipynb        B · DS0N · per-region analysis, same eight steps
├── 04_data_comparison.ipynb       consolidated — across regions and across datasets
├── 01T_coefficient_reconstruction.ipynb   T · parked: inverting HMI's calibration cubic
├── 03T_analysis_sandbox.ipynb     T · scratch, seeded with the open questions
└── archive/                       the notebooks these grew out of, kept as a record

src/
├── config.py            paths, dataset registries, per-region parameters
├── utilities.py         cube I/O, the uniform time grid, per-frame correction driver
├── plotting.py          generic figure helpers
├── download.py          stage 1 — HEK / JSOC                                    (01A)
├── processing.py        stage 2 — corrections, driving:                         (02A)
│   ├── doppler_calibration.py   observatory velocity, flows, CLV, redshift
│   ├── limb_darkening.py        continuum I/C(mu)
│   └── post_processing.py       HMI calibration-cubic inversion (parked)        (01T)
├── segmentation.py      umbra / penumbra / hot-spot masks
├── loaders.py           load_noaa_region (A)  |  load_ds0n_region (B)
├── analysis.py          metrics, time series, mu diagnostics                    (03A/03B)
├── spectra.py           FFT / PSD, peaks, detrending, notch filter
├── oscillation.py       amplitude by fitting
├── animation.py         HTML animations
└── comparison.py        stage 4 — cross-region, cross-dataset                   (04)

tests/                   run directly: python tests/test_spectra.py
data/
├── raw/                 downloads and delivered cubes  (gitignored)
└── processed/           corrected cubes, metrics, figures  (gitignored)
```

### The three lines

**A** and **B** are the same eight steps on different data, so `03A` and `03B` can be read
side by side. **T** is where an analysis is tried before it earns a step number.

| | line A (NOAA) | line B (DS0N) |
|---|---|---|
| loader | `loaders.load_noaa_region` | `loaders.load_ds0n_region` |
| on disk | `region_01_*_cube.fits` on a uniform grid | IDL-style `cube_*.fits` + `.sav` timing |
| thresholds | fractions of each frame's own quiet sun | absolute DN |
| corrections | applied by `02A` | applied before delivery |

The per-region analysis steps, identical on both lines:

1. load cubes, build regions
2. look at the masks
3. per-frame metrics → `metrics.csv`
4. time series
5. spectra (FFT, PSD, band-limited amplitude)
6. detrending (rolling mean, notch filter)
7. geometry check (μ)
8. animation

## Where the parameters live

Everything that used to be a hardcoded value in a notebook cell is in `src/config.py`:
which ARs to download, per-AR series and box overrides, the correction settings, the
segmentation thresholds, the Löptien reference table, and the fit windows.

A per-region deviation goes in `REGION_PARAMS`, and every notebook block reads it through
`config.params_for(region_dir)`. Nothing is retyped per block.

`src/` holds **general** functions only. Anything hardwired to one region, one dataset or
one figure — called once to make one plot — stays in the notebook that needs it.

## Running it

```bash
conda install --file requirements.txt      # or: pip install -r requirements.txt
python tests/test_spectra.py               # the tests run standalone, no pytest needed
```

Then, in order: `01A` (the only notebook needing the network) → `02A` → `03A`, and `03B`
independently, then `04`.

DS9 8.3 is useful for looking at the cubes directly; `03A` exports a mask cube for it.

## Reading the numbers

- **Intensities** are in DN/s, limb-darkening corrected. The DN→cgs factor is deliberately
  not applied: everything downstream is a ratio of intensities, so a global scale would only
  make the numbers harder to compare against the raw frames and against DS9.
- **Velocities** are absolute line-of-sight m/s, positive away from the observer. The
  umbra's own motion is `mean_dop_umb − mean_dop_quiet`. The *absolute* value still carries
  whatever the `v_SDO` correction left behind — which is itself diurnal, and so sits right on
  top of the 24 h period being measured. Always look at both.
- **B** is the line-of-sight field, not `|B|`, and is not corrected for `cos θ`.
- **Gaps** are slots where a series genuinely has no frame. They stay NaN. Spectra
  interpolate across them internally, which is a compromise rather than a fix — check the
  gap count from step 1 before trusting a spectrum.

## Known data problems

- `DS10`'s `cube_continuum.fits` is truncated (astropy: *buffer is too small for requested
  array*) and `DS11` has no continuum cube. Both are listed in `config.DS0N_BROKEN`; line B
  currently covers `DS00`–`DS09`.
- NOAA 11117 was downloaded with a magnetogram box (402×402) different from its other two
  series (433×433), and its box shrinks partway through the window. `02A` trims all three to
  their common data window as a repair — the real fix is to re-download it with one box.
- NOAA 11117 also runs its continuum and magnetogram 360 s out of phase from 2010-10-30, so
  they never share a grid slot and every mean B is NaN over that stretch.
