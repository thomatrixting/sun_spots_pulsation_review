# sun_spots_pulsation_review

This project analyzes sunspot pulsations using Solar Dynamics Observatory (SDO) data. It starts with 45-second cadence continuum, Dopplergram, and magnetogram observations for active regions observed on April 1, 2026. The initial workflow uses DS9 for image inspection and visualization, followed by deeper analysis with Python code. The goal is to later extend the pipeline to characterize a larger set of sunspots and more robustly quantify their pulsation behavior.


## Project Structure

├── data
│   ├── processed          # Processed data products and analysis outputs
│   └── raw                # Original SDO FITS files (April 1, 2026)
├── notebooks
│   └── 01_download_data.ipynb  # Data download and initial exploration
├── src                    # Python scripts and helper modules for the pipeline
├── README.md              # Project overview and usage instructions
└── requirements.txt       # Python dependencies for analysis and plotting

## Usage

1. Install the Python dependencies:
   - With pip:
     ```bash
     pip install -r requirements.txt
     ```
   - With conda:
     ```bash
     conda install --file requirements.txt
     ```

2. Install DS9 version 8.3 separately for image inspection and initial visualization.

3. Use the notebooks and scripts in `notebooks/` and `src/` to explore the SDO continuum, Dopplergram, and magnetogram data.


