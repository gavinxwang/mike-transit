# mike-transit -- an end-to-end package for performing transmission spectroscopy with Magellan/MIKE
`mike-transit` is a set of Python tools for generating an exoplanet transmission spectrum, starting from raw MIKE data. Key steps include wavelength calibration, outlier removal, stellar-effect and telluric correction, and building transmission spectra in the planetary rest frame. It is designed for the detection of atomic species in the upper atmospheres of exoplanets.

**Authors**: Gavin Wang ([gxwang22@gmail.com](gxwang22@gmail.com)), Jessica Spake

## Dependencies

1. `scipy`
2. `numpy`
3. `ray` -- not required but recommended; set `use_ray = False` in `constants.py` if not installed
4. `PyAstronomy`
5. `astropy`
6. `matplotlib`
7. `TelFit`

## Requirements

1. Install the [MIKE Pipeline](https://code.obs.carnegiescience.edu/pipelines/mike).
2. A model of stellar effects (center-to-limb variation and Rossiter-McLaughlin) is also needed and not provided by this code. See [Yan et al. (2017)](https://ui.adsabs.harvard.edu/abs/2017A%26A...603A..73Y/abstract) for an algorithm to calculate this model.
3. Download a BT-Settl stellar spectrum from [SVO](https://svo2.cab.inta-csic.es/theory/newov2/index.php?models=bt-settl) with properties closest to that of the host star.

## Workflow

Run every script from the data directory.

1. Set the target, folder names, and system properties in `constants.py`.
2. Run `change_target_names.py` to give each spectrum a unique object identifier. This prevents them from being merged into a single spectrum.
3. Extract the 1D spectra and related files using the [MIKE Pipeline](https://code.obs.carnegiescience.edu/pipelines/mike).
4. Run `calculate_phases.py` to compute the orbital phase, airmass and stellar radial velocity of each exposure from the FITS headers.
5. Run `write_order_fluxes.py` to extract the flux and uncertainties for each order. This also saves the pipeline wavelength solution, which is used directly for the blue arm.
6. Run `clean_uncertainties.py` to replace bad values and outliers in the pipeline uncertainties.
7. Run `get_tellurics.py` to fit a [TelFit](https://github.com/kgullikson88/Telluric-Fitter) telluric model to each red order, used in the red-arm wavelength calibration.
8. Refine the red-arm wavelength calibration with `solve_wavelength.py`, which calls `mike_wavelength_solver.py`. 
    1. Run with `roundno = 1` to fit each exposure against the BT-Settl spectrum and telluric model.
    2. Run `combine_spectrum.py` to build the combined out-of-transit stellar spectrum.
    3. Run with `roundno = 2` to fit each exposure against the combined stellar spectrum.
9. For the blue arm, run `blue_round2.py` to refine the pipeline wavelength solution. 
10. Run `generate_river.py` to build the 2D river plots for each feature. Steps include dividing by the out-of-transit stellar spectrum, removing outliers, subtracting stellar effects, and masking strong telluric lines.
11. Finally, run `transmission_spectrum.py` to shift the rivers into the planetary rest frame and generate the transmission spectrum around each feature.

## Sensitivity Prediction

To estimate the precision achievable with MIKE for any planet, input the system properties in `precision_prediction.py`.

## Acknowledgments

If you use this code, please cite Wang et al. (2026) and this repository via its Zenodo DOI.

Feel free to reach out with any questions.
