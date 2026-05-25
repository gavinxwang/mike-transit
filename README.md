# mike-transit -- an end-to-end package for performing transmission spectroscopy with Magellan/MIKE
`mike-transit` is a set of Python tools for generating an exoplanet transmission spectrum, starting from raw MIKE data. Key steps include wavelength calibration, outlier removal, and measuring empirical uncertainties via injection-recovery. It is designed for the detection of atomic species in the upper atmospheres of exoplanets.

**Authors**: Gavin Wang ([gxwang22@gmail.com](gxwang22@gmail.com)), Jessica Spake

## Requirements

1. Install the [MIKE Pipeline](https://code.obs.carnegiescience.edu/pipelines/mike).
2. A model of stellar effects (center-to-limb variation and Rossiter-McLaughlin) is also needed and not provided by this code. See [Yan et al. (2017)](https://ui.adsabs.harvard.edu/abs/2017A%26A...603A..73Y/abstract) for an algorithm to calculate this model.
3. Download a BT-Settl stellar spectrum from [SVO](https://svo2.cab.inta-csic.es/theory/newov2/index.php?models=bt-settl) with properties closest to that of the host star.

## Workflow

1. Edit and run `change-target-names.py` to give each spectrum a unique object identifier. This prevents them from being merged into a single spectrum.
2. Extract the 1D spectra and related files using the [MIKE Pipeline](https://code.obs.carnegiescience.edu/pipelines/mike).
3. Run `write-order-fluxes.py` to extract the flux for each order.
4. For the red arm, refine the wavelength calibration using `run_wl_solver.py`, which calls `MIKE_wavelength_solver.py`. Remember to input the system properties. Run first with `roundno = 1`, then `roundno = 2`.
5. For the blue arm, due to the lower signal and discrepancy with models, run `blue-arm-solver.py`.
6. Run `generate-spectrum.py` to get the transmission spectrum. Steps include building the combined out-of-transit stellar spectrum, subtracting stellar effects, and performing telluric correction with [SysRem](https://ui.adsabs.harvard.edu/abs/2005MNRAS.356.1466T/abstract) before generating the transmission spectrum in the planetary rest frame.
7. Run `inject-recovery.py` to obtain the optimal number of SysRem iterations as well as empirical uncertainties on measured planetary absorption/emission.
8. Finally, run `fit-absorption.py` to search for planetary atmospheric signals.

## Acknowledgments

If you use this code, please cite [add link] and provide a link to this repository.

Feel free to reach out with any questions.
