# Build combined out-of-transit stellar spectrum for each order

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from constants import (
    BASE_DIR, PHASES_FILE, SPECTRA_FOLDER_BLUE, SPECTRA_FOLDER_RED, target_string,
    get_geometry, oot_mask,
)

orders = range(40, 107)
TRUNC = 10  # pixels dropped from each order edge, where the interpolation extrapolates


def load_oot_spectra(spectra_dir, order, oot):
    wls, fluxes = [], []
    for exp in np.flatnonzero(oot):
        spec = np.load(spectra_dir / f'{target_string}_spec_wlsol_round1_order{order}_exp{exp}.npy')
        wls.append(spec[:, 0])
        fluxes.append(spec[:, 1])
    return np.array(wls), np.array(fluxes)


def combine(wls, fluxes):
    # interpolate every exposure onto the first exposure's grid, then take the median
    wl_grid = wls[0]
    interp_fluxes = np.array([
        interpolate.interp1d(wl, flux, kind='cubic', bounds_error=False, fill_value='extrapolate')(wl_grid)
        for wl, flux in zip(wls, fluxes)
    ])
    combined = np.median(interp_fluxes, axis=0)
    return np.column_stack((wl_grid[TRUNC:-TRUNC], combined[TRUNC:-TRUNC]))


def plot_combined(wls, fluxes, combined, order, blue, outpath):
    # blue arm stays in the earth frame; red arm is in the stellar frame after round 1
    frame = 'Earth' if blue else 'Stellar'
    arm = 'blue' if blue else 'red'

    plt.figure(figsize=(15, 6))
    plt.plot(wls.T, fluxes.T, 'o-', alpha=0.3, c='tab:blue')
    plt.plot(combined[:, 0], combined[:, 1], '-', lw=0.5, c='k')
    plt.xlim(combined[0, 0], combined[-1, 0])
    plt.ylim(0, 1.5)
    plt.xlabel(r'Wavelength ($\AA$)')
    plt.ylabel('Normalized flux')
    plt.title(f'{frame} frame, order {order} ({arm} arm)')
    plt.tight_layout()
    plt.savefig(outpath, facecolor='w', dpi=200)
    plt.close()


def main():
    plt.style.use('paper.mplstyle')
    plot_dir = BASE_DIR / 'Plots' / f'{target_string}_combined'
    plot_dir.mkdir(parents=True, exist_ok=True)

    phases_full = np.loadtxt(PHASES_FILE)

    for order in orders:
        blue, nexp, exposure_to_full = get_geometry(order, len(phases_full))
        spectra_dir = BASE_DIR / (SPECTRA_FOLDER_BLUE if blue else SPECTRA_FOLDER_RED)
        phases = phases_full[exposure_to_full] if exposure_to_full is not None else phases_full

        n_found = len(list(spectra_dir.glob(f'{target_string}_spec_wlsol_round1_order{order}_exp*.npy')))
        if n_found != nexp:
            print(f'order {order}: missing {nexp - n_found} exposures, skipping')
            continue

        wls, fluxes = load_oot_spectra(spectra_dir, order, oot_mask(phases))
        combined = combine(wls, fluxes)

        np.save(spectra_dir / f'{target_string}_stellar_spec_transit_order{order}.npy', combined)
        plot_combined(wls, fluxes, combined, order, blue, plot_dir / f'combined_order{order}.png')
        print(f'order {order}: combined {len(wls)} out-of-transit exposures')


if __name__ == '__main__':
    main()
