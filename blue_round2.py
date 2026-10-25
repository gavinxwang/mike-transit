# Refine the blue-arm wavelength solution against the combined out-of-transit stellar spectrum

import os
import re

import numpy as np
from scipy import interpolate
from scipy.optimize import differential_evolution

from constants import (
    BASE_DIR, BLUE_ORDERS, NUM_CPUS, PHASES_FILE, RAW_FOLDER, SPECTRA_FOLDER_BLUE,
    get_geometry, oot_mask, target_string, use_ray,
)
from mike_wavelength_solver import cheby3, cheby5, compute_rvs, get_spectra_list, scale_col
from planet import make_planet
from utils import doppler

orders = BLUE_ORDERS
force = False  # refit even if round2 files already exist
save_template = False  # also write the stellar-frame combined spectrum

TRUNC = 10
UNC_DIR = BASE_DIR / f'{target_string}_uncertainties_cleaned'

# wl' = wl_pipe + cheby3(x, a0..a3), f' = f / cheby5(x, c0..c5)
# identity: wavelength correction -> 0, continuum -> 1
BOUNDS = [
    (-0.5, 0.5),   # a0  T0 wavelength shift [A]
    (-2.0, 2.0),   # a1  T1
    (-2.0, 2.0),   # a2  T2
    (-2.0, 2.0),   # a3  T3
    (0.7, 1.3),    # c0  T0 continuum scale
    (-1.0, 1.0),   # c1  T1
    (-1.0, 1.0),   # c2  T2
    (-1.0, 1.0),   # c3  T3
    (-1.0, 1.0),   # c4  T4
    (-1.0, 1.0),   # c5  T5
]

if use_ray:
    import ray

    @ray.remote(num_cpus=1)
    def fit_exposure_remote(*args):
        return fit_exposure(*args)


def apply_corrections(pars, wl_pipe, flux, x):
    a0, a1, a2, a3, c0, c1, c2, c3, c4, c5 = pars
    wl_corr = wl_pipe + cheby3(x, a0, a1, a2, a3)
    flux_corr = flux / cheby5(x, c0, c1, c2, c3, c4, c5)
    return wl_corr, flux_corr


def load_unc_paths(order):
    # uncertainty files sorted by exposure number (matches get_spectra_list)
    files = list(UNC_DIR.glob(f'b_e*_o0{order}_unc.npy')) or list(UNC_DIR.glob(f'b_e*_o{order}_unc.npy'))
    if not files:
        raise FileNotFoundError(f'no uncertainty files for order {order} in {UNC_DIR}')
    return sorted(files, key=lambda p: int(re.search(r'_e(\d+)_', p.name).group(1)))


def weights_from_sigma(sigma):
    # inverse-variance weights; clips extreme sigma and normalizes mean(w) = 1 so chi^2 stays readable
    sig = np.asarray(sigma, dtype=float)
    sig = np.where(np.isfinite(sig) & (sig > 0), sig, np.nan)
    med = np.nanmedian(sig)
    if not np.isfinite(med) or med <= 0:
        return np.ones(len(sig), dtype=float)
    sig = np.where(np.isfinite(sig), sig, med)
    sig = np.clip(sig, 0.2 * med, 5.0 * med)
    w = 1.0 / sig**2
    return w / np.mean(w)


def chisq_fit(pars, wl_pipe, flux, x, mod_wl, mod_fl, weights):
    # mod_wl must already be in the same (earth) frame as wl_pipe
    wl_corr, flux_corr = apply_corrections(pars, wl_pipe, flux, x)
    fake_mod = interpolate.interp1d(mod_wl, mod_fl, bounds_error=False, fill_value=1.0)(wl_corr)
    return float(np.sum(weights * (flux_corr - fake_mod) ** 2))


def build_oot_template_stellar(round1_paths, oot, rvs):
    # median-combine OOT round-1 spectra in the stellar rest frame (each shifted by -RV)
    wls = []
    fluxes = []
    for exp, path in enumerate(round1_paths):
        if not oot[exp]:
            continue
        spec = np.load(path)
        wls.append(doppler(-rvs[exp], spec[:, 0]))
        fluxes.append(spec[:, 1])

    wls = np.asarray(wls)
    fluxes = np.asarray(fluxes)
    wl_grid = np.nanmedian(wls, axis=0)

    interp_fluxes = np.asarray([
        interpolate.interp1d(wl, flux, kind='cubic', bounds_error=False, fill_value='extrapolate')(wl_grid)
        for wl, flux in zip(wls, fluxes)
    ])
    return wl_grid[TRUNC:-TRUNC], np.median(interp_fluxes, axis=0)[TRUNC:-TRUNC]


def fit_exposure(i, wl_pipe, flux, x, mod_wl_stellar, mod_fl, rv, weights, order, outdir):
    # shift the stellar-frame template into the observer frame for this exposure
    mod_wl = doppler(rv, mod_wl_stellar)

    result = differential_evolution(chisq_fit, BOUNDS, args=(wl_pipe, flux, x, mod_wl, mod_fl, weights))
    np.save(outdir / f'results_blue_r2_order{order}_exp{i}.npy', result.x)

    wl_corr, flux_corr = apply_corrections(result.x, wl_pipe, flux, x)
    np.save(outdir / f'{target_string}_spec_wlsol_round2_order{order}_exp{i}.npy', np.column_stack((wl_corr, flux_corr)))
    return chisq_fit(result.x, wl_pipe, flux, x, mod_wl, mod_fl, weights)


def solve_order(order, spectra_folder, specdir, rawdir):
    phases_full = np.loadtxt(PHASES_FILE)
    _, nexp, exposure_to_full = get_geometry(order, len(phases_full))
    phases = phases_full[exposure_to_full] if exposure_to_full is not None else phases_full
    oot = oot_mask(phases)

    # fall back to the pipeline (round 0) solution if there is no round 1
    for roundno in [1, 0]:
        round1_paths = [spectra_folder / f'{target_string}_spec_wlsol_round{roundno}_order{order}_exp{i}.npy' for i in range(nexp)]
        if all(p.is_file() for p in round1_paths):
            break
    else:
        print(f'order {order}: missing round0/round1 spectra, skipping')
        return

    round2_paths = [spectra_folder / f'{target_string}_spec_wlsol_round2_order{order}_exp{i}.npy' for i in range(nexp)]
    if not force and all(p.is_file() for p in round2_paths):
        print(f'order {order}: all {nexp} round2 files exist, skipping')
        return

    planet = make_planet()
    _, nexp_raw, _, hjds = get_spectra_list(specdir, rawdir, order)
    if nexp_raw != nexp:
        raise RuntimeError(f'order {order}: {nexp} exposures expected, {nexp_raw} raw files found')

    rvs = compute_rvs(planet, hjds)
    mod_wl_stellar, mod_fl = build_oot_template_stellar(round1_paths, oot, rvs)
    if save_template:
        np.save(spectra_folder / f'{target_string}_stellar_spec_transit_order{order}.npy',
                np.column_stack((mod_wl_stellar, mod_fl)))

    unc_paths = load_unc_paths(order)
    if len(unc_paths) != nexp:
        raise RuntimeError(f'order {order}: {len(unc_paths)} uncertainty files for {nexp} exposures')

    ncol = np.load(round1_paths[0]).shape[0] - 2 * TRUNC
    x = scale_col(np.arange(ncol), float(ncol))

    print(f'order {order}: fitting {nexp} exposures ({int(oot.sum())} OOT in template)')
    chisq_vals = []
    for i in range(nexp):
        if round2_paths[i].is_file() and not force:
            continue
        spec = np.load(round1_paths[i])
        weights = weights_from_sigma(np.load(unc_paths[i])[TRUNC:-TRUNC])
        args = (i, spec[TRUNC:-TRUNC, 0], spec[TRUNC:-TRUNC, 1], x, mod_wl_stellar, mod_fl,
                float(rvs[i]), weights, order, spectra_folder)
        if use_ray:
            chisq_vals.append(fit_exposure_remote.remote(*args))
        else:
            chisq_vals.append(fit_exposure(*args))

    if use_ray:
        chisq_vals = ray.get(chisq_vals)

    with open(spectra_folder / 'chisqs_blue_round2.txt', 'a') as f:
        for cv in chisq_vals:
            f.write(f'order{order}\t{cv}\n')


def main():
    spectra_folder = BASE_DIR / SPECTRA_FOLDER_BLUE
    specdir = BASE_DIR / f'{target_string}_spectra_full'
    rawdir = BASE_DIR / RAW_FOLDER

    if use_ray and not ray.is_initialized():
        ray.init(num_cpus=NUM_CPUS, runtime_env={'env_vars': {'PYTHONPATH': os.path.dirname(os.path.abspath(__file__))}})

    for order in orders:
        solve_order(order, spectra_folder, specdir, rawdir)


if __name__ == '__main__':
    main()
