# fit the wavelength solution and continuum of each exposure against a stellar (+ sky) model

import os

import numpy as np
import scipy.ndimage
from scipy import interpolate, stats
from scipy.optimize import differential_evolution

from constants import NUM_CPUS, PHASES_FILE, RVS_FILE, get_geometry, rvs_for_arm, use_ray
from utils import doppler

# order -> (central wavelength [A], order width [A])
# estimated from MIKE red_atlas and blue_atlas files
# https://www.lco.cl/wp-content/uploads/telescopes/magellan/instruments/mike/thar_atlas/red-images/red_atlas.pdf
GRATING = {
    70: (4910, 125), 69: (4980, 145), 68: (5050, 150), 67: (5125.0, 150),
    66: (5200.0, 150), 65: (5287.5, 150), 64: (5362.5, 150), 63: (5450.0, 150),
    62: (5537.5, 150), 61: (5625.0, 160), 60: (5725.0, 160), 59: (5825.0, 160),
    58: (5919.0, 160), 57: (6025.0, 160), 56: (6137.5, 175), 55: (6242.5, 175),
    54: (6362.5, 175), 53: (6487.5, 175), 52: (6603.0, 180), 51: (6725.0, 180),
    50: (6875.0, 180), 49: (7015.0, 180), 48: (7150.0, 200), 47: (7310.0, 200),
    46: (7480.0, 200), 45: (7635.0, 220), 44: (7800.0, 220), 43: (7975.0, 220),
    42: (8175.0, 230), 41: (8375.0, 230), 40: (8585.0, 230), 39: (8825.0, 250),
    38: (9045.0, 250), 37: (9275.0, 250),
    106: (3330, 70), 105: (3360, 70), 104: (3390, 70), 103: (3420, 70),
    102: (3455, 80), 101: (3490, 70), 100: (3530, 70), 99: (3560, 60),
    98: (3590, 70), 97: (3630, 70), 96: (3670, 70), 95: (3710, 70),
    94: (3750, 70), 93: (3790, 70), 92: (3830, 70), 91: (3870, 70),
    90: (3910, 70), 89: (3950, 70), 88: (4000, 90), 87: (4050, 70),
    86: (4090, 70), 85: (4140, 90), 84: (4190, 70), 83: (4240, 90),
    82: (4290, 70), 81: (4340, 90), 80: (4400, 90), 79: (4460, 90),
    78: (4510, 70), 77: (4560, 90), 76: (4620, 90), 75: (4680, 90),
    74: (4750, 110), 73: (4820, 90), 72: (4880, 90), 71: (4950, 110),
}

TRUNC = 10

if use_ray:
    import ray

    @ray.remote(num_cpus=1)
    def solve_exposure_remote(i, datmed, bounds, ncol, shifted_modwl, smooth_model, spec_prefix, order, newdir):
        return solve_exposure(i, datmed, bounds, ncol, shifted_modwl, smooth_model, spec_prefix, order, newdir)


def solve_exposure(i, datmed, bounds, ncol, shifted_modwl, smooth_model, spec_prefix, order, newdir):
    # solves the wavelength solution for a single exposure
    results = differential_evolution(chisq, bounds, args=(datmed, np.arange(ncol), shifted_modwl, smooth_model))
    np.save(f'{newdir}/results_order{order}_exp{i}.npy', results.x)

    xcorr = scale_col(np.arange(ncol), results.x[0])
    datcorr = datmed / cheby5(xcorr, *results.x[5:11])
    wlcorr = cheby3(xcorr, *results.x[1:5])
    np.save(f'{newdir}/{spec_prefix}_order{order}_exp{i}.npy', np.column_stack((wlcorr, datcorr)))

    return chisq(results.x, datmed, np.arange(ncol), shifted_modwl, smooth_model)


def main(exoplanet, order, sky_fac, resolution, specdir, newdir, stellar_model_fn, skyspec_fn, modloc, modbase, roundno):
    # round 1 shifts a stellar model (+ sky) into the observer frame for each exposure
    # round 2 uses the combined stellar spectrum, which is already in the data frame
    needsky = roundno == 1

    speclist, nexp, npix = get_spectra_list(specdir, order)

    allspec = np.zeros([nexp, npix])
    for i in range(nexp):
        allspec[i, :] = np.genfromtxt(f'{specdir}/{speclist[i]}', usecols=(0,), unpack=True)

    # stellar RVs from calculate_phases.py
    _, nexp_arm, exposure_to_full = get_geometry(order, len(np.loadtxt(PHASES_FILE)))
    if nexp_arm != nexp:
        raise RuntimeError(f'order {order}: {nexp} spectra but {nexp_arm} exposures expected')
    rvs = rvs_for_arm(np.load(RVS_FILE), exposure_to_full)

    # truncate order edges
    spectra = allspec[:, TRUNC:-TRUNC]
    nexp, ncol = np.shape(spectra)

    b1, order_lims, mod_lims = grating_bounds(order)

    # variable bounds (a0f continuum scale centered on ~1)
    bounds = [(4050, 4090), (b1[0], b1[1]), (order_lims[1] - order_lims[0] - 40, order_lims[1] - order_lims[0] + 40),
              (-10, 10), (-10, 10), (0.7, 1.3), (-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10)]

    # in the second round, constrain bounds around the median of the best-fit values from round 1
    if roundno == 2:
        results_round1 = [np.load(f'{newdir}/results_order{order}_exp{i}.npy') for i in range(nexp)]
        med_r = np.median(results_round1, axis=0)
        mad_r = stats.median_abs_deviation(results_round1, axis=0)
        # n, a0, a1, a2, a3, a0f, a1f..a5f
        min_half = np.array([5.0, 0.5, 1.0, 0.1, 0.1, 0.05, 0.5, 0.5, 0.5, 0.5, 0.5])

        bounds = []
        for i in range(len(med_r)):
            half = max(float(mad_r[i] * 1.4826 * 3), float(min_half[i]))
            bounds.append((med_r[i] - half, med_r[i] + half))
        print(f'using bounds: {bounds}')

    sky_wl_air, sky_fl, mod_wl_air, mod_fl = get_sky_and_stellar_model(skyspec_fn, stellar_model_fn, mod_lims, modloc, modbase, order, needsky and sky_fac > 0)
    spec_prefix = f'{exoplanet.name}_spec_wlsol_round{roundno}'

    if use_ray and not ray.is_initialized():
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ray.init(num_cpus=NUM_CPUS, runtime_env={'env_vars': {'PYTHONPATH': script_dir}})

    chisq_vals = []
    for i in range(nexp):
        print(f'order {order} wavelength solution {i + 1} of {nexp}')
        datmed = spectra[i, :] / np.median(np.nan_to_num(spectra[i, :]))

        if needsky:
            shifted_modwl, smooth_model = build_stellar_sky_model(
                rvs[i], mod_wl_air, mod_fl, sky_wl_air, sky_fl, sky_fac, order_lims, resolution)
        else:
            shifted_modwl, smooth_model = mod_wl_air, mod_fl

        args = (i, datmed, bounds, ncol, shifted_modwl, smooth_model, spec_prefix, order, newdir)
        if use_ray:
            chisq_vals.append(solve_exposure_remote.remote(*args))
        else:
            chisq_vals.append(solve_exposure(*args))

    if use_ray:
        chisq_vals = ray.get(chisq_vals)

    with open(f'{newdir}/chisqs_round{roundno}.txt', 'a') as f:
        for cv in chisq_vals:
            f.write(f'{cv}\n')


def get_spectra_list(specdir, order):
    # find all text files containing spectra for this order (e.g. b_e039_o37_flux.txt), sorted by exposure number
    speclist = np.array([f for f in os.listdir(specdir) if f'{order}_flux.txt' in f])
    exp_num = np.array([f[3:6] for f in speclist]).astype(int)
    speclist = speclist[np.argsort(exp_num)]

    nexp = len(speclist)
    npix = len(np.genfromtxt(f'{specdir}/{speclist[0]}', usecols=0, unpack=True))
    return speclist, nexp, npix


def build_stellar_sky_model(rv, mod_wl_air, mod_fl, sky_wl_air, sky_fl, sky_fac, order_lims, resolution):
    # stellar model is in stellar rest frame; sky/telluric model is in Earth frame
    shifted_modwl = doppler(rv, mod_wl_air)
    in_order = (shifted_modwl > order_lims[0]) & (shifted_modwl < order_lims[1])
    modmed = mod_fl / np.nanmedian(mod_fl[in_order])
    if sky_fac:
        interpf_sky = interpolate.interp1d(sky_wl_air, sky_fl, kind='cubic', bounds_error=False, fill_value=0)
        modmed = modmed + sky_fac * interpf_sky(shifted_modwl) - sky_fac
    smooth_model = scipy.ndimage.gaussian_filter1d(modmed, resolution)
    return shifted_modwl, smooth_model


def get_sky_and_stellar_model(skyspec_fn, stellar_model_fn, mod_lims, modloc, modbase, order, needsky):
    if stellar_model_fn is None:
        # combined stellar spectrum from combine_spectrum.py
        modarr = np.load(f'{modloc}/{modbase.format(order)}')
        modwl, modspec = modarr[:, 0], modarr[:, 1]
        in_lims = (modwl > mod_lims[0]) & (modwl < mod_lims[1])
        mod_fl = modspec[in_lims]
        mod_wl_air = modwl[in_lims]
    else:
        # model already in air wavelengths
        lambda_star, fl_model = np.loadtxt(stellar_model_fn, usecols=[0, 1], unpack=True)  # wavelength [A], flux [erg/s/cm2/A]
        fl_model = fl_model[np.argsort(lambda_star)]
        lambda_star = np.sort(lambda_star)
        in_lims = (lambda_star > mod_lims[0]) & (lambda_star < mod_lims[1])
        mod_fl = fl_model[in_lims]
        mod_wl_air = lambda_star[in_lims]

    if not needsky:
        return None, None, mod_wl_air, mod_fl

    # read in sky spectrum, cut to size (TelFit already in air wavelengths)
    sky_wl_air, sky_fl = np.loadtxt(skyspec_fn, unpack=True)
    in_lims = (sky_wl_air > mod_lims[0]) & (sky_wl_air < mod_lims[1])
    return sky_wl_air[in_lims], sky_fl[in_lims], mod_wl_air, mod_fl


def grating_bounds(order):
    mid, width = GRATING[order]
    b1 = [mid - 5, mid + 5]
    order_lims = [mid - width / 2, mid + width / 2]
    mod_lims = [mid - 6 * width, mid + 6 * width]
    return b1, order_lims, mod_lims


def scale_col(col, n):
    return (col - n / 2) / n


def cheby3(x, a0, a1, a2, a3):
    return a0 + a1 * x + a2 * (2 * x**2 - 1) + a3 * (4 * x**3 - 3 * x)


def cheby5(x, a0, a1, a2, a3, a4, a5):
    return (a0 + a1 * x + a2 * (2 * x**2 - 1) + a3 * (4 * x**3 - 3 * x)
            + a4 * (8 * x**4 - 8 * x**2 + 1) + a5 * (16 * x**5 - 20 * x**3 + 5 * x))


def chisq(pars, data, col, modwl, mod):
    n, a0, a1, a2, a3, a0f, a1f, a2f, a3f, a4f, a5f = pars
    x = scale_col(col, n)
    guess_wls = cheby3(x, a0, a1, a2, a3)
    guess_cont = cheby5(x, a0f, a1f, a2f, a3f, a4f, a5f)
    interpf = interpolate.interp1d(modwl, mod, bounds_error=False, fill_value=1)
    fake_mod = interpf(guess_wls)
    return np.sum((data / guess_cont - fake_mod) ** 2)
