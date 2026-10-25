# Build the 2D residual spectra (rivers) for each feature: divide by the OOT stellar spectrum,
# remove outliers, subtract the CLV + RM model and mask strong tellurics

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.ndimage import median_filter

from constants import (
    AIRMASS_FILE, BASE_DIR, CLV_PACK, END_TRANSIT_PHASE, FEATURE_MAP, FEATURES, PHASES_FILE,
    RVS_FILE, SPECTRA_FOLDER_BLUE, SPECTRA_FOLDER_RED, START_TRANSIT_PHASE, baseline_oot_mask,
    get_geometry, rvs_for_arm, target_string,
)
from utils import doppler

WLSOL_ROUND = 2
RIVER_VMIN, RIVER_VMAX = -0.03, 0.03

# TelFit telluric masking: fit H2O/O2 once per order, mask pixels with >10% telluric absorption
TELLURIC_ORDERS = {41, 42, 44, 45, 47, 48, 49, 50, 54, 55}
TELLURIC_DEPTH_LIMIT = 0.9  # mask where transmission T < this (>10% absorption)
TEL_OBS_LAT = -29.01
TEL_OBS_ALT = 2.38
TEL_PRESSURE = 755.0
TEL_RESOLUTION = 65000.0
TEL_CO2 = 1368.5


def feature_window(feature):
    # full-order features named like 'Order53' display the entire order
    if feature.startswith('Order'):
        return int(feature[len('Order'):]), None, None, None
    order, rest_wl = FEATURE_MAP[feature]
    extent = 15 if feature == 'Halpha' else 10
    return order, rest_wl, rest_wl - extent, rest_wl + extent


def zenith_deg(airmass):
    return float(np.degrees(np.arccos(1.0 / airmass)))


def fit_telluric_params(order, nexp, airmass_full):
    '''Fit H2O and O2 on the median stack at median airmass (once per order).'''
    from telfit import DataStructures, Modeler, TelluricFitter

    params_dir = BASE_DIR / f'{target_string}_tellurics'
    params_dir.mkdir(parents=True, exist_ok=True)
    params_path = params_dir / f'fit_params_order{order}_round{WLSOL_ROUND}.npz'
    if params_path.is_file():
        loaded = np.load(params_path, allow_pickle=False)
        return {k: float(loaded[k]) for k in loaded.files}

    angle = zenith_deg(float(np.median(airmass_full)))

    waves_nm = []
    fluxes = []
    for exp in range(nexp):
        spec = np.load(BASE_DIR / SPECTRA_FOLDER_RED / f'{target_string}_spec_wlsol_round{WLSOL_ROUND}_order{order}_exp{exp}.npy')
        waves_nm.append(spec[:, 0] / 10.0)
        fluxes.append(spec[:, 1])

    wl_nm = np.nanmedian(np.array(waves_nm), axis=0)
    flux_med = np.nanmedian(np.array(fluxes), axis=0)
    wavestart = float(np.nanmin(wl_nm) - 1.0)
    waveend = float(np.nanmax(wl_nm) + 1.0)

    data = DataStructures.xypoint(x=wl_nm, y=flux_med, cont=np.ones_like(wl_nm))

    fitter = TelluricFitter()
    fitter.SetObservatory({'latitude': TEL_OBS_LAT, 'altitude': TEL_OBS_ALT})
    fitter.FitVariable({'h2o': 1.0, 'o2': 25e4})
    fitter.AdjustValue({'angle': angle, 'pressure': TEL_PRESSURE, 'wavestart': wavestart, 'waveend': waveend})
    fitter.SetBounds({
        'h2o': [0.0, 10.0],
        'o2': [1e4, 5e5],
        'resolution': [TEL_RESOLUTION - 5000.0, TEL_RESOLUTION + 5000.0],
    })

    initial = Modeler().MakeModel(
        humidity=1.0, o2=25e4, co2=TEL_CO2, angle=angle, pressure=TEL_PRESSURE,
        lat=TEL_OBS_LAT, alt=TEL_OBS_ALT, wavegrid=wl_nm, lowfreq=1e7 / waveend, highfreq=1e7 / wavestart,
    )
    if float(np.nanmin(initial.y)) >= 0.99:
        params = {
            'h2o': 1.0,
            'o2': 25e4,
            'pressure': TEL_PRESSURE,
            'temperature': 283.0,
            'resolution': TEL_RESOLUTION,
            'weak_tellurics': 1.0,
        }
    else:
        fitter.Fit(data=data)
        fit_dict = {name: float(fitter.const_pars[i]) for i, name in enumerate(fitter.parnames)}
        params = {
            'h2o': fit_dict['h2o'],
            'o2': fit_dict['o2'],
            'pressure': fit_dict.get('pressure', TEL_PRESSURE),
            'temperature': fit_dict.get('temperature', 283.0),
            'resolution': fit_dict.get('resolution', TEL_RESOLUTION),
            'weak_tellurics': 0.0,
        }

    np.savez(params_path, **params)
    h2o, o2 = params['h2o'], params['o2']
    print(f'  telluric fit order {order}: h2o={h2o:.3g}, o2={o2:.3g}')
    return params


def telluric_transmission_earth(wl_angstrom, zenith, params):
    '''Return TelFit transmission on the supplied Earth-frame air wavelength grid (A).'''
    from telfit import Modeler

    wl_nm = np.asarray(wl_angstrom, dtype=float) / 10.0
    wavestart = float(np.nanmin(wl_nm) - 0.1)
    waveend = float(np.nanmax(wl_nm) + 0.1)

    model = Modeler().MakeModel(
        pressure=params['pressure'], temperature=params.get('temperature', 283.0), angle=float(zenith),
        humidity=params['h2o'], o2=params['o2'], co2=TEL_CO2, lat=TEL_OBS_LAT, alt=TEL_OBS_ALT,
        wavegrid=wl_nm, lowfreq=1e7 / waveend, highfreq=1e7 / wavestart,
        resolution=params.get('resolution', TEL_RESOLUTION),
    )
    wl_model = np.asarray(model.x, dtype=float) * 10.0
    trans = np.asarray(model.y, dtype=float)
    idx = np.argsort(wl_model)
    return interpolate.interp1d(wl_model[idx], trans[idx], kind='linear', bounds_error=False, fill_value=np.nan)(wl_angstrom)


def compute_telluric_transmission(wls_earth, order, nexp, exposure_to_full, airmass_full):
    '''Build per-exposure TelFit transmission models T(lambda) on each spectrum's Earth-frame grid.'''
    params = fit_telluric_params(order, nexp, airmass_full)
    telluric = np.full_like(wls_earth, np.nan, dtype=float)
    for j in range(nexp):
        full_idx = exposure_to_full[j] if exposure_to_full is not None else j
        telluric[j] = telluric_transmission_earth(wls_earth[j], zenith_deg(airmass_full[full_idx]), params)
    return telluric


def mask_deep_telluric_pixels(*arrays, telluric_trans, depth_limit=TELLURIC_DEPTH_LIMIT):
    '''Return copies with NaN wherever telluric absorption exceeds ``1 - depth_limit``.'''
    deep = np.isfinite(telluric_trans) & (telluric_trans < depth_limit)
    masked = []
    for arr in arrays:
        out = arr.copy()
        out[deep] = np.nan
        masked.append(out)
    return tuple(masked)


def map_to_full_phases(data, n_full, exposure_to_full):
    mapped = np.full((n_full, data.shape[1]), np.nan, dtype=float)
    rows = exposure_to_full if exposure_to_full is not None else np.arange(len(data))
    mapped[rows] = data
    return mapped


def remove_outliers(flux, errors):
    cleaned_flux = flux.copy()
    noutliers = 0
    for j in range(flux.shape[0]):
        fitted_spectrum = median_filter(flux[j], size=100)
        outliers = np.abs(flux[j] - fitted_spectrum) > 4 * errors[j]
        cleaned_flux[j, outliers] = fitted_spectrum[outliers]
        noutliers += np.sum(outliers)
    print(f'{noutliers} outliers removed from a total of {flux.size} pixels')
    return cleaned_flux


def load_errors(order, blue, nexp):
    # pipeline uncertainties and their wavelengths, both in the earth frame
    arm_prefix = 'b' if blue else 'r'
    fs = sorted((BASE_DIR / f'{target_string}_uncertainties_cleaned').glob(f'{arm_prefix}*_o0{order}_unc.npy'))
    wls = sorted((BASE_DIR / f'{target_string}_uncertainties_full').glob(f'{arm_prefix}*_o0{order}_wl.npy'))
    if len(fs) != nexp:
        return None, None
    errors = np.array([np.load(f)[10:-10] for f in fs])
    errors_wls = np.array([np.load(f)[10:-10] for f in wls])
    return errors, errors_wls


def plot_river_panels(X, Y, data, titles, rest_wl, wl_min, wl_max, outpath):
    fig, axes = plt.subplots(len(data), 1, figsize=(10, 8), sharex=True, sharey=True)
    for ax, dat, title in zip(axes, data, titles):
        im = ax.pcolormesh(X, Y, dat, shading='auto', cmap='coolwarm', vmin=RIVER_VMIN, vmax=RIVER_VMAX)
        ax.set_ylabel('Orbital phase')
        ax.set_title(title)
        if rest_wl is not None:
            ax.axvline(rest_wl, c='k', ls='--')
        ax.axhline(START_TRANSIT_PHASE, color='k', ls='-', lw=1)
        ax.axhline(END_TRANSIT_PHASE, color='k', ls='-', lw=1)
    axes[-1].set_xlabel(r'Earth frame wavelength ($\AA$)')
    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.02, label='F_in/F_out - 1')
    if wl_min is not None:
        axes[0].set_xlim(wl_min, wl_max)
    plt.savefig(outpath, facecolor='w', dpi=200)
    plt.close()


def main():
    plt.style.use('paper.mplstyle')
    round = WLSOL_ROUND
    stellar_plot_dir = BASE_DIR / 'Plots' / f'{target_string}_stellar'
    river_plot_dir = BASE_DIR / 'Plots' / f'{target_string}_rivers'
    river_dir = BASE_DIR / 'Outputs' / f'{target_string}_rivers'
    for d in [stellar_plot_dir, river_plot_dir, river_dir]:
        d.mkdir(parents=True, exist_ok=True)

    rvs = np.load(RVS_FILE)
    phases_full = np.loadtxt(PHASES_FILE)
    airmass_full = np.loadtxt(AIRMASS_FILE)
    n_full = len(phases_full)

    for feature in FEATURES:
        print(feature)
        order, rest_wl, wl_min, wl_max = feature_window(feature)
        blue, nexp, exposure_to_full = get_geometry(order, n_full)
        phases_arm = phases_full[exposure_to_full] if exposure_to_full is not None else phases_full
        rvs_arm = rvs_for_arm(rvs, exposure_to_full)

        arm = 'blue' if blue else 'red'
        clv = np.load(BASE_DIR / CLV_PACK / f'Fin_Fout-1_{arm}.npy')
        wl_clv = np.load(BASE_DIR / CLV_PACK / f'wave_{arm}.npy')

        spectra_dir = BASE_DIR / (SPECTRA_FOLDER_BLUE if blue else SPECTRA_FOLDER_RED)
        specs = [np.load(spectra_dir / f'{target_string}_spec_wlsol_round{round}_order{order}_exp{j}.npy') for j in range(nexp)]
        wls = np.array([s[:, 0] for s in specs])
        fluxes = np.array([s[:, 1] for s in specs])

        # blue arm is in the earth frame; red arm is in the stellar frame after the wavelength solution
        if blue:
            wls_earth = np.copy(wls)
            wls_stellar = np.array([doppler(-rvs_arm[j], wls_earth[j]) for j in range(nexp)])
        else:
            wls_stellar = np.copy(wls)
            wls_earth = np.array([doppler(rvs_arm[j], wls_stellar[j]) for j in range(nexp)])

        telluric_native = None
        if not blue and order in TELLURIC_ORDERS:
            print(f'  building telluric transmission models for order {order} (mask T < {TELLURIC_DEPTH_LIMIT})')
            telluric_native = compute_telluric_transmission(wls_earth, order, nexp, exposure_to_full, airmass_full)

        errors_raw, errors_wls = load_errors(order, blue, nexp)
        if errors_raw is None:
            print('  number of error files does not match fluxes, skipping')
            continue

        # interpolate flux and errors to common earth grid
        wl_grid = np.nanmedian(wls_earth, axis=0)
        errors_wl_grid = np.nanmedian(errors_wls, axis=0)
        npix = wl_grid.size

        fluxes_common = np.full((nexp, npix), np.nan, dtype=float)
        errors_common = np.full((nexp, npix), np.nan, dtype=float)
        for j in range(nexp):
            fluxes_common[j] = interpolate.interp1d(
                wls_earth[j], fluxes[j], kind='cubic', bounds_error=False, fill_value='extrapolate')(wl_grid)
            # errors are already in the earth frame, but on the pipeline wavelength grid
            errors_common[j] = interpolate.interp1d(
                errors_wls[j], errors_raw[j], kind='cubic', bounds_error=False, fill_value='extrapolate')(errors_wl_grid)

        eval_tellurics = None
        telluric_deep_mask = None
        if telluric_native is not None:
            eval_tellurics = np.array([
                interpolate.interp1d(wls_earth[j], telluric_native[j], kind='linear', bounds_error=False, fill_value=np.nan)(wl_grid)
                for j in range(nexp)
            ])
            telluric_deep_mask = np.isfinite(eval_tellurics) & (eval_tellurics < TELLURIC_DEPTH_LIMIT)
            print(f'  telluric mask: {int(np.sum(telluric_deep_mask))} pixels with >{(1 - TELLURIC_DEPTH_LIMIT) * 100:.0f}% absorption')

        # build average OOT stellar model in stellar rest frame
        wl_grid_stellar = np.nanmedian(wls_stellar, axis=0)
        dat_stellar = np.array([
            interpolate.interp1d(wls_stellar[j], fluxes[j], kind='cubic', bounds_error=False, fill_value='1')(wl_grid_stellar)
            for j in range(nexp)
        ])

        # OOT baseline: skip first N post-egress (flare tail; see constants.N_SKIP_POST_EGRESS)
        master_stellar_model = np.nanmedian(dat_stellar[baseline_oot_mask(phases_arm)], axis=0)

        plt.figure(figsize=(15, 6))
        plt.plot(np.tile(wl_grid_stellar, (nexp, 1)).T, dat_stellar.T, c='k', ls='-', alpha=0.2)
        plt.xlabel(r'Stellar frame wavelength ($\AA$)')
        plt.ylabel('Normalized flux')
        if rest_wl is not None:
            plt.axvline(rest_wl, c='k', ls='--')
        plt.tight_layout()
        plt.savefig(stellar_plot_dir / f'order{order}_round{round}.png', facecolor='w', dpi=200)
        plt.close()

        # normalize on common grid: shift master stellar model to earth frame for each exposure
        river_normalized = np.full_like(fluxes_common, np.nan)
        for j in range(nexp):
            wl_earth_model = doppler(rvs_arm[j], wl_grid_stellar)
            eval_stellar = interpolate.interp1d(
                wl_earth_model, master_stellar_model, kind='cubic', bounds_error=False, fill_value='extrapolate')(wl_grid)
            denom = np.where(np.isfinite(eval_stellar) & (eval_stellar > 0), eval_stellar, np.nan)
            river_normalized[j] = (fluxes_common[j] / denom) - 1.0

        # use errors in the earth frame to remove outliers
        cleaned_flux = remove_outliers(river_normalized, errors_common)

        X, Y = np.meshgrid(wl_grid, phases_full)

        def full(data):
            return map_to_full_phases(data, n_full, exposure_to_full)

        plot_river_panels(
            X, Y, [full(river_normalized), full(cleaned_flux), full(river_normalized - cleaned_flux)],
            ['Original Normalized Data', 'Outliers Removed', 'Difference'], rest_wl, wl_min, wl_max,
            river_plot_dir / f'{feature}_order{order}_round{round}_outliers.png',
        )

        # subtract clv model
        eval_clvs = np.full_like(cleaned_flux, np.nan)
        for j in range(nexp):
            clv_idx = exposure_to_full[j] if exposure_to_full is not None else j
            wl_earth_clv = doppler(rvs_arm[j], wl_clv)
            eval_clvs[j] = interpolate.interp1d(
                wl_earth_clv, clv[:, clv_idx], kind='cubic', bounds_error=False, fill_value='extrapolate')(wl_grid)
        river_no_clv = cleaned_flux - eval_clvs

        if eval_tellurics is not None:
            river_telluric, cleaned_telluric, raw_telluric, errors_telluric = mask_deep_telluric_pixels(
                river_no_clv, cleaned_flux, river_normalized, errors_common, telluric_trans=eval_tellurics,
            )
        else:
            river_telluric = river_no_clv.copy()
            cleaned_telluric = cleaned_flux.copy()
            raw_telluric = river_normalized.copy()
            errors_telluric = errors_common.copy()

        plot_river_panels(
            X, Y, [full(cleaned_flux), full(eval_clvs), full(river_no_clv)],
            ['Outliers Removed', 'CLV', 'Outliers and CLV Removed'], rest_wl, wl_min, wl_max,
            river_plot_dir / f'{feature}_order{order}_round{round}_subtract_clv.png',
        )

        if eval_tellurics is not None:
            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            im0 = axes[0].pcolormesh(X, Y, full(river_no_clv), shading='auto', cmap='coolwarm', vmin=RIVER_VMIN, vmax=RIVER_VMAX)
            axes[0].set_title('River before telluric mask')
            fig.colorbar(im0, ax=axes[0], label=r'$F_{\rm in}/F_{\rm out} - 1$')
            im1 = axes[1].pcolormesh(X, Y, full(eval_tellurics), shading='auto', cmap='viridis', vmin=0.0, vmax=1.05)
            axes[1].set_title('Telluric transmission T(λ)')
            fig.colorbar(im1, ax=axes[1], label='T(λ)')
            im2 = axes[2].pcolormesh(X, Y, full(river_telluric), shading='auto', cmap='coolwarm', vmin=RIVER_VMIN, vmax=RIVER_VMAX)
            axes[2].set_title(f'River after masking T < {TELLURIC_DEPTH_LIMIT}')
            fig.colorbar(im2, ax=axes[2], label=r'$F_{\rm in}/F_{\rm out} - 1$')
            for ax in axes:
                ax.set_ylabel('Orbital phase')
            axes[-1].set_xlabel(r'Earth frame wavelength ($\AA$)')
            if wl_min is not None:
                axes[0].set_xlim(wl_min, wl_max)
            plt.savefig(river_plot_dir / f'{feature}_order{order}_round{round}_telluric.png', facecolor='w', dpi=200)
            plt.close()

        fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, sharey=True)
        im0 = axes[0].pcolormesh(X, Y, full(river_telluric), shading='auto', cmap='coolwarm', vmin=RIVER_VMIN, vmax=RIVER_VMAX)
        fig.colorbar(im0, ax=axes[0], label='Residual Flux')
        im1 = axes[1].pcolormesh(X, Y, full(errors_telluric), shading='auto', cmap='viridis')
        fig.colorbar(im1, ax=axes[1], label='Uncertainty')
        for ax, title in zip(axes, ['River', 'Errors']):
            ax.set_ylabel('Orbital phase')
            ax.set_title(title)
            if rest_wl is not None:
                ax.axvline(rest_wl, c='k', ls='--')
        axes[-1].set_xlabel(r'Wavelength ($\AA$)')
        if wl_min is not None:
            axes[0].set_xlim(wl_min, wl_max)
        plt.tight_layout()
        plt.savefig(river_plot_dir / f'{feature}_order{order}_round{round}_flux_errors.png', facecolor='w', dpi=200)
        plt.close()

        # in earth frame; also save the outlier-cleaned data and the clv model
        save_kwargs = dict(
            X=X, Y=Y,
            raw=river_normalized,
            river=river_no_clv,
            river_telluric=river_telluric,
            errors=errors_common,
            errors_telluric=errors_telluric,
            cleaned=cleaned_flux,
            cleaned_telluric=cleaned_telluric,
            raw_telluric=raw_telluric,
            clv=eval_clvs,
            telluric_masked=np.array(eval_tellurics is not None, dtype=bool),
            telluric_depth_limit=np.float64(TELLURIC_DEPTH_LIMIT),
        )
        if eval_tellurics is not None:
            save_kwargs['telluric'] = eval_tellurics
            save_kwargs['telluric_deep_mask'] = telluric_deep_mask
        np.savez(river_dir / f'{feature}_order{order}_round{round}.npz', **save_kwargs)


if __name__ == '__main__':
    main()
