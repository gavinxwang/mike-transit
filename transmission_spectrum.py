# Shift each river to the planet rest frame and compute the transmission spectrum: median(in-transit) / median(oot) - 1

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from constants import (
    BASE_DIR, END_TRANSIT_PHASE, FEATURE_MAP, FEATURES, K_MS, PHASES_FILE, RVS_FILE,
    START_TRANSIT_PHASE, TARGET_LABEL, VELOCITY_BOUND, get_geometry, in_transit_mask, oot_mask,
    rvs_for_arm, target_string,
)
from utils import doppler

WLSOL_ROUND = 2

C = 299792.458  # km/s
C_MS = C * 1000.0

# shared river color scale (matches generate-river.py)
RIVER_VMIN = -0.03
RIVER_VMAX = 0.03

# these features get a wider +/-5% color scale
WIDE_CLIM_FEATURES = {'NaD1_o59', 'NaD2_o59', 'MgI_1', 'MgI_2', 'MgI_3', 'FeI_1', 'FeI_2', 'FeI_3', 'Hbeta', 'Hgamma'}
CA_HK_FEATURES = {'Ca_H', 'Ca_K', 'Ca_H_o88', 'Ca_K_o90'}


def planet_rv_ms(phases):
    return K_MS * np.sin(2.0 * np.pi * phases)


def wavelength_to_velocity_km_s(wl, wl0):
    return C * (wl / wl0 - 1.0)


def map_to_full_phases(data, n_full, exposure_to_full):
    mapped = np.full((n_full, data.shape[1]), np.nan, dtype=float)
    rows = exposure_to_full if exposure_to_full is not None else np.arange(len(data))
    mapped[rows] = data
    return mapped


def to_planet_frame(river, wl_grid, phases, rvs_arm):
    # shift normalized spectra from earth frame to planet rest frame
    nexp, _ = river.shape
    planet_data = np.zeros_like(river)

    for j in range(nexp):
        v_total_ms = planet_rv_ms(phases[j]) + rvs_arm[j]
        wl_src = doppler(-v_total_ms, wl_grid)
        order_idx = np.argsort(wl_src)
        xs, ys = wl_src[order_idx], river[j][order_idx]
        ok = np.isfinite(xs) & np.isfinite(ys)
        if ok.sum() < 5:
            planet_data[j] = np.nan
            continue
        interp = interpolate.interp1d(
            xs[ok], ys[ok], kind='cubic', bounds_error=False, fill_value=0.0,
        )
        planet_data[j] = interp(wl_grid)

    return planet_data


def compute_transmission(planet_data, phases):
    in_mask = in_transit_mask(phases)
    out_mask = oot_mask(phases)

    if not np.any(in_mask) or not np.any(out_mask):
        return np.full(planet_data.shape[1], np.nan)

    in_med = np.nanmedian(planet_data[in_mask] + 1.0, axis=0)
    oot_med = np.nanmedian(planet_data[out_mask] + 1.0, axis=0)

    return in_med / np.maximum(np.abs(oot_med), 1e-12) - 1.0


def compute_transmission_errors(errors_planet, phases, transmission):
    # uncertainty on T = Z - 1, Z = F_in / F_oot, from pipeline per-exposure errors:
    # sigma_Z / Z = sqrt((f_in/sqrt(n_in))^2 + (f_oot/sqrt(n_oot))^2)
    in_err = errors_planet[in_transit_mask(phases)]
    oot_err = errors_planet[oot_mask(phases)]

    x_term = np.nanmedian(in_err, axis=0) / np.sqrt(max(in_err.shape[0], 1))
    y_term = np.nanmedian(oot_err, axis=0) / np.sqrt(max(oot_err.shape[0], 1))
    frac_sigma = np.sqrt(x_term ** 2 + y_term ** 2)

    return np.maximum(np.abs(1.0 + transmission), 1e-6) * frac_sigma


def planet_frame_trace(phases, rvs_vals, rest_wl):
    # earth-frame wavelength of the planet rest frame (a curve per exposure)
    v_total_ms = planet_rv_ms(phases) + rvs_vals
    return rest_wl * (1.0 + v_total_ms / C_MS)


def main():
    plt.style.use('paper.mplstyle')
    plot_dir = BASE_DIR / 'Plots' / f'{target_string}_transmission'
    out_dir = BASE_DIR / 'Outputs' / f'{target_string}_transmission'
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    rvs = np.load(RVS_FILE)
    phases_full = np.loadtxt(PHASES_FILE)
    n_full = len(phases_full)

    for feature in FEATURES:
        print(feature)

        order, rest_wl = FEATURE_MAP[feature]
        _, nexp, exposure_to_full = get_geometry(order, n_full)
        phases_arm = phases_full[exposure_to_full] if exposure_to_full is not None else phases_full
        rvs_arm = rvs_for_arm(rvs, exposure_to_full)

        river_path = BASE_DIR / 'Outputs' / f'{target_string}_rivers' / f'{feature}_order{order}_round{WLSOL_ROUND}.npz'
        if not river_path.is_file():
            print(f'  No river file found for {feature}, skipping')
            continue

        f = np.load(river_path)
        X = f['X']
        Y = f['Y']
        river = f['river']
        errors = f['errors']
        cleaned = f['cleaned']
        eval_clvs = f['clv']
        wl_grid = X[0]

        planet_data = to_planet_frame(river, wl_grid, phases_arm, rvs_arm)
        errors_planet = to_planet_frame(errors, wl_grid, phases_arm, rvs_arm)

        v_km_s = wavelength_to_velocity_km_s(wl_grid, rest_wl)
        trans = compute_transmission(planet_data, phases_arm)
        trans_err = compute_transmission_errors(errors_planet, phases_arm, trans)

        report_mask = np.isfinite(v_km_s) & (np.abs(v_km_s) <= VELOCITY_BOUND)
        s_order = np.argsort(v_km_s[report_mask])
        rep_v = v_km_s[report_mask][s_order]
        rep_t = trans[report_mask][s_order] * 100.0
        rep_e = trans_err[report_mask][s_order] * 100.0

        np.savez(out_dir / f'{feature}_order{order}_round{WLSOL_ROUND}.npz', velocity=rep_v, transmission=rep_t, error=rep_e)

        # planet-rest-frame dashed trace for the top panels (a curve in the earth frame)
        ph_order = np.argsort(phases_arm)
        ph_sorted = phases_arm[ph_order]
        rvs_sorted = rvs_arm[ph_order]

        wl_half = rest_wl * VELOCITY_BOUND / C

        if feature in CA_HK_FEATURES:
            river_vmin, river_vmax = (-0.25, 0.25)
        elif feature in WIDE_CLIM_FEATURES:
            river_vmin, river_vmax = (-0.05, 0.05)
        else:
            river_vmin, river_vmax = (RIVER_VMIN, RIVER_VMAX)

        # 5 stacked panels; dedicated colorbar column so it can span all of them
        fig = plt.figure(figsize=(8, 15), constrained_layout=True)
        gs = fig.add_gridspec(5, 2, width_ratios=[1, 0.04])

        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[1, 0], sharex=ax0, sharey=ax0)
        ax2 = fig.add_subplot(gs[2, 0], sharex=ax0, sharey=ax0)
        ax3 = fig.add_subplot(gs[3, 0], sharey=ax0)
        ax4 = fig.add_subplot(gs[4, 0], sharex=ax3)
        cax = fig.add_subplot(gs[:, 1])

        # top three panels: cleaned data, clv model, clv-subtracted river (earth frame)
        top_axes = [ax0, ax1, ax2]
        top_data = [cleaned, eval_clvs, river]
        top_titles = ['Outliers removed', 'CLV + RM model', 'CLV + RM removed']

        for ax, dat, title in zip(top_axes, top_data, top_titles):
            im_top = ax.pcolormesh(
                X, Y, map_to_full_phases(dat, n_full, exposure_to_full),
                shading='auto', cmap='coolwarm', vmin=river_vmin, vmax=river_vmax, rasterized=True,
            )
            ax.set_ylabel('Orbital phase')
            ax.set_title(title)
            ax.axhline(START_TRANSIT_PHASE, color='k', ls='-', lw=1)
            ax.axhline(END_TRANSIT_PHASE, color='k', ls='-', lw=1)

        # dashed planet-rest-frame trace, extended one step past the panel boundaries
        ymin, ymax = ax0.get_ylim()
        pad = 0.04 * (ymax - ymin)
        ph_line = np.concatenate(([ymin - pad], ph_sorted, [ymax + pad]))
        rvs_lo = rvs_sorted[0] - (rvs_sorted[1] - rvs_sorted[0])
        rvs_hi = rvs_sorted[-1] + (rvs_sorted[-1] - rvs_sorted[-2])
        rvs_line = np.concatenate(([rvs_lo], rvs_sorted, [rvs_hi]))
        lam_line = planet_frame_trace(ph_line, rvs_line, rest_wl)
        for ax in top_axes:
            ax.plot(lam_line, ph_line, c='k', ls='--', lw=1.5)
        ax0.set_ylim(ymin, ymax)
        ax0.set_xlim(rest_wl - wl_half, rest_wl + wl_half)

        # fourth panel: river in the planet rest frame, in velocity space
        Xv, Yv = np.meshgrid(v_km_s, phases_full)
        ax3.pcolormesh(
            Xv, Yv, map_to_full_phases(planet_data, n_full, exposure_to_full),
            shading='auto', cmap='coolwarm', vmin=river_vmin, vmax=river_vmax, rasterized=True,
        )
        ax3.axvline(0.0, c='k', ls='--', lw=1.5)
        ax3.axhline(START_TRANSIT_PHASE, color='k', ls='-', lw=1)
        ax3.axhline(END_TRANSIT_PHASE, color='k', ls='-', lw=1)
        ax3.set_ylabel('Orbital phase')
        ax3.set_title('Planet rest frame')
        ax3.set_xlim(-VELOCITY_BOUND, VELOCITY_BOUND)

        # fifth panel: transmission spectrum (percent) in velocity
        ax4.fill_between(rep_v, rep_t - rep_e, rep_t + rep_e, color='0.7', alpha=0.6, lw=0)
        ax4.plot(rep_v, rep_t, c='k', ls='-', lw=1)
        ax4.axvline(0.0, c='k', ls='--', lw=1)
        ax4.set_ylabel(r'$F_{\rm in}/F_{\rm out} - 1$ (%)')

        vals = rep_t[np.isfinite(rep_t)]
        if vals.size:
            lim = np.nanmax(np.abs(vals))
            ax4.set_ylim(-1.1 * lim, 1.1 * lim)

        ax0.tick_params(labelbottom=False)
        ax1.tick_params(labelbottom=False)
        ax3.tick_params(labelbottom=False)
        ax2.set_xlabel(r'Earth frame wavelength ($\AA$)')
        ax4.set_xlabel('Planet frame velocity (km/s)')

        fig.colorbar(im_top, cax=cax, label=r'$F_{\rm in}/F_{\rm out} - 1$')
        fig.suptitle(f'{TARGET_LABEL} {feature}', fontsize=plt.rcParams['axes.labelsize'])

        outpath = plot_dir / f'{feature}_order{order}_round{WLSOL_ROUND}_transmission.pdf'
        plt.savefig(outpath, dpi=200, bbox_inches='tight')
        plt.close()
        print(f'  Wrote {outpath}')


if __name__ == '__main__':
    main()
