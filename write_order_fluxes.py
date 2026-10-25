# Write the flux, uncertainty and pipeline wavelength solution of each order from the MIKE pipeline products

import re

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from astropy.io import fits
from numpy.polynomial import Chebyshev

from constants import (
    BASE_DIR, PIPELINE_FOLDER, SPECTRA_FOLDER_BLUE, SPECTRA_FOLDER_RED, TARGET, target_string,
)

# continuum normalization of the round-0 spectra
cheb_deg = 5
depth_lim = 0.05
max_iter = 40

SPEC_RE = re.compile(r'spec(\d+)\s*=\s*"([^"]+)"')


def atof(text):
    try:
        return float(text)
    except ValueError:
        return text


def natural_keys(text):
    return [atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]


def first_order(arm):
    return 71 if arm == 'blue' else 37


def exposure_number(path, arm):
    name = str(path)
    return name[-17:-15] if arm == 'blue' else name[-16:-14]


def pipeline_wavelengths(header, npix):
    # parse the linear wavelength solution of each order from the WAT2_ header keys
    wat_text = ''.join(str(header[k]) for k in header if k.startswith('WAT2_'))
    wl_map = {}
    for match in SPEC_RE.finditer(wat_text):
        row_idx = int(match.group(1)) - 1
        body = match.group(2)

        # A regex fix in case wstart and delta strings lose their separating space (e.g., 3418.230.017)
        # 1. Unified fix for the first three tokens (order, 1, 0)
        # This fixes "851 0", "85 10", "8510", and normalizes "85 1 0"
        body = re.sub(r'^\s*(\d{2,3})\s*1\s*0\b', r'\1 1 0 ', body)
        # 2. Fix '0' merged with wstart (e.g., "1 03786.7" -> "1 0 3786.7")
        body = re.sub(r'\b1\s*0\s*(\d{4}\.\d+)', r'1 0 \1', body)
        # 3. Fix wstart merged with delta (e.g., "3418.230.017" -> "3418.23 0.017")
        body = re.sub(r'(\.\d+)(0\.\d+)', r'\1 \2', body)

        tokens = body.split()
        wstart, delta = float(tokens[3]), float(tokens[4])
        wl_map[row_idx] = wstart + np.arange(npix) * delta
    return wl_map


def normalize(spec):
    # iterative chebyshev fit to the continuum of the median-filtered spectrum
    flux = ndi.median_filter(spec, 100)
    xpix = np.linspace(-1.0, 1.0, flux.size)
    continuum_mask = np.isfinite(flux)
    cont = np.full_like(flux, np.nan)
    for _ in range(max_iter):
        use = continuum_mask & np.isfinite(flux)
        if np.count_nonzero(use) < cheb_deg + 3:
            break
        cont = Chebyshev.fit(xpix[use], flux[use], cheb_deg)(xpix)
        eps_c = np.finfo(float).eps * max(1.0, float(np.nanmax(np.abs(cont))))
        depth = np.zeros_like(flux)
        ok = np.isfinite(flux) & np.isfinite(cont) & (np.abs(cont) > eps_c)
        depth[ok] = 1.0 - (flux[ok] / cont[ok])
        new_mask = ok & (depth <= depth_lim)
        if np.array_equal(new_mask, continuum_mask):
            break
        continuum_mask = new_mask

    f_flat = spec / np.where(np.isfinite(cont) & (np.abs(cont) > eps_c), cont, np.nan)
    med_pix = continuum_mask & np.isfinite(f_flat)
    med = np.nanmedian(f_flat[med_pix]) if np.any(med_pix) else np.nanmedian(f_flat)
    if not np.isfinite(med) or med == 0:
        med = 1.0
    return f_flat / med


def main():
    plt.style.use('paper.mplstyle')
    spectra_full = BASE_DIR / f'{target_string}_spectra_full'
    unc_full = BASE_DIR / f'{target_string}_uncertainties_full'
    plot_dir = BASE_DIR / 'Plots'
    for d in [spectra_full, unc_full, plot_dir]:
        d.mkdir(parents=True, exist_ok=True)

    products = BASE_DIR / PIPELINE_FOLDER / f'Final-Products_{target_string}'
    if (BASE_DIR / PIPELINE_FOLDER / 'Final-Products').exists():
        (BASE_DIR / PIPELINE_FOLDER / 'Final-Products').rename(products)

    for arm in ['red', 'blue']:
        spectra_dir = BASE_DIR / (SPECTRA_FOLDER_BLUE if arm == 'blue' else SPECTRA_FOLDER_RED)
        spectra_dir.mkdir(parents=True, exist_ok=True)

        # only the target's exposures (other stars observed the same night can be in the same folder)
        fs = sorted((str(p) for p in products.glob(f'{TARGET.lower()}-*{arm}_multi.fits')), key=natural_keys)
        print(f'{arm} arm: found {len(fs)} exposures')

        n_orders = fits.getdata(fs[0]).shape[1]
        fig, axs = plt.subplots(n_orders, 1, figsize=(10, n_orders * 5))

        for j, f in enumerate(fs):
            expnum = exposure_number(f, arm)
            data, header = fits.getdata(f, header=True)
            wl_map = pipeline_wavelengths(header, data.shape[2])

            for i in range(data.shape[1]):
                order = i + first_order(arm)
                spec = np.asarray(data[6][i], dtype=float)  # relative fluxing (blaze corrected)
                snr = data[3][i]
                wl = wl_map[i]

                np.savetxt(spectra_full / f'{arm[0]}_e0{expnum}_o{order}_flux.txt', data[6][i])
                np.save(unc_full / f'{arm[0]}_e{expnum}_o0{order}_unc.npy', snr)

                f_norm = normalize(spec)
                np.save(spectra_dir / f'{target_string}_spec_wlsol_round0_order{order}_exp{j}.npy', np.column_stack((wl, f_norm)))

                axs[i].plot(wl, f_norm)
                iqr = np.quantile(f_norm, 0.9) - np.quantile(f_norm, 0.1)
                axs[i].set_ylim(np.median(f_norm) - 2 * iqr, np.median(f_norm) + 2 * iqr)
                axs[i].axhline(1, c='k', ls='--')
                axs[i].set_ylabel('Normalized flux')

        axs[-1].set_xlabel(r'Wavelength ($\AA$)')
        plt.savefig(plot_dir / f'{target_string}_{arm}_round0.png', facecolor='w', dpi=200)
        plt.close()


if __name__ == '__main__':
    main()
