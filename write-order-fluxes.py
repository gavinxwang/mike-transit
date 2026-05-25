import glob
import re

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os

def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

folder = "run_carpy" # folder in which mike pipeline was run
target = "WASP-107"
target_short = "w107"
planet = "wasp107b"

os.makedirs(f'{target_short}_spectra_full', exist_ok=True)
os.makedirs(f'{target_short}_uncertainties_full', exist_ok=True)
os.makedirs('Plots', exist_ok=True)

for arm in ['red', 'blue']:
    print(f'{arm} arm')

    fs = glob.glob(f"{folder}/Final-Products_{target}/{target.lower()}*{arm}_multi.fits") # change name of folder before running
    fs.sort(key=natural_keys)
    print(f'Found {len(fs)} exposures')
    
    for f in fs: 
        if arm == 'blue': 
            expnum = f[-17:-15]
        else: 
            expnum = f[-16:-14]
        print(f'expnum: {str(expnum)}')
        
        hdul = fits.open(f)

        spec = []
        d = hdul[0].data

        print(len(d[1])) # no. orders
        
        for i in range(len(d[1])): 
            if arm == 'blue': 
                order = i + 71
            else: 
                order = i + 37
            
            spec = d[6][i] # relative fluxing (blaze corrected)
            unc  = d[3][i] # snr

            # remove outliers and negative values

            if arm == 'blue': 
                string = 'b'
            else: 
                string = 'r'
            
            np.save(f'{target_short}_spectra_full/{string}_e{expnum}_o0{order}_flux.npy', spec)
            np.save(f'{target_short}_uncertainties_full/{string}_e{expnum}_o0{order}_unc.npy', unc)

# blue arm wavelength solution
import scipy.ndimage as ndi
from numpy.polynomial import Chebyshev
plt.style.use('paper.mplstyle')

SPEC_RE = re.compile(r'spec(\d+)\s*=\s*"([^"]+)"')

from datetime import datetime
now = datetime.now()
dir = target_short + '_results_spectra_' + str(now.date()) + '/'
os.makedirs(dir, exist_ok=True)

end = len(d[1]) # should be approx 36

for arm in ['blue']: 
    fs = glob.glob(f"{folder}/Final-Products_{target}/{target.lower()}*{arm}_multi.fits")
    fs.sort(key=natural_keys)
    
    fig, axs = plt.subplots(end, 1, figsize=(10, end*5))
    
    for j, f in enumerate(fs[:]): 
        expnum = f[-17:-15] if arm == 'blue' else f[-16:-14]
        
        hdul = fits.open(f)
        d = hdul[0].data
        header = hdul[0].header
        npix = d.shape[2]
        
        wat_text = "".join(str(header[k]) for k in header if k.startswith("WAT2_"))
        wl_map = {}
        
        for match in SPEC_RE.finditer(wat_text):
            row_idx = int(match.group(1)) - 1
            # A regex fix in case wstart and delta strings lose their separating space (e.g., 3418.230.017)
            body = match.group(2)
            
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
#             print(body)

        for i in range(len(d[1]))[:end]: 
            order = i + 71 if arm == 'blue' else i + 37
            
            spec = d[6][i] 
            wl = wl_map[i]

            flux = np.copy(ndi.median_filter(spec,100))

            cheb_deg = 5
            depth_lim = 0.05
            max_iter = 40
            f = np.asarray(flux, dtype=float)
            xpix = np.linspace(-1.0, 1.0, f.size)
            continuum_mask = np.isfinite(f) # & (f > 0)
            cont = np.full_like(f, np.nan)
            for _ in range(max_iter):
                use = continuum_mask & np.isfinite(f) # & (f > 0)
                if np.count_nonzero(use) < cheb_deg + 3:
                    break
                cfit = Chebyshev.fit(xpix[use], f[use], cheb_deg)
                cont = cfit(xpix)
                eps_c = np.finfo(float).eps * max(1.0, float(np.nanmax(np.abs(cont))))
                depth = np.zeros_like(f)
                ok = np.isfinite(f) & np.isfinite(cont) & (np.abs(cont) > eps_c)
                depth[ok] = 1.0 - (f[ok] / cont[ok])
                new_mask = ok & (depth <= depth_lim)
                if np.array_equal(new_mask, continuum_mask):
                    break
                continuum_mask = new_mask
            
            f_flat = spec / np.where(np.isfinite(cont) & (np.abs(cont) > eps_c), cont, np.nan)
            med_pix = continuum_mask & np.isfinite(f_flat) # & (f_flat > 0)
            med = np.nanmedian(f_flat[med_pix]) if np.any(med_pix) else np.nanmedian(f_flat)
            if not np.isfinite(med) or med == 0:
                med = 1.0
            f_norm = f_flat / med

            axs[i].plot(wl, f_norm)
            iqr = np.quantile(f_norm,0.9)-np.quantile(f_norm,0.1)
            axs[i].set_ylim(np.median(f_norm) - 2*iqr,np.median(f_norm) + 2*iqr)
            axs[i].axhline(1, c='k',ls='--')
            axs[i].set_ylabel('Normalized flux')
            
            g = np.column_stack((wl, f_norm))
            np.save(f'{dir}{planet}_spec_wlsol_round1_order{order}_exp{j}.npy', g)
            
# axs[18].axvline(3933.66)
# axs[18].axvline(3968.47)
# axs[18].set_xlim(3930,3940)
axs[-1].set_xlabel('Wavelength ($\AA$)')
plt.savefig(f'Plots/{target}_blue_round1.png', facecolor='w', dpi = 200)
plt.close()
