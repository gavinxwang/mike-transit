import numpy as np
import matplotlib.pyplot as plt
import glob
import os

from constants import SPECTRA_FOLDER_BLUE, SPECTRA_FOLDER_RED, BASE_DIR, target_string
phases_full = np.loadtxt(BASE_DIR / "wasp107_mike_phases.txt")

from scipy import interpolate

trunc = 10

# no doppler shifting, keep in frames: earth for blue and stellar for red
os.makedirs(f'Plots/{target_string}_combined', exist_ok=True)

for order in range(40, 107): 
    
    if order >= 71: 
        blue = True
        spectra_folder = SPECTRA_FOLDER_BLUE
    else: 
        blue = False
        spectra_folder = SPECTRA_FOLDER_RED
    
    if not blue: 

        nexp = len(phases_full) - 1

        missing_slot = 3
        exposure_to_full = np.array(
            [j if j < missing_slot else j + 1 for j in range(nexp)], dtype=int
        )
    else: 
        nexp = len(phases_full)

    # check that all files exist, otherwise skip
    fs = glob.glob(f'{spectra_folder}/{target_string}_spec_wlsol_round1_order{order}_exp*.npy')
    if len(fs) != nexp: 
        print(f'missing {nexp - len(fs)} exposures for order {order}, skipping')
        continue

    plt.figure(figsize=(15,6))

    wvls = []
    fluxes = []
    for exp in range(nexp): 

        # omit exposures that are in-transit
        if blue: 
            in_transit = np.arange(8, 18)
        else: 
            in_transit = np.arange(7, 17)

        if exp in in_transit: 
#             print(f'omitting exp {exp}')
            continue
        
        f = np.load(f'{spectra_folder}/{target_string}_spec_wlsol_round1_order{order}_exp{exp}.npy')

        wvls.append(f[:,0])
        fluxes.append(f[:,1])
    
    wvls=np.array(wvls)
    fluxes=np.array(fluxes)
    
    wvl_grid = np.median(wvls, axis = 0)
    wvl_grid = wvls[0]

    interp_fluxes = []
    for i in range(len(wvls)): 
        interp = interpolate.interp1d( wvls[i], fluxes[i], kind='cubic',bounds_error=False, fill_value='extrapolate')
        interp_fluxes.append(interp(wvl_grid))
    interp_fluxes=np.array(interp_fluxes)
    g = np.column_stack((wvl_grid[trunc:-trunc], np.median(interp_fluxes, axis = 0)[trunc:-trunc]))

    plt.figure(figsize=(15,6))
    plt.plot(wvls.T, fluxes.T, 'o-', alpha = 0.3, c='tab:blue')
    plt.plot(g[:,0], g[:,1], '-', lw = 0.5, alpha = 1.0, c='k')
    # plt.plot(g[:,0], g[:,1], 'o-', alpha = 1.0, c='k')
    plt.xlim(g[:,0][0], g[:,0][-1]) # 200
    plt.ylim(0,1.5)
    plt.ylabel('Normalized flux')
    plt.xlabel('Wavelength ($\AA$)')
    
    if blue: 
        plt.title(f'Earth frame, order {order} (blue arm)')
    else: 
        plt.title(f'Stellar frame, order {order} (red arm)')
    plt.tight_layout()
    plt.savefig(f'Plots/{target_string}_combined/combined_order{order}.png', facecolor='w', dpi = 200)
    # plt.show()
    plt.close()

    np.save(f'{spectra_folder}/{target_string}_stellar_spec_transit_order{order}.npy', g)
