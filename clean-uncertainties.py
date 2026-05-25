import numpy as np
import matplotlib.pyplot as plt

import os
import glob

from scipy.ndimage import median_filter
from scipy import stats

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

target_short = 'w55'

skip = ['r071', 'b100', 'b101', 'b102', 'b103', 'b104', 'b105', 'b106']

os.makedirs(f'{target_short}_uncertainties_cleaned', exist_ok=True)
os.makedirs(f'Plots/{target_short}_uncertainties', exist_ok=True)

plt.style.use('paper.mplstyle')

# def pca_rebuild_numpy(data, n_components):
    
#     U, S, Vt = np.linalg.svd(data, full_matrices=False)
    
#     U_k = U[:, :n_components]
#     S_k = np.diag(S[:n_components])
#     Vt_k = Vt[:n_components, :]
    
#     return U_k @ S_k @ Vt_k

for arm in ['r', 'b']: 

    if arm == 'r': 
        string = 'Red'
    else: 
        string = 'Blue'
    
    all_files = glob.glob(f'{target_short}_uncertainties_full/{arm}_e*_o*_unc.npy')

    orders = [f[-11:-8] for f in all_files]
    orders_unique = sorted(list(set(orders)))
    print(f'Unique orders: {orders_unique}')

    for order in orders_unique: 

        if arm+order in skip: 
            print(f'Skipping order {order}...')
            continue
        
        print(f'Order {order}...')
        fs = glob.glob(f'{target_short}_uncertainties_full/{arm}_e*_o{order}_unc.npy') # eg 090
        fs = sorted(fs)

        plt.figure(figsize=(15,6))
        uncs = []

        for f in fs: 
            unc = 1./np.load(f)
            # print(f'Removing nans: {np.sum(np.isinf(unc))}')
            med = median_filter(unc, size = 50)
            unc[np.isinf(unc)] = med[np.isinf(unc)]

            # print(f'Removing negative: {np.sum(unc <= 0)}')
            unc[unc <= 0] = med[unc <= 0]

            med = median_filter(unc, size = 50)
            mad = stats.median_abs_deviation(unc-med)

            flag = (unc-med) < -5 * mad * 1.4826 # only remove downward outliers
            # print(f'Outliers: {np.sum(flag)}')

            if np.sum(flag) > len(flag) * 0.10 * 10: 
                print('>10% outliers, using median filtered version instead')
                unc = med
            else: 
                unc[flag] = med[flag]
            
            plt.plot(unc)
            uncs.append(unc)

            np.save(f.replace('full', 'cleaned'), unc)

        plt.title(f'{string} arm order {order}')

        unc = np.nanmedian(uncs, axis = 0)
        
        low_iqr = np.quantile(unc,0.5)-np.quantile(unc,0.1)
        up_iqr = np.quantile(unc,0.9)-np.quantile(unc,0.5)
        plt.ylim(np.nanmedian(unc) - 3*low_iqr,np.nanmedian(unc) + 10*up_iqr)
        
        plt.ylabel('Uncertainty')
        plt.xlabel('Pixel #')
        plt.tight_layout()
        plt.savefig(f'Plots/{target_short}_uncertainties/order_{order}.png', facecolor='w', dpi = 200)
        plt.close()

        # smoothed_uncs = pca_rebuild_numpy(uncs, 5)

        # plt.figure(figsize=(15,6))
        # plt.plot(smoothed_uncs.T)
        # plt.ylabel('Uncertainty')
        # plt.xlabel('Pixel #')
        # plt.tight_layout()
        # plt.ylim(np.nanmedian(unc) - 6*low_iqr,np.nanmedian(unc) + 6*up_iqr)
        # plt.savefig(f'Plots/{target_short}_uncertainties/order_{order}_smooth.png', facecolor='w', dpi = 200)
        # plt.close()
