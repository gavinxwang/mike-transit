# convert pipeline S/N to uncertainties and replace bad values and outliers

import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.ndimage import median_filter

from constants import BASE_DIR, target_string

warnings.filterwarnings('ignore', category=RuntimeWarning)

# orders to skip (arm + order)
skip = ['r071', 'b100', 'b101', 'b102', 'b103', 'b104', 'b105', 'b106']

# clean uncertainties
def clean(unc):
    med = median_filter(unc, size=50)
    unc[np.isinf(unc)] = med[np.isinf(unc)]
    unc[unc <= 0] = med[unc <= 0]

    med = median_filter(unc, size=50)
    mad = stats.median_abs_deviation(unc - med)

    # only remove downward outliers
    flag = (unc - med) < -5 * mad * 1.4826
    if np.sum(flag) > len(flag) * 0.10 * 10:
        print('>10% outliers, using median filtered version instead')
        return med
    unc[flag] = med[flag]
    return unc


def main():
    plt.style.use('paper.mplstyle')
    unc_full = BASE_DIR / f'{target_string}_uncertainties_full'
    unc_cleaned = BASE_DIR / f'{target_string}_uncertainties_cleaned'
    plot_dir = BASE_DIR / 'Plots' / f'{target_string}_uncertainties'
    unc_cleaned.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    for arm in ['r', 'b']:
        all_files = unc_full.glob(f'{arm}_e*_o*_unc.npy')
        orders = sorted({f.name[-11:-8] for f in all_files})
        print(f'Unique orders: {orders}')

        for order in orders:
            if arm + order in skip:
                print(f'Skipping order {order}...')
                continue

            print(f'Order {order}...')
            plt.figure(figsize=(15, 6))
            uncs = []
            for f in sorted(unc_full.glob(f'{arm}_e*_o{order}_unc.npy')):
                unc = clean(1. / np.load(f))
                plt.plot(unc)
                uncs.append(unc)
                np.save(unc_cleaned / f.name, unc)

            unc = np.nanmedian(uncs, axis=0)
            low_iqr = np.quantile(unc, 0.5) - np.quantile(unc, 0.1)
            up_iqr = np.quantile(unc, 0.9) - np.quantile(unc, 0.5)
            plt.ylim(np.nanmedian(unc) - 3 * low_iqr, np.nanmedian(unc) + 10 * up_iqr)

            arm_name = 'Red' if arm == 'r' else 'Blue'
            plt.title(f'{arm_name} arm order {order}')
            plt.ylabel('Uncertainty')
            plt.xlabel('Pixel #')
            plt.tight_layout()
            plt.savefig(plot_dir / f'order_{order}.png', facecolor='w', dpi=200)
            plt.close()


if __name__ == '__main__':
    main()
