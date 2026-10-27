# fit a TelFit telluric model to each red order, used as the sky model in round 1 of the wavelength solution

import matplotlib.pyplot as plt
import numpy as np
from telfit import DataStructures, Modeler, TelluricFitter

from constants import AIRMASS_FILE, BASE_DIR, RED_ORDERS, SPECTRA_FOLDER_RED, target_string

resolution = 65000  # red arm
pressure = 755  # hPa
observatory = {'latitude': -29.01, 'altitude': 2.38}  # LCO


def plot_fit(data_x, data_y, model_x, model_y, outpath, data_label, model_label, ylim=None):
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={'height_ratios': [2, 1]})
    ax_top.plot(data_x, data_y, 'k', alpha=0.5, label=data_label)
    ax_top.plot(model_x, model_y, alpha=0.5, label=model_label)
    if ylim is not None:
        ax_top.set_ylim(*ylim)
    ax_top.set_ylabel('Flux')
    ax_top.legend()
    ax_bottom.plot(model_x, data_y - model_y)
    ax_bottom.set_ylabel('Residual')
    ax_bottom.set_xlabel('Wavelength (nm)')
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def main():
    plt.style.use('paper.mplstyle')
    outdir = BASE_DIR / f'{target_string}_tellurics'
    outdir.mkdir(parents=True, exist_ok=True)
    spectra_dir = BASE_DIR / SPECTRA_FOLDER_RED

    fitter = TelluricFitter()
    fitter.SetObservatory(observatory)

    # zenith angle at the median airmass of all exposures
    airmass = np.loadtxt(AIRMASS_FILE)
    angle = np.arccos(1 / np.median(airmass)) * 180 / np.pi

    for order in RED_ORDERS:
        print(f'Fitting order {order}')
        # fit the median spectrum, matching the median airmass used for the zenith angle
        specs = [np.load(f) for f in spectra_dir.glob(f'{target_string}_spec_wlsol_round0_order{order}_exp*.npy')]
        wl = np.median([s[:, 0] for s in specs], axis=0) / 10  # nm
        flux = np.median([s[:, 1] for s in specs], axis=0)

        wavestart = min(wl) - 1
        waveend = max(wl) + 1
        data = DataStructures.xypoint(x=wl, y=flux, cont=np.ones_like(wl))

        fitter.FitVariable({'h2o': 1, 'o2': 25e4})
        fitter.AdjustValue({'angle': angle, 'pressure': pressure, 'wavestart': wavestart, 'waveend': waveend})
        fitter.SetBounds({'h2o': [0.0, 10.0], 'o2': [1e4, 5e5],
                          'resolution': [resolution - 5000., resolution + 5000.]})

        # plot initial guess against data
        model = Modeler().MakeModel(humidity=1, o2=25e4, co2=1368.5, wavegrid=data.x,
                                    lowfreq=1e7 / waveend, highfreq=1e7 / wavestart)
        initial_model = model.y / np.median(model.y)
        plot_fit(data.x, data.y, data.x, initial_model, outdir / f'order{order}_initial_guess.png',
                 'Data', 'Initial guess')

        # if there are no tellurics in this region, skip and directly write the model
        if min(initial_model) >= 0.99:
            print('No tellurics in this region, skipping...')
        else:
            model = fitter.Fit(data=data)
            plot_fit(data.x, data.y, model.x, model.y, outdir / f'order{order}_correction.png',
                     'Data', 'Fitted telluric model', ylim=(0, 1.2))

        np.savetxt(outdir / f'telluric_model_order{order}.txt', np.array([model.x * 10, model.y]).T)


if __name__ == '__main__':
    main()
