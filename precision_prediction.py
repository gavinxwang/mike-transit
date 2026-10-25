# Predict the 1-sigma precision on line absorption depth achievable with MIKE
#   sigma_A(%) = C * 10^(0.2*(V-11)) * (sigma_v / 5.5 km/s)^p * sqrt(1/tau_in + 1/tau_out) / sqrt(N_tr)

import numpy as np

name    = 'WASP-33 b'
V       = 8.14  # host star V magnitude
tau_in  = 2.85  # in-transit time, hours
tau_out = 2.0   # out-of-transit baseline, hours
N_tr    = 1     # number of transits
fwhm    = 30.5  # expected line FWHM, km/s

# coefficients calibrated on WASP-55 b and WASP-107 b, % * hr^(1/2), at sigma_v = 5.5 km/s
SIGMA_REF = 5.5  # km/s
P_WIDTH = -1.0 / 3.0  # sigma_A ∝ sigma_v^p

COEFFS = {
    'H alpha':       0.619,
    'H beta':        0.659,
    'H gamma':       0.822,
    'Ca II 8498':    0.529,
    'Ca II 8542':    0.492,
    'Ca II 8662':    0.528,
    'Ca II H':       1.974,
    'Ca II K':       2.008,
    'Na I D2 5890':  0.830,
    'Na I D1 5896':  0.801,
    'He I D3 (o58)': 0.794,
    'He I D3 (o59)': 0.951,
    'Mg I 5184':     1.097,
    'Mg I 5173':     1.308,
    'Mg I 5167':     1.481,
    'Fe I 4046':     1.206,
    'Fe I 4384':     0.894,
    'Fe I 4405':     0.825,
}


def main():
    sigma_v = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    brightness = 10 ** (0.2 * (V - 11.0))
    width = (sigma_v / SIGMA_REF) ** P_WIDTH
    time = np.sqrt(1.0 / tau_in + 1.0 / tau_out) / np.sqrt(N_tr)

    print(f'{name}: V = {V}, tau_in = {tau_in} hr, tau_out = {tau_out} hr, N_tr = {N_tr}, '
          f'FWHM = {fwhm} km/s (sigma_v = {sigma_v:.2f} km/s)')
    print('Species'.ljust(16) + 'sigma_A (%)'.rjust(12))
    for species, C in COEFFS.items():
        print(f'{species:<16}{C * brightness * width * time:>12.3f}')


if __name__ == '__main__':
    main()
