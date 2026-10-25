# Target settings, folder names and shared exposure-geometry helpers

import os
from pathlib import Path

import numpy as np

# run every script from the data directory
BASE_DIR = Path.cwd()

# allow `TARGET=WASP-55 python script.py` without editing this file
TARGET = os.environ.get('TARGET', 'WASP-107')

PIPELINE_FOLDER = 'run_carpy'

use_ray = True
NUM_CPUS = 8

# Magellan / Las Campanas
LCO_LAT, LCO_LON, LCO_ALT = -29.01416667, -70.6925, 2282.0

FEATURES = [
    'Halpha', 'Hbeta', 'Hgamma',
    'CaII_1', 'CaII_2', 'CaII_3',
    'Ca_H', 'Ca_K',
    'NaD1_o58', 'NaD2_o58',
    'HeI_o58', 'HeI_o59',
    'MgI_1', 'MgI_2', 'MgI_3',
    'FeI_1', 'FeI_2', 'FeI_3',
]

RED_ORDERS = list(range(40, 71))
BLUE_ORDERS = list(range(71, 107))

# half-width of the planet-frame velocity window for transmission spectra
VELOCITY_BOUND = 400.0  # km/s

# feature -> (echelle order, rest wavelength in air [A])
FEATURE_MAP = {
    'Halpha': (52, 6562.801),
    'Hbeta': (72, 4861.35),
    'Hgamma': (81, 4340.472),
    'Hdelta': (86, 4101.734),
    'Hepsilon': (88, 3970.075),
    'CaII_1': (40, 8498.023),
    'CaII_2': (40, 8542.091),
    'CaII_3': (40, 8662.141),
    'Ca_H': (89, 3968.47),
    'Ca_K': (89, 3933.66),
    'Ca_H_o88': (88, 3968.47),
    'Ca_K_o90': (90, 3933.66),
    'NaD1_o58': (58, 5889.950),
    'NaD1_o59': (59, 5889.950),
    'NaD2_o58': (58, 5895.924),
    'NaD2_o59': (59, 5895.924),
    'KI_1': (45, 7664.94),
    'KI_2': (45, 7698.97),
    'HeI_o58': (58, 5875.615),
    'HeI_o59': (59, 5875.615),
    'MgI_1': (66, 5183.60),
    'MgI_2': (66, 5172.68),
    'MgI_3': (66, 5167.32),
    'FeI_1': (87, 4045.82),
    'FeI_2': (80, 4383.56),
    'FeI_3': (80, 4404.8),
}

if TARGET == 'WASP-107':
    target_string = 'wasp107b'
    TARGET_LABEL = 'WASP-107 b'
    RAW_FOLDER = 'ut240502_03'  # contains original FITS files

    SPECTRA_FOLDER_RED = 'w107_results_spectra_2025-08-08'
    SPECTRA_FOLDER_BLUE = 'w107_results_spectra_2026-05-25'

    PHASES_FILE = BASE_DIR / 'wasp107_mike_phases.txt'

    # stellar model for round 1 of the wavelength solution
    STELLAR_MODEL_FILE = BASE_DIR / 'wavesol' / 'lte043-0.0-0.0a+0.0.BT-Settl.7'
    # sky model for round 1; None uses the TelFit models from get_tellurics.py
    SKY_FILE = None

    # folder with CLV (Fin/Fout - 1) model arrays
    CLV_PACK = 'W107-pack'

    # missing exposure per arm
    MISSING_SLOT_RED = 3
    MISSING_SLOT_BLUE = None


    # offset added to red-arm object numbers so they match the blue arm
    RED_OBJECT_OFFSET = 0

    # host star
    RA_DEC = '12 33 32.8441110816 -10 08 46.225572324'
    SYS_RV = 13667.0  # m/s
    T0 = 2459958.747244  # BJD
    K_STAR = 13.74  # stellar RV semi-amplitude, m/s

    # transit contact times (BJD_TDB - 2450000)
    START_TRANSIT = 10433.5736
    END_TRANSIT = 10433.6878
    MID_TRANSIT = 10433.6307

    # orbital period and planet RV semi-amplitude
    P_DAYS = 5.7214876
    K_MS = 105.149627353 * 1000.0  # m/s

elif TARGET == 'WASP-55':
    target_string = 'wasp55b'
    TARGET_LABEL = 'WASP-55 b'
    RAW_FOLDER = 'ut250628_29'

    SPECTRA_FOLDER_RED = 'w55_results_spectra_2025-07-09'
    SPECTRA_FOLDER_BLUE = 'w55_results_spectra_2026-05-25'

    PHASES_FILE = BASE_DIR / 'wasp55_mike_phases.txt'

    STELLAR_MODEL_FILE = BASE_DIR / 'wavesol' / 'lte060-4.5-0.0a+0.0.BT-NextGen.7'
    SKY_FILE = BASE_DIR / 'wavesol' / 'flat_sky_angstrom.txt'

    CLV_PACK = 'W55-pack'

    MISSING_SLOT_RED = None
    MISSING_SLOT_BLUE = 17

    RED_OBJECT_OFFSET = 5

    RA_DEC = '13 35 01.9537865424 -17 30 12.527623872'
    SYS_RV = -4093.0
    T0 = 2457207.132185
    K_STAR = 68.7

    START_TRANSIT = 10855.4785
    END_TRANSIT = 10855.6255
    MID_TRANSIT = 10855.5520

    P_DAYS = 4.46563013
    K_MS = 131640.34077

# written by calculate_phases.py, one value per exposure
RVS_FILE = BASE_DIR / f'{target_string}_rvs.npy'
AIRMASS_FILE = BASE_DIR / f'{target_string}_airmass.txt'

START_TRANSIT_PHASE = (START_TRANSIT - MID_TRANSIT) / P_DAYS
END_TRANSIT_PHASE = (END_TRANSIT - MID_TRANSIT) / P_DAYS

# post-egress frames excluded from F_OOT when building rivers (flare tail)
N_SKIP_POST_EGRESS = 2


def get_geometry(order, n_full):
    blue = order >= 71
    missing_slot = MISSING_SLOT_BLUE if blue else MISSING_SLOT_RED
    if missing_slot is not None:
        nexp = n_full - 1
        exposure_to_full = np.array(
            [j if j < missing_slot else j + 1 for j in range(nexp)], dtype=int
        )
    else:
        nexp = n_full
        exposure_to_full = None
    return blue, nexp, exposure_to_full


def rvs_for_arm(rvs_full, exposure_to_full):
    '''Per-arm RVs aligned with exposure index j (same order as phases_arm).'''
    rvs_full = np.asarray(rvs_full)
    if exposure_to_full is not None:
        return rvs_full[exposure_to_full]
    return rvs_full


def in_transit_mask(phases_arm):
    '''True for exposures inside transit (orbital phase vs contact phases).'''
    return (phases_arm >= START_TRANSIT_PHASE) & (phases_arm <= END_TRANSIT_PHASE)


def oot_mask(phases_arm):
    '''True for out-of-transit exposures.'''
    return (phases_arm < START_TRANSIT_PHASE) | (phases_arm > END_TRANSIT_PHASE)


def baseline_oot_mask(phases_arm, n_skip_post=N_SKIP_POST_EGRESS):
    '''
    OOT exposures for river normalization / transmission baseline.

    All OOT frames except the first ``n_skip_post`` post-egress exposures
    (still in-transit for normalization: they divide by F_OOT but do not
    enter the median that defines F_OOT).
    '''
    phases_arm = np.asarray(phases_arm)
    mask = oot_mask(phases_arm)
    post_ix = np.where(phases_arm > END_TRANSIT_PHASE)[0]
    mask[post_ix[:n_skip_post]] = False
    return mask
