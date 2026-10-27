# compute orbital phase, airmass, and stellar RV of each exposure from raw FITS headers

import math

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time
from PyAstronomy import pyasl

from constants import (
    AIRMASS_FILE, BASE_DIR, LCO_ALT, LCO_LAT, LCO_LON, MID_TRANSIT, MISSING_SLOT_BLUE, P_DAYS,
    PHASES_FILE, RAW_FOLDER, RVS_FILE, TARGET,
)
from planet import make_planet

# use arm with no missing exposures
arm = 'b' if MISSING_SLOT_BLUE is None else 'r'


def compute_rvs(exoplanet, jds, bjds):
    # observer-frame stellar RVs in m/s
    # barycentric correction uses UTC JD; reflex uses BJD_TDB
    rvs = np.zeros(len(jds))
    for i, (jd, bjd) in enumerate(zip(jds, bjds)):
        corr, _ = pyasl.helcorr(LCO_LON, LCO_LAT, LCO_ALT, exoplanet.radeg, exoplanet.decdeg, jd)
        if exoplanet.geom == 'eclipse':
            reflex = exoplanet.K * math.sin(2 * math.pi * (exoplanet.t0 + bjd) / exoplanet.P)
        elif exoplanet.geom == 'transit':
            reflex = exoplanet.K * math.sin(2 * math.pi * (exoplanet.t0 - bjd) / exoplanet.P)
        rvs[i] = exoplanet.sysrv + reflex - corr * 1000
    return rvs


def main():
    bjds = []
    jds = []
    airmass = []
    for f in sorted((BASE_DIR / RAW_FOLDER).glob(f'{arm}*.fits')):
        header = fits.getheader(f)
        if not str(header['OBJECT']).startswith(TARGET):
            continue

        # mid-exposure time converted to BJD_TDB
        start_time = Time(header['DATE-OBS'], format='isot', scale='utc')
        mid_time = start_time + (header['EXPTIME'] / 2.0) * u.second
        lco = EarthLocation(
            lat=header['SITELAT'] * u.deg,
            lon=header['SITELONG'] * u.deg,
            height=header['SITEALT'] * u.m,
        )
        target_coord = SkyCoord(header['RA'], header['DEC'], unit=(u.hourangle, u.deg))
        ltt_bary = mid_time.light_travel_time(target_coord, location=lco)
        bjds.append((mid_time.tdb + ltt_bary).jd)
        jds.append(mid_time.jd)
        airmass.append(header['AIRMASS'])

    bjds = np.array(bjds)
    print(f'Found {len(bjds)} exposures of {TARGET}')

    # MID_TRANSIT is BJD_TDB - 2450000
    phases = (bjds - (MID_TRANSIT + 2450000.0)) / P_DAYS
    np.savetxt(PHASES_FILE, phases)
    np.savetxt(AIRMASS_FILE, airmass)
    np.save(RVS_FILE, compute_rvs(make_planet(), np.array(jds), bjds))
    print(f'Wrote {PHASES_FILE}, {AIRMASS_FILE} and {RVS_FILE}')


if __name__ == '__main__':
    main()
