# Compute the orbital phase of each exposure from the raw FITS headers

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time

from constants import BASE_DIR, MID_TRANSIT, P_DAYS, PHASES_FILE, RAW_FOLDER, TARGET

arm = 'b'  # use the arm with no missing exposures


def main():
    bjds = []
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

    bjds = np.array(bjds)
    print(f'Found {len(bjds)} exposures of {TARGET}')

    # MID_TRANSIT is BJD_TDB - 2450000
    phases = (bjds - (MID_TRANSIT + 2450000.0)) / P_DAYS
    np.savetxt(PHASES_FILE, phases)
    print(f'Wrote {PHASES_FILE}')


if __name__ == '__main__':
    main()
