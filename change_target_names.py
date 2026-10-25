# Rename the OBJECT key in the raw FITS files so each spectrum is unique

import re

import numpy as np
from astropy.io import fits

from constants import BASE_DIR, RAW_FOLDER, RED_OBJECT_OFFSET, TARGET

exclude = ['WASP-52', 'LTT9779', 'test_Flat', 'placeholder']  # exclude other targets in same folder


def atof(text):
    try:
        return float(text)
    except ValueError:
        return text


def natural_keys(text):
    return [atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]


def main():
    # renamed files go in a folder named after the target
    outdir = BASE_DIR / TARGET
    outdir.mkdir()

    # red, then blue arm data
    for arm in ['r', 'b']:
        fs = sorted((str(p) for p in (BASE_DIR / RAW_FOLDER).glob(f'{arm}*.fits')), key=natural_keys)
        print(f'Found {len(fs)} files:')
        print(np.array(fs))

        for f in fs:
            data, header = fits.getdata(f, header=True)

            object_name = header['OBJECT']
            if object_name in exclude:
                print(f'Skipping {object_name}')
                continue

            if TARGET in object_name:
                # offset makes red and blue object numbers match
                number = f[-7:-5]
                if arm == 'r' and RED_OBJECT_OFFSET:
                    number = int(number) + RED_OBJECT_OFFSET
                header['OBJECT'] = f'{TARGET}-{number}'

            fits.writeto(outdir / f[-10:], data, header, overwrite=True)
            print(header['OBJECT'])


if __name__ == '__main__':
    main()
