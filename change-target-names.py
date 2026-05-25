# renames "OBJECT" key in FITS files to be unique for each spectrum

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

import glob
import re
import os

folder = "ut250628_29" # contains original FITS files
target = "WASP-55" # name of target, and name of folder with new FITS files

exclude = ["WASP-52", "LTT9779", "test_Flat", "placeholder"] # exclude other targets in same folder

os.mkdir(f"{target}")

def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

# red, then blue arm data
for string in ['r', 'b']: 
    fs = glob.glob(f"{folder}/{string}*.fits")

    fs.sort(key=natural_keys)

    print(f'Found {len(fs)} files:')
    print(np.array(fs))

    for i in range(len(fs)): 
        data, header = fits.getdata(fs[i], header = True)
        
        if header["OBJECT"] in exclude: 
            print(f'Skipping {header["OBJECT"]}')
            continue

        if target in header["OBJECT"]: 

            header["OBJECT"] = f'{target}-{fs[i][-7:-5]}'

            # custom offset to make blue/red match
            if target == "WASP-55" and string == 'r': 
                header["OBJECT"] = f'{target}-{int(fs[i][-7:-5])+5}'

        fits.writeto(f'{target}/{fs[i][-10:]}', data, header, overwrite=True)
        print(header["OBJECT"])
