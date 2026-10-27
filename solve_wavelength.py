# run the wavelength solver for each order (round 1: stellar model, round 2: combined stellar spectrum)

import mike_wavelength_solver
from constants import (
    BASE_DIR, SKY_FILE, SPECTRA_FOLDER_BLUE, SPECTRA_FOLDER_RED, STELLAR_MODEL_FILE, target_string,
)
from planet import make_planet

roundno = 1
orders = range(40, 71)

resolution = 1  # gaussian smoothing of the model, pixels


def main():
    planet = make_planet()
    specdir = BASE_DIR / f'{target_string}_spectra_full'

    for order in orders:
        blue = order >= 71
        newdir = BASE_DIR / (SPECTRA_FOLDER_BLUE if blue else SPECTRA_FOLDER_RED)
        newdir.mkdir(parents=True, exist_ok=True)

        # how much sky to add to the stellar model; no strong tellurics in blue arm
        sky_fac = 0 if blue else 1

        if roundno == 2:
            stellar_model_fn = None
            skyspec_fn = None
        else:
            stellar_model_fn = STELLAR_MODEL_FILE
            skyspec_fn = SKY_FILE or BASE_DIR / f'{target_string}_tellurics' / f'telluric_model_order{order}.txt'

        mike_wavelength_solver.main(
            exoplanet=planet, order=order, sky_fac=sky_fac, resolution=resolution,
            specdir=specdir, newdir=newdir, stellar_model_fn=stellar_model_fn, skyspec_fn=skyspec_fn,
            modloc=newdir, modbase=f'{target_string}_stellar_spec_transit_order{{0}}.npy', roundno=roundno,
        )


if __name__ == '__main__':
    main()
