# Planetary system parameters used by the wavelength solution

from astropy import units as u
from astropy.coordinates import SkyCoord

from constants import K_STAR, P_DAYS, RA_DEC, SYS_RV, T0, target_string


class exoplanet:
    'exoplanet class, with important parameters'
    def __init__(self, name):
        self.name = name
        self.radec = []
        self.sysrv = []
        self.t0 = []
        self.P = []
        self.K = []
        self.tdur = []
        self.a = []
        self.geom = []
        self.inc = []

    def get_coords_deg(self):
        coords = SkyCoord(self.radec, unit=(u.hourangle, u.deg))
        self.radeg = coords.ra.degree
        self.decdeg = coords.dec.degree


def make_planet():
    planet = exoplanet(target_string)
    planet.radec = RA_DEC
    planet.sysrv = SYS_RV
    planet.t0 = T0
    planet.P = P_DAYS
    planet.K = K_STAR
    planet.geom = 'transit'
    planet.get_coords_deg()
    return planet
