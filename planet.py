from astropy import units as u
from astropy.coordinates import SkyCoord

class exoplanet:
    "exoplanet class, with important parameters"
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
