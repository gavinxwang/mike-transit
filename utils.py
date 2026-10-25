# Shared helper functions

C = 299792458  # m/s


def doppler(rv, wl):
    return (wl * rv) / C + wl
