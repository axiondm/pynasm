import numpy as np
import scipy as sp
from scipy import constants as c
import zarr
from matplotlib import pyplot as plt
from multiprocessing import Pool

from .ns_field import ns, norm


def integrate_single_orbit(orbit, my_ns):
    """orbit should ba array-like with orbit[0] time, orbit[1:3] x, y, and z,
    and orbit[4:6] px, py, and pz. ns should be a ns object, imported from
    the ns_field module, representing the neutron star in question."""

    H = my_ns.mag_field(orbit[0], orbit[1:4].T)

    return H


def integrate_orbits(path, ns_axis=np.array((0.1, 0, 1)),
                           ns_radius=None, # meters
                           ns_dmoment=1e30 * np.array((1, 0, 1)),
                           ns_qmoment=1e34 * np.array([[0, 0, 1], [0, 1, 0]]),
                           ns_day=1 # seconds
                           ):

    results = []
    if ns_radius is None:
        # we need to get the neutron star radius from the log file, which is in
        # km. we want meters, so convert it. I don't care much for this code, it
        # looks fragile. It also works for Sebastians sim, not Davids.
        ns_radius = np.genfromtxt(path + "/general.txt", usecols=0)[1] * 1000
    my_ns = ns(ns_axis, ns_radius, ns_dmoment, ns_qmoment, ns_day)
    orbits = zarr.load(path + "/orbits.zarr")

    p = Pool()
    results = p.map(lambda orbit: integrate_single_orbit(orbit, my_ns), orbits.values())


#     for orbit in orbits.values():
#         result = integrate_single_orbit(orbit, my_ns)
#         results.append(result)

    return results



def conversion_probability(B, p, T):
    # first compute the magnetic field in the direction of the axion
    p_hat = p / norm(p)
    Bz = np.dot(B, p_hat)

    # plasma frequency in GHz, field in gauss, period (T) is seconds
    wp = 1.5e2 * np.sqrt((Bz / 1e14) * (1 / T))


