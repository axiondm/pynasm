import numpy as np
import scipy as sp
from scipy import constants as c
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
import zarr
from matplotlib import pyplot as plt
from multiprocessing import Pool

from .ns_field import ns, norm

def find_conversion_radius(orbit_interpolator, ns, axion_mass, time_bracket):
    """
    We expect orbit_interpolator to be the output of integrate_orbits(),
    a special function that takes a single time or an array of times and spits
    out x,y,z,px,py,pz for each time.

    ns should be a neutron star object. Axion mass should be in eV.
    This function find the place where a axion trajectory encounters the
    conversion surface in the neutron star.
    """

    axion_frequency = axion_mass / c.physical_constants['Planck constant in eV/Hz'][0]

    def to_minimize(t):
        wp = ns.wp(t, orbit_interpolator(t)[:3])
        return axion_frequency - wp

    solution = root_scalar(to_minimize, bracket=time_bracket)
    if type(solution.root) is float:
        return solution.root
    conversion_points = [orbit_interpolator(i) for i in solution.root]
    return conversion_points
    


def interpolated_orbit(orbit):
    def interpolator(t):
        interps = [interp1d(orbit[0], i, fill_value=0, bounds_error=False) for i in orbit[1:]]
        return np.array([i(t) for i in interps])
    return interpolator


def integrate_orbits(path, ns_axis=np.array((0.1, 0, 1)),
                           ns_radius=None, # meters
                           ns_dmoment=1e30 * np.array((1, 0, 1)),
                           ns_qmoment=1e34 * np.array([[0, 0, 1], [0, 1, 0]]),
                           ns_day=1, # seconds
                           axion_mass=1e-6 #eV
                           ):

    results = []
    if ns_radius is None:
        # we need to get the neutron star radius from the log file, which is in
        # km. we want meters, so convert it. I don't care much for this code, it
        # looks fragile. It also works for Sebastians sim, not Davids.
        ns_radius = np.genfromtxt(path + "/general.txt", usecols=0)[1] * 1000
    my_ns = ns(ns_axis, ns_radius, ns_dmoment, ns_qmoment, ns_day)
    orbits = zarr.load(path + "/orbits.zarr")

    # p = Pool()
    # results = p.map(lambda orbit: integrate_single_orbit(orbit, my_ns), orbits.values())

    for orbit in orbits.values():
        ointerp = interpolated_orbit(orbit)
        tbracket = [orbit[0][0], orbit[0][-1]]
        result = find_conversion_radius(ointerp, my_ns, axion_mass, tbracket)
        results.append(result)

    return results



def conversion_probability(B, p, T):
    # first compute the magnetic field in the direction of the axion
    p_hat = p / norm(p)
    Bz = np.dot(B, p_hat)

    # plasma frequency in GHz, field in gauss, period (T) is seconds
    wp = 1.5e2 * np.sqrt((Bz / 1e14) * (1 / T))


