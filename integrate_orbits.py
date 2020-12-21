import os
import numpy as np
import scipy as sp
from scipy import constants as c
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
from scipy.misc import derivative
from matplotlib import pyplot as plt
from multiprocessing import Pool

from .ns_field import ns, norm

def find_conversion_radius(orbit_interpolator, ns, axion_mass, time_bracket,
        depth=0):
    """
    We expect orbit_interpolator to be the output of integrate_orbits(),
    a special function that takes a single time or an array of times and spits
    out x,y,z,px,py,pz for each time.

    ns should be a neutron star object. Axion mass should be in eV.
    This function find the place where a axion trajectory encounters the
    conversion surface in the neutron star.
    """
    if depth > 6:
        print("don't go too deep!")
        return None
    axion_frequency = axion_mass / c.physical_constants['Planck constant in eV/Hz'][0]

    def to_minimize(t):
        wp = ns.wp(t, orbit_interpolator(t)[:3].T)
        return axion_frequency - wp

    # first do a sanity check, if our function is always negative or always
    # positive don't try to find roots! remeber that ^ is exclusive or in python
    t = np.linspace(*time_bracket, 1000)
    test_article = to_minimize(t)
    if np.alltrue(test_article < 0) ^ np.alltrue(test_article > 0):
        return None
    else:
        print("passed sanity check")
        # import ipdb; ipdb.set_trace()
    try:
        solution = root_scalar(to_minimize, bracket=time_bracket)
    except ValueError:
        # time to split and recurse!
        print('split and recurse')
        midpoint = t[np.argmax(to_minimize(t))]
        tb1 = [time_bracket[0], midpoint]
        tb2 = [midpoint, time_bracket[1]]
        root1 = find_conversion_radius(orbit_interpolator, ns, axion_mass, tb1,
                depth=depth+1)
        root2 = find_conversion_radius(orbit_interpolator, ns, axion_mass, tb2,
                depth=depth+1)
        return root1 + root2
    print("Found a intersection at {}".format(solution.root))
    return [solution.root]
    


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

    if ns_radius is None:
        # we need to get the neutron star radius from the log file, which is in
        # km. we want meters, so convert it. I don't care much for this code, it
        # looks fragile. It also works for Sebastians sim, not Davids.
        ns_radius = np.genfromtxt(path + "/general.txt", usecols=0)[1] * 1000
    my_ns = ns(ns_axis, ns_radius, ns_dmoment, ns_qmoment, ns_day)
    orbits = sorted(os.listdir(path + "/orbits"))
    # def do_single_orbit(orbit_path):
    #     orbit = np.loadtxt(path + "/orbits/" + orbit_path).T
    #     ointerp = interpolated_orbit(orbit)
    #     tbracket = [orbit[0][0], orbit[0][-1]]
    #     result = find_conversion_radius(ointerp, my_ns, axion_mass, tbracket)
    #     if result is not None:
    #         conversion_points = [ointerp(i) for i in result]
    #         return conversion_points


    # p = Pool(8)
    # results = p.map(do_single_orbit, orbits)

    results = []
    for orbit_path in orbits:
        orbit = np.loadtxt(path + "/orbits/" + orbit_path).T
        ointerp = interpolated_orbit(orbit)
        tbracket = [orbit[0][0], orbit[0][-1]]
        result = find_conversion_radius(ointerp, my_ns, axion_mass, tbracket)
        if result is not None:
            answers = [wp_grad(my_ns, ointerp, i) for i in result]
            results.append(answers)

    return answers


def wp_grad(my_ns, ointerp, t_intersection):
    '''
    Calculate the 2d gradient of the plasma frequency of the neutron star in the
    plane defined by the magnetic field and the axion momentum vectors, at the
    point of intersection between the axion path and the conversion surface
    (where w_p = w_axion)
    '''
    # first, we need to define the cardinal directions in this plane.
    # one is the direction of axion propigation
    x, y, z, px, py, pz = ointerp(t_intersection)
    r = np.array([x, y, z])
    p = np.array([px, py, pz])

    lhat = p / norm(p)
    
    # to get the other, we need the magnetic field at the conversion point
    B = my_ns.mag_field(t_intersection, r)[0]

    # now time for The Gram–Schmidt process (*cries*)
    b = B - p * p.dot(B) / p.dot(p)
    bhat = b / norm(b)

    # now make functions to differentiate
    def wp_l(l):
        return my_ns.wp(t_intersection, r + l * lhat)

    def wp_b(b):
        return my_ns.wp(t_intersection, r + b * bhat)

    # now we can take derivatives
    dwp_dl = derivative(wp_l, 0)
    dwp_db = derivative(wp_b, 0)

    # Now we have all the information we want!
    return r, p, B, dwp_dl, dwp_db




def conversion_probability(B, p, T):
    # first compute the magnetic field in the direction of the axion
    p_hat = p / norm(p)
    Bz = np.dot(B, p_hat)

    # plasma frequency in GHz, field in gauss, period (T) is seconds
    wp = 1.5e2 * np.sqrt((Bz / 1e14) * (1 / T))


