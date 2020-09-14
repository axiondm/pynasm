#!/usr/bin/python

import math

import numpy as np
import scipy as sp
import scipy.constants

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class ns:
    def __init__(
            self,
            axis=np.array((0, 0, 1)),
            radius=1e5,  #meters
            dipole_moment=1e13 * np.array((0.1, 0, 1)),
            quad_moment=1e12 * np.array((1, 0, 0)),
            ns_day=0.1  #seconds
    ):

        self.axis = np.asarray(axis)
        self.radius = radius
        self.dipole_moment = np.asarray(dipole_moment)
        self.quad_moment = np.asarray(quad_moment)
        self.ns_day = ns_day

    def mag_field(self, time, pos):
        pos = np.asarray(pos)
        # first compute the current moments
        rot_matrix = rotation_matrix(self.axis, 2 * np.pi * time / self.ns_day)
        dmoment = np.dot(rot_matrix, self.dipole_moment)
        qmoment = np.dot(rot_matrix, self.quad_moment)

        # for a array of 3d vectors, this finds the norm of each
        rnorm = np.sqrt(np.einsum('ij,ij->i', pos, pos))
        # next compute field from dipole moment
        B_dip = ((sp.constants.mu_0 / (4 * np.pi))
                * ((3 * pos * (np.einsum('ij,j->i', pos, dmoment)[:, np.newaxis] /
                    (rnorm**5)[:, np.newaxis]))
                     - (dmoment[:, np.newaxis] / rnorm**3).T))
        return B_dip
        # and the quadrupole field


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])



