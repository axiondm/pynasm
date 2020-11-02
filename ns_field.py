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
            radius=1e4,  #meters
            dipole_moment=1e30 * np.array((1, 0, 1)),
            quad_moment=1e34 * np.array([[0, 0, 1], [0, 1, 0]]),
            ns_day=1  #seconds
    ):

        self.axis = np.asarray(axis)
        self.radius = radius
        self.dipole_moment = np.asarray(dipole_moment)
        # go ahead and make sure our quadrupole moment is of the right form
        self.quad_moment = np.asarray(quad_moment)
        assert np.dot(quad_moment[0], quad_moment[1]) == 0
        quad_moment[1] = quad_moment[1] / norm(quad_moment[1])
        self.ns_day = ns_day

    def mag_field(self, time, pos):
        """
        computes the magnetic field at a given location in time and space.
        pos should be array-like and either [x, y, z] or an array of multiple
        such coordinates.
        """ 
        pos = np.asarray(pos)
        # first compute the current moments
        rot_matrix = rotation_matrix(self.axis, 2 * np.pi * time /
                self.ns_day).T
        # rotate the dipole moment (easy)
        dmoment = np.dot(rot_matrix, self.dipole_moment)
        # rotate the two vectors in the quadrupole moment
        qmoment = np.array([np.dot(rot_matrix, self.quad_moment[0]),
                            np.dot(rot_matrix, self.quad_moment[1])])
 
        # if we are working at a single time slice, we still need a dipole
        # and quadrupole moment for every position
        if rot_matrix.ndim == 3:
            assert len(time) == len(pos)
        elif rot_matrix.ndim == 2:
            dmoment = np.tile(dmoment, (len(pos), 1))
            qmoment = np.array([np.tile(qmoment[0], (len(pos), 1)),
                                np.tile(qmoment[1], (len(pos), 1))])
        # next compute field from dipole moment
        B_dip = dipole(dmoment, pos)
        # and the quadrupole moment
        B_quad = quadrupole(qmoment, pos)
        return B_dip + B_quad

def dipole(dmoment, pos, dip_location=np.array([0., 0., 0.])):
    if len(pos.shape) == 1:
        pos = pos[np.newaxis]
    pos = pos - dip_location
    rnorm = norm(pos)
    B_dip = ((sp.constants.mu_0 / (4 * np.pi))
            * ((3 * pos.T * (np.einsum('ij,ij->i', pos, dmoment) /
                (rnorm**5)))
                 - (dmoment.T / rnorm**3)))
    return B_dip.T

def quadrupole(qmoment, pos, quad_location=np.array([0., 0., 0.]), size=1):
    """qmoment should be 3x2 array, like [[x0, y0, z0], [x1, y1, z1]]
    The first vector is the axial vector of the quadrupole, and sets the
    strangth of the quadrupole. The second vector sets the orthogonal axis on
    which lie the two out-facing dipoles.
    Size sets the distance from the cetner that each dipole is placed. Defaults
    to one and should be left at that without a good reason I guess.
    """
    qmag = norm(qmoment[0])
    # normalized axial vector
    naxial = (qmoment[0].T / qmag).T

    # these are unit vectors for the four dipole moments we will sum
    A = qmoment[1]
    B = - A
    C = np.cross(naxial, A)
    D = - C
    # now compute the field
    #import ipdb; ipdb.set_trace()
    field = dipole((qmag * A.T).T, pos, A * size)
    field += dipole((qmag * B.T).T, pos, B * size)
    field += dipole(-(qmag * C.T).T, pos, C * size)
    field += dipole(-(qmag * D.T).T, pos, D * size)

    return field - quad_location


def norm(vect):
    if len(vect.shape) == 1:
        return np.sqrt(np.sum(vect * vect))
    else:
        return np.sqrt(np.einsum('ij,ij->i', vect, vect))

 
def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b = -axis[0] * np.sin(theta / 2.0)
    c = -axis[1] * np.sin(theta / 2.0)
    d = -axis[2] * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])



