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
        pos = np.asarray(pos)
        # first compute the current moments
        rot_matrix = rotation_matrix(self.axis, 2 * np.pi * time / self.ns_day)
        # rotate the dipole moment (easy)
        dmoment = np.dot(rot_matrix, self.dipole_moment)
        # rotate the two vectors in the quadrupole moment
        qmoment = np.copy(self.quad_moment)
        qmoment[0] = np.dot(rot_matrix, self.quad_moment[0])
        qmoment[1] = np.dot(rot_matrix, self.quad_moment[1])
        # next compute field from dipole moment
        B_dip = dipole(dmoment, pos)
        # and the quadrupole moment
        B_quad = quadrupole(qmoment, pos)
        return B_dip + B_quad
        # return B_dip
        # return B_quad

def dipole(dmoment, pos, dip_location=np.array([0., 0., 0.])):
    if len(pos.shape) == 1:
        pos = pos[np.newaxis]
    pos = pos - dip_location
    rnorm = norm(pos)
    B_dip = ((sp.constants.mu_0 / (4 * np.pi))
            * ((3 * pos * (np.einsum('ij,j->i', pos, dmoment)[:, np.newaxis] /
                (rnorm**5)[:, np.newaxis]))
                 - (dmoment[:, np.newaxis] / rnorm**3).T))
    return B_dip 

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
    naxial = qmoment[0] / qmag

    # these are unit vectors for the four dipole moments we will sum
    A = qmoment[1]
    B = - A
    C = np.cross(naxial, A)
    D = - C
    # now compute the field
    #import ipdb; ipdb.set_trace()
    field = dipole(qmag * A, pos, A * size)
    field += dipole(qmag * B, pos, B * size)
    field += dipole(- qmag * C, pos, C * size)
    field += dipole(- qmag * D, pos, D * size)

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
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])



