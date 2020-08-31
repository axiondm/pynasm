from __future__ import division

import sys
import time
import os
import string
import random
import zarr
from numcodecs import LZ4, Blosc
import numpy as np


# ----------------------------------------------------
# enter parameters
# units: Masses [solar masses]
#        Distances [km]
#        Time      [s]

Nparticles = int(1e4)    # number of particles in axion star

# inital conditions for axion structure
AS_mass = 1e-13    # mass [M_sol]
AS_radius = 8e2    # radius [km]
AS_r0 = np.array([1e12, 6e3, 0.])    # inital position [km]
AS_v0 = np.array([-300., 0., 0.])    # inital velocity [km/s]

# neutron star parameter
NS_mass = 1.4    # mass [M_sol]
NS_radius = 10.    # radius [km]

# ----------------------------------------------------
# constants
G_N = 1.325e11    # Newton constant in km^3/Msol/s^2

# compute gravitational parameter
mu = NS_mass * G_N
# calculate velocity dispersion of axion star
AS_sigmav = np.sqrt(G_N * AS_mass / AS_radius)
# AS_sigmav = 1.151086e-3
# calculate Roche disruption radius
R_dis = AS_radius * (2. * NS_mass / AS_mass)**(1. / 3.)    # [km]

# some parameters for the code and the output
# radius [km] at which partices leaving the neutron star are dropped
# from the calculation
Rdrop = 1e4
Rsave = 1e3    # radius [km] within in which orbits are written to file
nwrite = 100    # number of steps in output to skip when writing to file
mem_size = 1.    # target size of memory [GB] the calculation fills

# ----------------------------------------------------
# functions


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def update_r_v(rx, ry, rz, vx, vy, vz, mu, NSR, rprecision=1e-3, dtmax=1e25):
    """ returns updated r and v and dt for particles """
    r = np.sqrt(rx**2. + ry**2. + rz**2.)
    v = np.sqrt(vx**2. + vy**2. + vz**2.)
    dt = r / v * rprecision
    if np.isscalar(dt):
        dt = np.min([dt, dtmax])
    else:
        np.minimum(dt, np.full(dt.shape, dtmax))
    # calculate the acceleration
    r[np.where(
        r < NSR
    )] = NSR    # soften acceleration inside the neutron star assuming that it has uniform density
    ax = -mu * rx / r**3.
    ay = -mu * ry / r**3.
    az = -mu * rz / r**3.
    # update velocity and position
    out_rx = rx + vx * dt + .5 * ax * dt**2.
    out_ry = ry + vy * dt + .5 * ay * dt**2.
    out_rz = rz + vz * dt + .5 * az * dt**2.
    out_vx = vx + ax * dt
    out_vy = vy + ay * dt
    out_vz = vz + az * dt
    return out_rx, out_ry, out_rz, out_vx, out_vy, out_vz, dt


def mk_axstar(x0, y0, z0, vx0, vy0, vz0, Np):
    """ returns Np long list of particles centered around x0, v0
       gaussian distribution for positions with 90% of mass in AS_radius
       maxwellian velocity profile truncated at the escape velocity """
    Np = int(Np)
    xout = np.zeros(Np)
    yout = np.zeros(Np)
    zout = np.zeros(Np)
    vxout = np.zeros(Np)
    vyout = np.zeros(Np)
    vzout = np.zeros(Np)
    # make random variables for v
    vesc = np.sqrt(2. * mu / AS_radius)
    v_vec = np.linspace(0, vesc, int(1e6))
    v_pdf = v_vec**2. * np.exp(-v_vec**2. / (2. * AS_sigmav**2.))
    v_cT = np.random.rand(Np) * 2. - 1.
    v_sT = np.sin(np.arccos(v_cT))
    v_phi = np.random.rand(Np) * 2. * np.pi
    v_sphi = np.sin(v_phi)
    v_cphi = np.cos(v_phi)
    v_v = np.random.choice(v_vec, p=v_pdf / np.sum(v_pdf), size=Np)
    # make random variabls for x
    r_vec = np.linspace(0, 3. * AS_radius, int(3e6))
    r_pdf = r_vec**2 * np.exp(
        -r_vec**2. / (2. * (AS_radius / 2.5)**2.)
    )    # includes fudge factor to define AS_radius as R_90
    x_cT = np.random.rand(Np) * 2. - 1.
    x_sT = np.sin(np.arccos(x_cT))
    x_phi = np.random.rand(Np) * 2. * np.pi
    x_sphi = np.sin(x_phi)
    x_cphi = np.cos(x_phi)
    x_r = np.random.choice(r_vec, p=r_pdf / np.sum(r_pdf), size=Np)
    for n in range(Np):
        xout[n] = x0 + x_r[n] * x_sT[n] * x_cphi[n]
        yout[n] = y0 + x_r[n] * x_sT[n] * x_sT[n] * x_sphi[n]
        zout[n] = z0 + x_r[n] * x_sT[n] * x_cT[n]
        vxout[n] = vx0 + v_v[n] * v_sT[n] * v_cphi[n]
        vyout[n] = vy0 + v_v[n] * v_sT[n] * v_sphi[n]
        vzout[n] = vz0 + v_v[n] * v_cT[n]
    return xout, yout, zout, vxout, vyout, vzout


def write_general_info():
    """ write general info to file """
    fo = open(fpath_out + '/general.txt', 'w')
    fo.write('# neutron star parameters\n')
    fo.write('{:3E} # mass [solar masses]\n'.format(NS_mass))
    fo.write('{:3E} # radius [km]\n'.format(NS_radius))
    fo.write('# axion clump parameters\n')
    fo.write('{:3E} # mass [solar masses]\n'.format(AS_mass))
    fo.write('{:3E} # radius R90 [km]\n'.format(AS_radius))
    fo.write('{:3E} # velocity dispersion [km/s]\n'.format(AS_sigmav))
    fo.write('{:3E}  {:3E}  {:3E} # inital coordinates [km]\n'.format(
        AS_r0[0], AS_r0[1], AS_r0[2]))
    fo.write('{:3E}  {:3E}  {:3E} # inital velocity [km/s]\n'.format(
        AS_v0[0], AS_v0[1], AS_v0[2]))
    fo.write('{:3E} # number of particles in lump\n'.format(Nparticles))
    fo.write(
        '{:3E} # radius [km] at which outgoing particles are dropped in calculation\n'
        .format(Rdrop))
    fo.write(
        '{:3E} # radius [km] within which orbits are written to file\n'.format(
            Rsave))
    fo.write(
        '{} # number of steps skipping when writing output\n'.format(nwrite))
    fo.close()


def write_pointParticle_orbit_zarr(AS_x, AS_y, AS_z, AS_vx, AS_vy, AS_vz, t):
    """ write axion star to file """
    fo = zarr.open(fpath_out + '/AS_pointParticle_orbit.zarr', 'w')
    fo.array("t", t)
    fo.array("AS_x", AS_x)
    fo.array("AS_y", AS_y)
    fo.array("AS_z", AS_z)
    fo.array("AS_vx", AS_vx)
    fo.array("AS_vy", AS_vy)
    fo.array("AS_vz", AS_vz)


def find_nsteps(Np, target_size=mem_size):
    """ returns number of steps,
       for which the output from Np particles
       should not take more than [target_size] GB in memory """
    return max([int(1e9 * target_size / (Np * 40.)), 1])


def find_inds_active(x, y, z, vx, vy, vz, Rcut=Rdrop):
    inds_ingoing = np.where(x * vx + y * vy + z * vz < 0)[0]
    inds_Rmin = np.where(x**2. + y**2. + z**2. < Rcut**2.)[0]
    return sorted(list(set(inds_ingoing) | set(inds_Rmin)))


def run_particles_nsteps(x0, y0, z0, vx0, vy0, vz0, t0, target_size=mem_size):
    """ runs the particles, choosing the number of steps
       such that the memory requirement does not exceed 
       [target_size] GB.
       return lists of position, velocity, and time vectors """
    startTfun = time.time()
    nstep = find_nsteps(len(x0), target_size)
    print('the time is ' + time.strftime("%Hh%M"))
    print('starting to run {} active particles for {} steps'.format(
        len(x0), nstep))
    x = [x0]
    y = [y0]
    z = [z0]
    vx = [vx0]
    vy = [vy0]
    vz = [vz0]
    t = [t0]
    for i in range(nstep):
        temp_vals = update_r_v(x[-1], y[-1], z[-1], vx[-1], vy[-1], vz[-1], mu,
                               NS_radius)
        x.append(temp_vals[0])
        y.append(temp_vals[1])
        z.append(temp_vals[2])
        vx.append(temp_vals[3])
        vy.append(temp_vals[4])
        vz.append(temp_vals[5])
        t.append(t[-1] + temp_vals[6])
    print('finished  calculation after {} seconds'.format(
        (time.time() - startTfun)))
    return x, y, z, vx, vy, vz, t


def write_orbits_to_disk(x,
                         y,
                         z,
                         vx,
                         vy,
                         vz,
                         t,
                         inds_active,
                         Rcut=Rsave,
                         nskip=nwrite):
    """ appends the orbit files of particles in inds_active with the results 
       only every nskip-th timestep is written to file"""
    startTfun = time.time()
    t = np.array(t[::nskip]).T
    x = np.array(x[::nskip]).T
    y = np.array(y[::nskip]).T
    z = np.array(z[::nskip]).T
    vx = np.array(vx[::nskip]).T
    vy = np.array(vy[::nskip]).T
    vz = np.array(vz[::nskip]).T

    for i, ind in enumerate(inds_active):
        mask = x[i]**2 + y[i]**2 + z[i]**2 < Rcut**2

        out_zarr[str(ind)].append([t[i][mask],
                                 x[i][mask],
                                 y[i][mask],
                                 z[i][mask],
                                 vx[i][mask],
                                 vy[i][mask],
                                 vz[i][mask]], axis=1)
    print('finished writing data after {} seconds'.format(time.time() -
                                                          startTfun))


# ----------------------------------------------------
# run
# ----------------------------------------------------

# generate folder for output
startT = time.time()
str_startT = time.strftime("%Y%m%d_%H%M")
basedir = '/cfs/home/mala2765/scratch/pynasm/'
fpath_out = basedir + 'run_' + str_startT + id_generator()
os.mkdir(fpath_out)

# run the axion star as a point particle until it either reaches the
# disruption radius or flies away from the NS
AS_x = np.array([AS_r0[0]])
AS_y = np.array([AS_r0[1]])
AS_z = np.array([AS_r0[2]])
AS_vx = np.array([AS_v0[0]])
AS_vy = np.array([AS_v0[1]])
AS_vz = np.array([AS_v0[2]])
t = np.array([0.])
flag = 0
while flag == 0:
    x, y, z, vx, vy, vz, dt = update_r_v(np.array([AS_x[-1]]),
                                         np.array([AS_y[-1]]),
                                         np.array([AS_z[-1]]),
                                         np.array([AS_vx[-1]]),
                                         np.array([AS_vy[-1]]),
                                         np.array([AS_vz[-1]]), mu, NS_radius)
    AS_x = np.append(AS_x, x[0])
    AS_y = np.append(AS_y, y[0])
    AS_z = np.append(AS_z, z[0])
    AS_vx = np.append(AS_vx, vx[0])
    AS_vy = np.append(AS_vy, vy[0])
    AS_vz = np.append(AS_vz, vz[0])
    t = np.append(t, t[-1] + dt)
    if np.sqrt(
            x**2. + y**2. + z**2.
    ) < R_dis:    # check if axion star has reached the disuption radius
        flag = 1
    elif x * vx + y * vy + z * vz > 0:    # check if axion star is outbound
        flag = 2

# write results of inital calculation to file, and generate axion
# star as collection of particles
print("Calculation of axion star as point particle finished.")
if flag == 1:
    print("Your axion made it to the disruption radius")
    print("That took {:3E} years".format(t[-1] / 3.154e7))
    print("writing parameters and orbit to file")
    write_general_info()
    write_pointParticle_orbit_zarr(AS_x, AS_y, AS_z, AS_vx, AS_vy, AS_vz, t)
    print("generating axion star as {} particles".format(int(Nparticles)))
    pAS_x, pAS_y, pAS_z, pAS_vx, pAS_vy, pAS_vz = mk_axstar(
        AS_x[-1], AS_y[-1], AS_z[-1], AS_vx[-1], AS_vy[-1], AS_vz[-1],
        Nparticles)
elif flag == 2:
    print("Your axion star never came close enough to the neutron star")
    print("writing parameters and orbit to file")
    write_pointParticle_orbit_zarr(AS_x, AS_y, AS_z, AS_vx, AS_vy, AS_vz, t)
    print("aborting program...")
    sys.exit()

# log file

original_stdout = sys.stdout # Save a reference to the original standard output

logfile = open(fpath_out + '/log.txt', 'w')
sys.stdout = logfile # Change the standard output to the file we created.

# output zarr group:
out_zarr = zarr.open(fpath_out + '/orbits.zarr')
compressor = LZ4()
#compressor = Blosc(cname='lz4')
for i in range(Nparticles):
    out_zarr.array(str(i), np.empty((7, 0)), chunks=((7, 25000)), compressor=compressor)

# run the particles until all (except at most 5) are outbound and outside
# Rcut set in find_inds_active
t = np.full(pAS_x.shape, t[-1])
inds_active = find_inds_active(pAS_x, pAS_y, pAS_z, pAS_vx, pAS_vy, pAS_vz)
while len(inds_active) > 5:
    x_list, y_list, z_list, vx_list, vy_list, vz_list, t_list = run_particles_nsteps(
        pAS_x[inds_active], pAS_y[inds_active], pAS_z[inds_active],
        pAS_vx[inds_active], pAS_vy[inds_active], pAS_vz[inds_active],
        t[inds_active])
    write_orbits_to_disk(x_list, y_list, z_list, vx_list, vy_list, vz_list,
                         t_list, inds_active)
    # prepare next step
    pAS_x[inds_active], pAS_y[inds_active], pAS_z[inds_active], pAS_vx[
        inds_active], pAS_vy[inds_active], pAS_vz[inds_active], t[
            inds_active] = x_list[-1], y_list[-1], z_list[-1], vx_list[
                -1], vy_list[-1], vz_list[-1], t_list[-1]
    inds_active = find_inds_active(pAS_x, pAS_y, pAS_z, pAS_vx, pAS_vy, pAS_vz)

logfile.close()
sys.stdout = original_stdout # Reset the standard output to its
