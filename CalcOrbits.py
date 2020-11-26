from __future__ import division

import os
import sys
import time
import random
import string

import numpy as np
from scipy.optimize import root
import zarr
from numcodecs import LZ4, Blosc

# ----------------------------------------------------
# enter parameters
# units: Masses [solar masses]
#        Distances [km]
#        Time      [s]

Nparticles = int(1e5)  # number of particles in axion clump

# initial conditions for axion clump
AC_r0 = np.array([1e14, 6e3, 0.])  # initial position [km]
AC_v0 = np.array([-300., 0., 0.])  # initial velocity [km/s]

# parameters for axion minicluster
MC_switch = False  # set to true if you want to simulate a minicluster
MC_mass = 1e-11  # minicluster mass in [solar masses]
MC_delta = 1  # initial overdensity of the minicluster
# NFW profile:
MC_NFW_switch = True  # set to true for a NFW density profile for the minicluster
MC_c = 100  # concentration parameter for the minicluster
# Power law profile:
MC_PL_switch = False  # set to true for a power law density profile for the minicluster

# parameters for (dilute) axion star
AS_switch = True
AS_mass = 1e-13  # mass [M_sol]
ax_mass = 2.6e-5  # axion (particle) mass in [eV]

# parameters for neutron star
NS_mass = 1.4  # mass [M_sol]
NS_radius = 10.  # radius [km]

# some parameters for the code and the output
Rdrop = 1e4  # radius [km] at which particles leaving the neutron star are dropped from the calculation
Rsave = 1e3  # radius [km] within in which orbits are written to file
nwrite = 100  # number of steps in output to skip when writing to file
mem_size = 1.  # target size of memory [GB] the calculation fills


# switch for the way the results are written to disk
out_format_switch = 1
# if out_format_switch == 1, orbits are written as a collection of plain text files
# if out_format_switch == 2, orbits written as zarr files


# ----------------------------------------------------
# constants
G_N = 1.325e11  # Newton constant in km^3/Msol/s^2

# ----------------------------------------------------
# functions


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def update_r_v(rx, ry, rz, vx, vy, vz, mu, NSR, rprecision=1e-3, dtmax=1e25):
    """ returns updated r and v and dt for particles """
    r = np.sqrt(rx**2 + ry**2 + rz**2)
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    dt = r / v * rprecision
    if np.isscalar(dt):
        dt = np.min([dt, dtmax])
    else:
        dt = np.minimum(dt, dtmax * np.ones(dt.shape))
    # calculate the acceleration
    #r[np.where(r<NSR)]=NSR # soften acceleration inside the neutron star assuming that it has uniform density
    ax = -mu * rx / r**3
    ay = -mu * ry / r**3
    az = -mu * rz / r**3
    # update velocity and position
    out_rx = rx + vx * dt + 0.5 * ax * dt**2
    out_ry = ry + vy * dt + 0.5 * ay * dt**2
    out_rz = rz + vz * dt + 0.5 * az * dt**2
    out_vx = vx + ax * dt
    out_vy = vy + ay * dt
    out_vz = vz + az * dt
    # remove particles inside NSR:
    rout2 = out_rx**2 + out_ry**2 + out_rz**2
    if np.isscalar(dt):
        if rout2 < NSR**2:
            out_rx = out_ry = out_rz = 1e10 * Rdrop
            out_vx = out_vy = out_vz = 1e10
    else:
        inds = np.where(rout2 < NSR**2)
        out_rx[inds] = out_ry[inds] = out_rz[inds] = 1e3 * np.sqrt(3) * Rdrop
        out_vx[inds] = out_vy[inds] = out_vz[inds] = 1e5
    return out_rx, out_ry, out_rz, out_vx, out_vy, out_vz, dt


def rho_MC(delta, rhoeq=4.39e-38):
    """
   returns the characteristic density of an 
   axion minicluster in [solar masses/km^3]
   forming from an overdensity with 
   overdensity parameter delta.
   rhoeq is the matter density at matter 
   radiation equality in [solar masses/km^3]
   """
    return 140 * (1 + delta) * delta**3 * rhoeq


def NFW_rs(MCM, MCrho, MCc):
    """
   returns the scale radius in [km] for a minicluster
   with 
   - mass MCM in [solar masses]
   - characteristic density MCrho [solar masses/km^3]
   - concentration parameter MCc
   """
    f = np.log(1. + MCc) - MCc / (1. + MCc)
    return (MCM / (4. * np.pi * MCrho * f))**(1. / 3.)


def NFW_R90_fun(x, c):
    """ 
   helper function to get R90
   for a NFW profile
   """
    return 0.9 * (np.log(1. + c) - c / (1. + c)) - (np.log(1. + x) - x /
                                                    (1. + x))


def R90_NFW(MCM, MCrho, MCc):
    """
   returns R90 in [km] for a minicluster with a
   NFW density profile and
   - mass MCM in[solar masses]
   - characteristic density MCrho [solar masses/km^3]
   - concentration parameter MCc
   """
    rs = NFW_rs(MCM, MCrho, MCc)
    x = root(NFW_R90_fun, MCc, args=MCc).x[0]
    return x * rs


def dens_dist_NFW(x0, y0, z0, MCM, MCrho, MCc, Np):
    """
   returns Np long lists of positions in [km]
   in cartesian coordinates,
   assuming a NFW distribution centered at 
   x0, z0, y0 in [km] for an axion minicluster with
   - mass MCM in [solar masses]
   - characteristic density MCrho in [solar masses/km^3]
   - concentration parameter MCc  
   """
    rs = NFW_rs(MCM, MCrho, MCc)
    rvec = np.linspace(0, MCc * rs, int(1e6))
    rpdf = rvec**2 / (rvec / rs * (1. + rvec / rs)**2)
    rpdf[0] = 0.  # fix first entry
    # generate random distributions
    rng = np.random.default_rng()
    costheta = rng.uniform(-1., 1., size=int(Np))
    phi = rng.uniform(0, 2. * np.pi, size=int(Np))
    r = rng.choice(rvec, p=rpdf / np.sum(rpdf), size=int(Np))
    # generate out vectors
    x = x0 + r * np.sin(np.arccos(costheta)) * np.cos(phi)
    y = y0 + r * np.sin(np.arccos(costheta)) * np.sin(phi)
    z = z0 + r * costheta
    return x, y, z


def R90_PL(MCM, MCrho):
    """
   returns R90 in [km] for a minicluster with a
   Power-Law (index 9/4) density profile and
   - mass MCM in[solar masses]
   - characteristic density MCrho [solar masses/km^3]
   """
    return (0.9**4 * 3. * MCM / (4. * np.pi * MCrho))**(1. / 3.)


def dens_dist_PL(x0, y0, z0, MCM, MCrho, Np):
    """
   returns Np long lists of positions in [km]
   in cartesian coordinates,
   assuming a Power-Law profile (index 9/4) centered at 
   x0, z0, y0 in [km] for an axion minicluster with
   - mass MCM in [solar masses]
   - characteristic density MCrho in [solar masses/km^3]
   """
    RPL = (3. * MCM / (4. * np.pi * MCrho))**(1. / 3.)  # truncation radius
    rvec = np.linspace(0, RPL, int(1e6))
    rpdf = rvec**-0.25
    rvec = rvec[1:]
    rpdf = rpdf[1:]
    # generate random distributions
    rng = np.random.default_rng()
    costheta = rng.uniform(-1., 1., size=int(Np))
    phi = rng.uniform(0, 2. * np.pi, size=int(Np))
    r = rng.choice(rvec, p=rpdf / np.sum(rpdf), size=int(Np))
    # generate out vectors
    x = x0 + r * np.sin(np.arccos(costheta)) * np.cos(phi)
    y = y0 + r * np.sin(np.arccos(costheta)) * np.sin(phi)
    z = z0 + r * costheta
    return x, y, z


def R90_AS(ASM, ma):
    """
   returns R90 in [km] for a 
   dilute axion star with
   - axion star mass ASM in [solar masses]
   - axion particle mass ma in [eV]
   """
    ak = 9.9  # numerical factor from [1710.08910]
    c_kms = 2.998e5  # speed of light in [km/s]
    hbarc_eVkm = 1.973e-10  # hbar*c in [eV.km]
    unitfac = c_kms**2 * hbarc_eVkm**2
    return unitfac * ak / G_N / ma**2 / ASM


def dens_dist_sech(x0, y0, z0, ASM, ma, Np):
    """
   returns Np long lists of positions in [km]
   in cartesian coordinates,
   assuming the sech density profile [1710.04729]
   for an dilute axion star with mass ASM in [solar masses] 
   for an axion (particle) mass ma in [eV] centered at 
   x0, z0, y0 in [km] for an axion minicluster with
   - axion star mass ASM in [solar masses]
   - axion particle mass ma in [eV]
   """
    Rsech = R90_AS(ASM, ma) / 2.799  # numerical factor from [1710.04729]
    rvec = np.linspace(0, 10. * Rsech, int(1e6))
    rpdf = rvec**2 / np.cosh(rvec / Rsech)**2
    # generate random distributions
    rng = np.random.default_rng()
    costheta = rng.uniform(-1., 1., size=int(Np))
    phi = rng.uniform(0, 2. * np.pi, size=int(Np))
    r = rng.choice(rvec, p=rpdf / np.sum(rpdf), size=int(Np))
    # generate out vectors
    x = x0 + r * np.sin(np.arccos(costheta)) * np.cos(phi)
    y = y0 + r * np.sin(np.arccos(costheta)) * np.sin(phi)
    z = z0 + r * costheta
    return x, y, z


def velocity_dist_flat(vx0, vy0, vz0, Np, vesc):
    """
   returns Np long lists of velocities in [km/s] 
   in cartesian coordinates.
   assumes a flat distribution cut off at vesc in the frame 
   of the axion clump,
   boosted to the frame specified by vx0, vy0, vz0 in [km/s]
   out:
      tuple of velocity vectors in cartesian coordinates 
      (vx, vy, vz)
      units fixed by input units of vxi and vesc
   """
    vxout = np.zeros(Np)
    vyout = np.zeros(Np)
    vzout = np.zeros(Np)
    # generate random distributions
    rng = np.random.default_rng()
    costheta = rng.uniform(-1., 1., size=int(Np))
    phi = rng.uniform(0, 2. * np.pi, size=int(Np))
    v = vesc * rng.power(3, size=int(Np))
    # generate out vectors
    vx = vx0 + v * np.sin(np.arccos(costheta)) * np.cos(phi)
    vy = vy0 + v * np.sin(np.arccos(costheta)) * np.sin(phi)
    vz = vz0 + v * costheta
    return vx, vy, vz


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


def write_general_info():
    """ write general info to file """
    fo = open(fpath_out + '/general.txt', 'w')
    fo.write('# neutron star parameters\n')
    fo.write('{:3E} # mass [solar masses]\n'.format(NS_mass))
    fo.write('{:3E} # radius [km]\n'.format(NS_radius))
    fo.write('# axion clump parameters\n')
    if MC_switch and MC_NFW_switch:
        fo.write('# this is an axion minicluster with a NFW density profile\n')
    elif MC_switch and MC_PL_switch:
        fo.write('# this is an axion minicluster with a power law profile\n')
    elif AS_switch:
        fo.write('# this is a dilute axion star\n')
        fo.write('{:3E} # axion mass [eV]\n'.format(ax_mass))
    fo.write('{:3E} # mass [solar masses]\n'.format(AC_mass))
    fo.write('{:3E} # radius R90 [km]\n'.format(AC_R90))
    fo.write('{:3E} # escape velocity [km/s]\n'.format(AC_vesc))
    fo.write('{:3E}  {:3E}  {:3E} # initial coordinates [km]\n'.format(
        AC_r0[0], AC_r0[1], AC_r0[2]))
    fo.write('{:3E}  {:3E}  {:3E} # initial velocity [km/s]\n'.format(
        AC_v0[0], AC_v0[1], AC_v0[2]))
    fo.write('{:3E} # disruption radius [km]\n'.format(R_dis))
    fo.write('{:3E} # number of particles in clump\n'.format(Nparticles))
    fo.write(
        '{:3E} # radius [km] at which outgoing particles are dropped in calculation\n'
        .format(Rdrop))
    fo.write(
        '{:3E} # radius [km] within which orbits are written to file\n'.format(
            Rsave))
    fo.write(
        '{} # number of steps skipping when writing output\n'.format(nwrite))
    fo.close()


def write_pointParticle_orbit(AC_x, AC_y, AC_z, AC_vx, AC_vy, AC_vz, t):
    """ write axion clump to file """
    fo = open(fpath_out + '/AC_pointParticle_orbit.txt', 'w')
    fo.write('# t[s]  x[km]  y[km]  z[km]  vx[km]  vy[km]  vz[km]\n')
    for i in range(len(AC_x)):
        fo.write('{:.12E}  '.format(t[i]))
        fo.write('{:.6E}  '.format(AC_x[i]))
        fo.write('{:.6E}  '.format(AC_y[i]))
        fo.write('{:.6E}  '.format(AC_z[i]))
        fo.write('{:.6E}  '.format(AC_vx[i]))
        fo.write('{:.6E}  '.format(AC_vy[i]))
        fo.write('{:.6E}'.format(AC_vz[i]))
        fo.write('\n')
    fo.close()


def write_pointParticle_orbit_zarr(AC_x, AC_y, AC_z, AC_vx, AC_vy, AC_vz, t):
    """ write axion star to file """
    fo = zarr.open(fpath_out + '/AC_pointParticle_orbit.zarr', 'w')
    fo.array("t", t)
    fo.array("AC_x", AC_x)
    fo.array("AC_y", AC_y)
    fo.array("AC_z", AC_z)
    fo.array("AC_vx", AC_vx)
    fo.array("AC_vy", AC_vy)
    fo.array("AC_vz", AC_vz)


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
    for i in range(len(inds_active)):
        fo = open(fout_orbit_names[inds_active[i]], 'a')
        j = 0
        while j < len(t) - 1:
            if (x[j][i]**2. + y[j][i]**2. + z[j][i]**2.) < Rcut**2.:
                fo.write(
                    '{:.12E}  {:.6E}  {:.6E}  {:.6E}  {:.6E}  {:.6E}  {:.6E}\n'
                    .format(t[j][i], x[j][i], y[j][i], z[j][i], vx[j][i],
                            vy[j][i], vz[j][i]))
            j += nskip
        fo.close()
    print('finished writing data after {} seconds'.format(time.time() -
                                                          startTfun))


def write_orbits_to_disk_zarr(x,
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
        out_zarr[str(ind)].append([
            t[i][mask], x[i][mask], y[i][mask], z[i][mask], vx[i][mask],
            vy[i][mask], vz[i][mask]
        ],
                                  axis=1)
    print('finished writing data after {} seconds'.format(time.time() -
                                                          startTfun))

# ----------------------------------------------------
# run
# ----------------------------------------------------
# generate folder for output
startT = time.time()
str_startT = time.strftime("%Y%m%d_%H%M")
basedir = './'
fpath_out = basedir + 'run_' + str_startT + id_generator()
os.mkdir(fpath_out)

# compute gravitational parameter
mu = NS_mass * G_N
# check if switches are set up correctly
if MC_switch * MC_NFW_switch + MC_switch * MC_PL_switch + AS_switch != 1:
    print(
        "you did not make a reasonable selection of the options for the axion clump (NFW_minicluster/PL_minicluster/axion star)"
    )
    print("aborting the code...")
    sys.exit()

# check if the minicluster is less dense than an dilute axion star
# this assumes that the axion mass is given by ax_mass
if MC_switch:
    if MC_NFW_switch and R90_NFW(MC_mass, rho_MC(MC_delta), MC_c) < R90_AS(
            MC_mass, ax_mass):
        print(
            "your minicluster is denser than an axion star for a {} eV axion".
            format(ax_mass))
        print("aborting the code...")
        sys.exit()
    elif MC_PL_switch and R90_PL(MC_mass, rho_MC(MC_delta)) < R90_AS(
            MC_mass, ax_mass):
        print(
            "your minicluster is denser than an axion star for a {} eV axion".
            format(ax_mass))
        print("aborting the code...")
        sys.exit()

# set up the mk_axclump function and calculate the disruption radius
if MC_switch:
    AC_mass = MC_mass
    if MC_NFW_switch:
        AC_R90 = R90_NFW(MC_mass, rho_MC(MC_delta), MC_c)
        AC_vesc = np.sqrt(2. * G_N * AC_mass / AC_R90)

        def mk_axclump(x0, y0, z0, vx0, vy0, vz0, Np):
            x, y, z = dens_dist_NFW(x0, y0, z0, AC_mass, rho_MC(MC_delta),
                                    MC_c, Np)
            vx, vy, vz = velocity_dist_flat(vx0, vy0, vz0, Np, AC_vesc)
            return x, y, z, vx, vy, vz
    elif MC_PL_switch:
        AC_R90 = R90_PL(MC_mass, rho_MC(MC_delta))
        AC_vesc = np.sqrt(2. * G_N * AC_mass / AC_R90)

        def mk_axclump(x0, y0, z0, vx0, vy0, vz0, Np):
            x, y, z = dens_dist_PL(x0, y0, z0, AC_mass, rho_MC(MC_delta), Np)
            vx, vy, vz = velocity_dist_flat(vx0, vy0, vz0, Np, AC_vesc)
            return x, y, z, vx, vy, vz
elif AS_switch:
    AC_mass = AS_mass
    AC_R90 = R90_AS(AC_mass, ax_mass)
    AC_vesc = np.sqrt(2. * G_N * AC_mass / AC_R90)

    def mk_axclump(x0, y0, z0, vx0, vy0, vz0, Np):
        x, y, z = dens_dist_sech(x0, y0, z0, AC_mass, ax_mass, Np)
        vx, vy, vz = velocity_dist_flat(vx0, vy0, vz0, Np, AC_vesc)
        return x, y, z, vx, vy, vz


# calculate Roche disruption radius
R_dis = AC_R90 * (2. * NS_mass / AC_mass)**(1. / 3.)  #[km]

# run the axion clump as a point particle until it either reaches the disruption radius or flies away from the NS
AC_x = np.array([AC_r0[0]])
AC_y = np.array([AC_r0[1]])
AC_z = np.array([AC_r0[2]])
AC_vx = np.array([AC_v0[0]])
AC_vy = np.array([AC_v0[1]])
AC_vz = np.array([AC_v0[2]])
t = np.array([0.])
flag = 0
while flag == 0:
    x, y, z, vx, vy, vz, dt = update_r_v(np.array([AC_x[-1]]),
                                         np.array([AC_y[-1]]),
                                         np.array([AC_z[-1]]),
                                         np.array([AC_vx[-1]]),
                                         np.array([AC_vy[-1]]),
                                         np.array([AC_vz[-1]]), mu, NS_radius)
    AC_x = np.append(AC_x, x[0])
    AC_y = np.append(AC_y, y[0])
    AC_z = np.append(AC_z, z[0])
    AC_vx = np.append(AC_vx, vx[0])
    AC_vy = np.append(AC_vy, vy[0])
    AC_vz = np.append(AC_vz, vz[0])
    t = np.append(t, t[-1] + dt)
    if np.sqrt(
            x**2. + y**2. + z**2.
    ) < R_dis:  # check if axion clump has reached the disuption radius
        flag = 1
    elif x * vx + y * vy + z * vz > 0:  # check if axion clump is outbound
        flag = 2

# write results of initial calculation to file, and generate axion clump as collection of particles
print("Calculation of axion clump as point particle finished.")
if flag == 1:
    print("Your clump made it to the disruption radius")
    print("That took {:3E} years".format(t[-1] / 3.154e7))
    print("writing parameters and orbit to file")
    write_general_info()
    if out_format_switch == 1:
      write_pointParticle_orbit(AC_x, AC_y, AC_z, AC_vx, AC_vy, AC_vz, t)
    elif out_format_switch == 2:
      write_pointParticle_orbit_zarr(AC_x, AC_y, AC_z, AC_vx, AC_vy, AC_vz, t)
    print("generating axion clump as {} particles".format(int(Nparticles)))
    pAC_x, pAC_y, pAC_z, pAC_vx, pAC_vy, pAC_vz = mk_axclump(
        AC_x[-1], AC_y[-1], AC_z[-1], AC_vx[-1], AC_vy[-1], AC_vz[-1],
        Nparticles)
elif flag == 2:
    print("Your axion clump never came close enough to the neutron star")
    print("writing parameters and orbit to file")
    write_general_info()
    if out_format_switch == 1:
      write_pointParticle_orbit(AC_x, AC_y, AC_z, AC_vx, AC_vy, AC_vz, t)
    elif out_format_switch == 2:
      write_pointParticle_orbit_zarr(AC_x, AC_y, AC_z, AC_vx, AC_vy, AC_vz, t)
    print("aborting program...")
    sys.exit()

# create file structure for particle output
if out_format_switch == 1:
  os.mkdir(fpath_out + '/orbits')
  fout_orbit_names = [
      fpath_out + '/orbits/p_' + str(int(i)) + '.txt' for i in range(Nparticles)
  ]
  for i in range(Nparticles):
      fo = open(fout_orbit_names[i], 'w')
      fo.write('# t[s]  x[km]  y[km]  z[km]  vx[km]  vy[km]  vz[km]\n')
      fo.close()
elif out_format_switch == 2:
  out_zarr = zarr.open(fpath_out + '/orbits.zarr')
  compressor = LZ4()
  #compressor = Blosc(cname='lz4')
  for i in range(Nparticles):
      out_zarr.array(str(i),
                     np.empty((7, 0)),
                     chunks=((7, 25000)),
                     compressor=compressor)


# reset the clock
t = [0.]
# run the particles until all (except at most 5) are outbound and outside Rcut set in find_inds_active
t = np.full(pAC_x.shape, t[-1])
inds_active = find_inds_active(pAC_x, pAC_y, pAC_z, pAC_vx, pAC_vy, pAC_vz)
while len(inds_active) > 5:
    x_list, y_list, z_list, vx_list, vy_list, vz_list, t_list = run_particles_nsteps(
        pAC_x[inds_active], pAC_y[inds_active], pAC_z[inds_active],
        pAC_vx[inds_active], pAC_vy[inds_active], pAC_vz[inds_active],
        t[inds_active])
    if out_format_switch == 1:
      write_orbits_to_disk(x_list, y_list, z_list, vx_list, vy_list, vz_list,
                         t_list, inds_active)
    elif out_format_switch == 2:
      write_orbits_to_disk_zarr(x_list, y_list, z_list, vx_list, vy_list, vz_list,
                         t_list, inds_active)
    # prepare next step
    pAC_x[inds_active], pAC_y[inds_active], pAC_z[inds_active], pAC_vx[
        inds_active], pAC_vy[inds_active], pAC_vz[inds_active], t[
            inds_active] = x_list[-1], y_list[-1], z_list[-1], vx_list[
                -1], vy_list[-1], vz_list[-1], t_list[-1]
    inds_active = find_inds_active(pAC_x, pAC_y, pAC_z, pAC_vx, pAC_vy, pAC_vz)
