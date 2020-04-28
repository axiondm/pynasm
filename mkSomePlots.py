from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import os

# ----------------------------------------------------
# select particles to plot, load data, and add to list "orbits"
#  # ----------------------------------------------------
# select some list of orbits:
inds_plot=range(1000)
fi_orbit_names=['orbits/p_'+str(int(i))+'.txt' for i in inds_plot]
orbits=[np.loadtxt(name) for name in fi_orbit_names]
# # ----------------------------------------------------
# # select orbits coming within Rcut list of more or less random orbits:
# Rcut=1e5
# inds_try=range(int(1e3))
# fi_orbit_names=['orbits/p_'+str(int(i))+'.txt' for i in inds_try]
# orbits=[]
# for fname in fi_orbit_names:
#    orbit=np.loadtxt(fname)
#    if np.min(np.sqrt(orbit[:,1]**2+orbit[:,2]**2+orbit[:,3]**2))<Rcut:
#       orbits.append(orbit)

# ----------------------------------------------------
# select sizes of plotwindows
r_boxsize=5e2
v_boxsize=3e5

t_box_l=-1e1
t_box_r=1e1
t_offset=0
rmin=1e10
for orbit in orbits:
   r=np.sqrt(orbit[:,1]**2+orbit[:,2]**2+orbit[:,3]**2)
   if np.min(r)<rmin:
      t_offset=orbit[np.argmin(r),0]

# ----------------------------------------------------
# mk plots
# plot
fs = 22
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({'legend.fontsize': fs,
       'axes.labelsize': fs,
         'axes.titlesize': fs,
         'xtick.labelsize': fs,
         'ytick.labelsize': fs})
plt.rcParams["figure.figsize"] = (8,8)


if os.path.isdir('plots')==False:
   os.makedirs('plots')

# plot in x-y plane
plt.close('all')
for pi in orbits:
   plt.plot(pi[:,1],pi[:,2])

plt.xlabel(r'$x$ [km]')
plt.ylabel(r'$y$ [km]')
plt.axes().xaxis.set_minor_locator(AutoMinorLocator(5))
plt.axes().yaxis.set_minor_locator(AutoMinorLocator(5))
plt.xlim(-r_boxsize,r_boxsize)
plt.ylim(-r_boxsize,r_boxsize)
plt.tick_params(top=True,right=True,length=8)
plt.tick_params(which='minor',top=True,right=True,length=5)
plt.tight_layout()
plt.savefig('plots/x-y.pdf')

# plot in x-z plane
plt.close('all')
for pi in orbits:
   plt.plot(pi[:,1],pi[:,3])

plt.xlabel(r'$x$ [km]')
plt.ylabel(r'$z$ [km]')
plt.axes().xaxis.set_minor_locator(AutoMinorLocator(5))
plt.axes().yaxis.set_minor_locator(AutoMinorLocator(5))
plt.xlim(-r_boxsize,r_boxsize)
plt.ylim(-r_boxsize,r_boxsize)
plt.tick_params(top=True,right=True,length=8)
plt.tick_params(which='minor',top=True,right=True,length=5)
plt.tight_layout()
plt.savefig('plots/x-z.pdf')

# plot in y-z plane
plt.close('all')
for pi in orbits:
   plt.plot(pi[:,2],pi[:,3])

plt.xlabel(r'$y$ [km]')
plt.ylabel(r'$z$ [km]')
plt.axes().xaxis.set_minor_locator(AutoMinorLocator(5))
plt.axes().yaxis.set_minor_locator(AutoMinorLocator(5))
plt.xlim(-r_boxsize,r_boxsize)
plt.ylim(-r_boxsize,r_boxsize)
plt.tick_params(top=True,right=True,length=8)
plt.tick_params(which='minor',top=True,right=True,length=5)
plt.tight_layout()
plt.savefig('plots/y-z.pdf')

# plot in x-v_x plane
plt.close('all')
for pi in orbits:
   plt.plot(pi[:,1],pi[:,4])

plt.xlabel(r'$x$ [km]')
plt.ylabel(r'$v_x$ [km/s]')
plt.axes().xaxis.set_minor_locator(AutoMinorLocator(5))
plt.axes().yaxis.set_minor_locator(AutoMinorLocator(5))
plt.xlim(-r_boxsize,r_boxsize)
plt.ylim(-v_boxsize,v_boxsize)
plt.tick_params(top=True,right=True,length=8)
plt.tick_params(which='minor',top=True,right=True,length=5)
plt.tight_layout()
plt.savefig('plots/x-vx.pdf')

# plot in y-v_y plane
plt.close('all')
for pi in orbits:
   plt.plot(pi[:,2],pi[:,5])

plt.xlabel(r'$y$ [km]')
plt.ylabel(r'$v_y$ [km/s]')
plt.axes().xaxis.set_minor_locator(AutoMinorLocator(5))
plt.axes().yaxis.set_minor_locator(AutoMinorLocator(5))
plt.xlim(-r_boxsize,r_boxsize)
plt.ylim(-v_boxsize,v_boxsize)
plt.tick_params(top=True,right=True,length=8)
plt.tick_params(which='minor',top=True,right=True,length=5)
plt.tight_layout()
plt.savefig('plots/y-vy.pdf')

# plot in z-v_z plane
plt.close('all')
for pi in orbits:
   plt.plot(pi[:,3],pi[:,6])

plt.xlabel(r'$z$ [km]')
plt.ylabel(r'$v_z$ [km/s]')
plt.axes().xaxis.set_minor_locator(AutoMinorLocator(5))
plt.axes().yaxis.set_minor_locator(AutoMinorLocator(5))
plt.xlim(-r_boxsize,r_boxsize)
plt.ylim(-v_boxsize,v_boxsize)
plt.tick_params(top=True,right=True,length=8)
plt.tick_params(which='minor',top=True,right=True,length=5)
plt.tight_layout()
plt.savefig('plots/z-vz.pdf')

# plot in |r|, |v| plane
plt.close('all')
for pi in orbits:
   plt.plot(np.sqrt(pi[:,1]**2+pi[:,2]**2+pi[:,3]**2),
      np.sqrt(pi[:,4]**2+pi[:,5]**2+pi[:,6]**2))

plt.xlabel(r'$|\vec{r}|$ [km]')
plt.ylabel(r'$|\vec{v}|$ [km/s]')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e0,r_boxsize)
plt.ylim(1e2,v_boxsize)
plt.tick_params(top=True,right=True,length=8)
plt.tick_params(which='minor',top=True,right=True,length=5)
plt.tight_layout()
plt.savefig('plots/r-v.pdf')

# plot in t, |r| plane
plt.close('all')
for pi in orbits:
   plt.plot(pi[:,0]-t_offset,np.sqrt(pi[:,1]**2+pi[:,2]**2+pi[:,3]**2))

plt.xlabel(r'$t$ [s]')
plt.ylabel(r'$|\vec{r}|$ [km]')
plt.yscale('log')
plt.xlim(t_box_l,t_box_r)
plt.ylim(1e0,r_boxsize)
plt.tick_params(top=True,right=True,length=8)
plt.tick_params(which='minor',top=True,right=True,length=5)
plt.tight_layout()
plt.savefig('plots/t-r.pdf')

# plot in t, |v| plane
plt.close('all')
for pi in orbits:
   plt.plot(pi[:,0]-t_offset,np.sqrt(pi[:,4]**2+pi[:,5]**2+pi[:,6]**2))

plt.xlabel(r'$t$ [s]')
plt.ylabel(r'$|\vec{v}|$ [km]')
plt.yscale('log')
plt.xlim(t_box_l,t_box_r)
plt.ylim(1e2,1e6)
plt.tick_params(top=True,right=True,length=8)
plt.tick_params(which='minor',top=True,right=True,length=5)
plt.tight_layout()
plt.savefig('plots/t-v.pdf')