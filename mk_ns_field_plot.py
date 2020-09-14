import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

import ns_field

ns = ns_field.ns()

# Mean magnitude of the Earth's magnetic field at the equator in T
B0 = 3.12e-5
# Radius of Earth, Mm (10^6 m: mega-metres!)
RE = 6.370
# Deviation of magnetic pole from axis
# alpha = np.radians(9.6)

def B(r, theta, alpha):
    """Return the magnetic field vector at (r, theta)."""
    fac = B0 * (RE / r)**3
    return -2 * fac * np.cos(theta + alpha), -fac * np.sin(theta + alpha)

fig, ax = plt.subplots()

# Grid of x, y points on a Cartesian grid
nx, ny = 64, 64
XMAX, YMAX = 40, 40
x = np.linspace(-XMAX, XMAX, nx)
y = np.linspace(-YMAX, YMAX, ny)
X, Y = np.meshgrid(x, y)
r, theta = np.hypot(X, Y), np.arctan2(Y, X)

coords = np.column_stack((X.flat, Y.flat, np.zeros(len(Y.flat))))
t = 0

# Magnetic field vector, B = (Ex, Ey), as separate components
Br, Btheta = B(r, theta, np.radians(9.6))
# Transform to Cartesian coordinates: NB make North point up, not to the right.
c, s = np.cos(np.pi/2 + theta), np.sin(np.pi/2 + theta)
Bx = -Btheta * s + Br * c
By = Btheta * c + Br * s


# Plot the streamlines with an appropriate colormap and arrow style
color = 2 * np.log(np.hypot(Bx, By))
ln = ax.streamplot(x, y, Bx, By, color=color, linewidth=1, cmap=plt.cm.inferno,
              density=2, arrowstyle='->', arrowsize=1.5)


def init():
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_xlim(-XMAX, XMAX)
    ax.set_ylim(-YMAX, YMAX)
    ax.set_aspect('equal')


    return ln.lines,

def update(frame):
    t = frame
    alpha = np.radians(9.6) * np.cos(t)

    ax.collections = [] # clear lines streamplot
    ax.patches = [] # clear arrowheads streamplot
    # Magnetic field vector, B = (Ex, Ey), as separate components
    Br, Btheta = B(r, theta, alpha)
    # Transform to Cartesian coordinates: NB make North point up, not to the right.
    c, s = np.cos(np.pi/2 + theta), np.sin(np.pi/2 + theta)
    Bx = -Btheta * s + Br * c
    By = Btheta * c + Br * s


    # Plot the streamlines with an appropriate colormap and arrow style
    color = 2 * np.log(np.hypot(Bx, By))
    ln = ax.streamplot(x, y, Bx, By, color=color, linewidth=1, cmap=plt.cm.inferno,
                  density=2, arrowstyle='->', arrowsize=1.5)


    # Add a filled circle for the Earth; make sure it's on top of the streamlines.
    ax.add_patch(Circle((0,0), RE, color='b', zorder=100))
    return ln.lines,

ani = FuncAnimation(fig, update, frames=np.linspace(0, np.pi, 16),
                    init_func=init, blit=True)
ani.save('animation.mp4', writer='ffmpeg', fps=5)
print("done saving")
plt.show()
