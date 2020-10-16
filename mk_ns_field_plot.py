import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

import ns_field

ns = ns_field.ns()

fig, ax = plt.subplots()

# Grid of x, y points on a Cartesian grid
nx, ny = 64, 64
XMAX, YMAX = ns.radius * 10, ns.radius * 10
x = np.linspace(-XMAX, XMAX, nx)
y = np.linspace(-YMAX, YMAX, ny)
X, Y = np.meshgrid(x, y)

coords = np.column_stack((X.flat, Y.flat, np.zeros(len(Y.flat))))
t = 0
B_coords = ns.mag_field(t, coords)
Bx = np.reshape(B_coords[:, 0], X.shape)
By = np.reshape(B_coords[:, 1], X.shape)


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

    ax.add_patch(Circle((0,0), ns.radius, color='b', zorder=100))

    return ln.lines,

def update(t):
    ax.collections = [] # clear lines streamplot
    ax.patches = [] # clear arrowheads streamplot
    # Magnetic field vector, B = (Ex, Ey), as separate components
    B_coords = ns.mag_field(t, coords)
    Bx = np.reshape(B_coords[:, 0], X.shape)
    By = np.reshape(B_coords[:, 1], X.shape)

    # Plot the streamlines with an appropriate colormap and arrow style
    color = 2 * np.log(np.hypot(Bx, By))
    ln = ax.streamplot(x, y, Bx, By, color=color, linewidth=1, cmap=plt.cm.inferno,
                  density=2, arrowstyle='->', arrowsize=1.5)


    # Add a filled circle for the Earth; make sure it's on top of the streamlines.
    ax.add_patch(Circle((0,0), ns.radius, color='b', zorder=100))
    return ln.lines,

ani = FuncAnimation(fig, update, frames=np.linspace(0, ns.ns_day, 16),
                    init_func=init, blit=True)
#ani.save('animation.mp4', writer='ffmpeg', fps=5)
print("done saving")
plt.show()
