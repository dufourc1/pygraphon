# -----------------------------------------------------------------------------
# Copyright (c) 2020 Nicolas P. Rougier. All rights reserved.
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------
# This example shows how to render bars
# -----------------------------------------------------------------------------
# import sys
# sys.path.insert(0, r"c:\users\arthu\documents\swittz-epfl\phd\code\lib\site-packages")
from mpl3d import glm
import numpy as np
import matplotlib as mpl
from matplotlib.collections import PolyCollection

import matplotlib.pyplot as plt


class Bar:
    """ Bar (histogram) """
    def __init__(self, ax, transform,  Z,
                 facecolors="white", edgecolors="black",  linewidth=0, clip=False):
        """ """
        self.Z = Z
        if isinstance(facecolors, np.ndarray):
            shape = facecolors.shape
            facecolors = facecolors.reshape(-1, shape[-1])
            facecolors = mpl.colors.to_rgba_array(facecolors)
            self.facecolors = facecolors.reshape(shape[0], shape[1], 4)
        else:
            shape = Z.shape
            self.facecolors = np.zeros((shape[0], shape[1], 4))
            self.facecolors[...] = mpl.colors.to_rgba(facecolors)

        if isinstance(edgecolors, np.ndarray):
            shape = edgecolors.shape
            edgecolors = edgecolors.reshape(-1, shape[-1])
            edgecolors = mpl.colors.to_rgba_array(edgecolors)
            self.edgecolors = edgecolors.reshape(shape[0], shape[1], 4)
        else:
            shape = Z.shape
            self.edgecolors = np.zeros((shape[0], shape[1], 4))
            self.edgecolors[...] = mpl.colors.to_rgba(edgecolors)

        self.linewidth = linewidth
        self.xlim = -0.5, +0.50
        self.ylim = -0.5, +0.50
        self.zlim = -0.5, 0.5
        self.clip = clip

        # Because all the bars have the same orientation, we can use a hack to
        # shade each face at once instead of computing individual face lighting.
        self.shade = np.array([[1.00, 1.00, 0.75, 1.00, 0.50, 1.00]])

        self.collection = PolyCollection([], clip_on=self.clip, snap=False)
        self.update(transform)
        ax.add_collection(self.collection, autolim=False)

    def update(self, transform):
        """ """

        Z = self.Z
        xmin, xmax = self.xlim
        ymin, ymax = self.ylim
        zmin, _ = self.zlim
        dx, dy = 0.5 * 1/Z.shape[0], 0.5 * 1/Z.shape[1]

        # Each bar is described by 8 vertices and 6 faces
        V = np.zeros((Z.shape[0], Z.shape[1], 8, 3))
        F = np.zeros((Z.shape[0], Z.shape[1], 6, 4), dtype=int)

        # Face and edge colors for the six faces
        FC = np.zeros((Z.shape[0], Z.shape[1], 6, 4))
        FC[:, :] = self.facecolors.reshape(Z.shape[0], Z.shape[1], 1, 4)
        FC *= self.shade.T
        FC[:, :, :, 3] = 1

        EC = np.zeros((Z.shape[0], Z.shape[1], 6, 4))
        EC[:, :] = self.edgecolors.reshape(Z.shape[0], Z.shape[1], 1, 4)

        # Build vertices
        X, Y = np.meshgrid(np.linspace(xmin, xmax, Z.shape[0]),
                           np.linspace(ymin, ymax, Z.shape[1]))
        V[..., 0] = X.reshape(Z.shape[0], Z.shape[1], 1)
        V[..., 1] = Y.reshape(Z.shape[0], Z.shape[1], 1)

        V[:, :, 0] += [+dx, +dy, zmin]
        V[:, :, 1] += [+dx, -dy, zmin]
        V[:, :, 2] += [-dx, -dy, zmin]
        V[:, :, 3] += [-dx, +dy, zmin]

        V[:, :, 4] += [+dx, +dy, zmin]
        V[:, :, 5] += [+dx, -dy, zmin]
        V[:, :, 6] += [-dx, -dy, zmin]
        V[:, :, 7] += [-dx, +dy, zmin]
        V[:, :, 4:, 2] += Z.reshape(Z.shape[0], Z.shape[1], 1)

        # Build faces
        f = 8*np.arange(Z.shape[0]*Z.shape[1])
        F[:, :] = f.reshape(Z.shape[0], Z.shape[1], 1, 1)
        F[:, :] += [[0, 1, 2, 3],  # -Z
                    [0, 1, 5, 4],  # +X
                    [2, 3, 7, 6],  # -X
                    [1, 2, 6, 5],  # -Y
                    [0, 3, 7, 4],  # +Y
                    [4, 5, 6, 7]]  # +Z

        # Actual transformation
        V = V.reshape(-1, 3)
        V = glm.transform(V[F], transform)  # [...,:2]

        # Depth computation
        # We combine the global "depth" of the bar (depth of the bottom face)
        # and the local depth of each face. This trick avoids problems when
        # sorting all the different faces.
        Z1 = (V[:, :, 0, :, 2].mean(axis=2)).reshape(Z.shape[0], Z.shape[1], 1)
        Z2 = (V[..., 2].mean(axis=3) + 10*Z1).ravel()

        # Sorting
        id = np.argsort(-Z2)
        V = (V[..., :2].reshape(Z.shape[0]*Z.shape[1]*6, 4, 2))

        self.collection.set_verts(V[id])
        self.collection.set_facecolors(FC.reshape(-1, 4)[id])
        self.collection.set_edgecolors(EC.reshape(-1, 4)[id])
        self.collection.set_linewidths(self.linewidth)
        if self.linewidth == 0.0:
            self.collection.set_antialiased(False)
        else:
            self.collection.set_antialiased(True)


def plot_3d_and_reflection(arr):
    # Create a colormap for visualization
    cmap = plt.get_cmap('jet')

    # 3D Plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Generate coordinates for the bars
    x_pos, y_pos = np.meshgrid(np.arange(arr.shape[1]), np.arange(arr.shape[0]))
    x_pos = x_pos.flatten()
    y_pos = y_pos.flatten()
    z_pos = np.zeros_like(x_pos)

    # Generate the heights of the bars proportional to the array values
    dx = dy = 0.8
    dz = arr.flatten()

    # Plot the 3D bars
    ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, shade=True, color=cmap(arr.flatten() / arr.max()))

    # Set labels and title for the 3D plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Plot with Color and Height')

    # 2D Reflection Plot
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    # Plot the 2D array with colors
    im = ax2.imshow(arr, cmap=cmap)

    # Create a colorbar for the 2D plot
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Value')

    # Set labels and title for the 2D plot
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('2D Reflection Plot with Color')

    # Show the plots
    plt.show()
