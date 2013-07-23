import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
from mpl_toolkits.mplot3d import Axes3D


def plot_3d(pts, elev=8, azim=-10, show=False, **kwargs):
    """ Plots the given point cloud in stero 3D.

    Parameters
    ----------
    elev : int
        Elevation of camera between 0 and 360
    azim : int
        Azimuth of camera between 0 and 360
    show : bool
        Should plot be shown immediately?
    kwargs : optional
        Optional kw arguments to be fed to plot method

    """
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], **kwargs)
    ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], **kwargs)
    ax1.elev, ax1.azim = elev, azim
    ax2.elev, ax2.azim = elev, azim - 7
    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if show:
        plt.show()
    return fig


def save_figure(fig, name='fig', size=None, out_dir='out'):
    if size is not None:
        fig.set_size_inches(size[0], size[1])
    fig.savefig('{}/{}.png'.format(out_dir, name))


def save_animation(cloud, step=5):
    for azim in np.arange(0, 360, step):
        fig = plot_3d(cloud, azim=azim, s=50)
        save_figure(fig, name='{0:03d}'.format(azim), size=(10, 5))


def get_rotation_matrix(angle):
    # Rotate about the z-axis
    cos, sin = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[cos,  sin, 0],
                                [-sin, cos, 0],
                                [0,    0,   1]])
    return rotation_matrix


def project_polygon(polygon, xy_res=3, z_res=30, scale=.9):
    m = polygon.shape[0] - 1
    projected_polygon = np.ndarray((0, 2))
    for i, point in enumerate(polygon):
        next_point = polygon[(i + 1) % m]  #wrap around
        line_x = np.linspace(scale * point[0], scale * next_point[0], xy_res)
        line_y = np.linspace(scale * point[1], scale * next_point[1], xy_res)
        pts = np.vstack([line_x, line_y]).T

        projected_polygon = np.vstack([projected_polygon, pts])
        x_vector = np.tile(np.linspace(-.5, .5, z_res)[:, np.newaxis],
                           projected_polygon.shape[0]).reshape(-1,1)
        # XXX This is ugly!
    return np.hstack([x_vector,
                      np.tile(projected_polygon.T, np.linspace(-.5, .5, z_res)[:, np.newaxis].shape[0]).T])


def project_cloud(polygons):
    angle = np.pi / len(polygons)
    rotation_matrix = get_rotation_matrix(angle)
    cloud = np.ndarray((0,3))
    for polygon in polygons:
        interpolated_polygon = project_polygon(polygon)
        cloud = np.vstack([cloud, interpolated_polygon])
        cloud = cloud.dot(rotation_matrix)
    return cloud


def whittle(cloud, polygons):
    """
    Parameters
    ----------
    cloud : ndarray
        Mx3 array containing the point cloud. Columns are x, y, and z
        Should be centered about the origin
    polygons : list of ndarray
        Each array is Mx2 and represents a series of x, y points defining a
        polygon.

    Returns
    -------
    cloud : ndarray
        Mx3 array representing the points from the original cloud that are
        contained inside the 3d volume defined by the series of polygons
    """
    # Divide half the circle evenly among the polygons
    angle = np.pi / len(polygons)
    rotation_matrix = get_rotation_matrix(angle)
    for polygon in polygons:
        # Find and remove the points that fall outside the current projection
        mask = mlab.inside_poly(cloud[:, 1:], polygon)
        cloud = cloud[mask]
        1/0
        # Rotate the cloud and go to the next projection
        cloud = cloud.dot(rotation_matrix)
    return cloud


def get_global_min_max(polygons):
    min_x = min_y = np.inf
    max_x = max_y = -np.inf
    for polygon in polygons:
        min_x = min(polygon.min(0)[0], min_x)
        min_y = min(polygon.min(0)[1], min_y)
        max_x = max(polygon.max(0)[0], max_x)
        max_y = max(polygon.max(0)[1], max_y)
    return ((min_x, min_y), (max_x, max_y))


def center_normalize_polygon(polygon, min=None, max=None):
    if min is None:
        min_x, min_y = polygon.min(0)
        max_x, max_y = polygon.max(0)
    else:
        min_x, min_y = min
        max_x, max_y = max
    polygon[:, 0] -= min_x
    polygon[:, 1] -= min_y
    polygon[:, 0] /= max_x
    polygon[:, 1] /= max_y
    polygon -= 0.5
    return polygon


# Polygons to be fed to wittle algorithm
polygons = [np.array([[1, 0], [0,-1], [-1, 0], [0, 1]]),
            np.array([[1, 0], [0,-1], [-1, 0], [0, 1]]),]
print 'hiiii'
tree_polys = [np.load('4.npy'),
              np.load('3.npy'),
              np.load('2.npy'),
              np.load('1.npy')]

# Ensure that all polygons are normalized with the same constants
min, max = get_global_min_max(tree_polys)
tree_polys = [center_normalize_polygon(p, min, max) for p in tree_polys]

#Start with a random cloud of points
rand_cloud = (np.random.random((300000, 3)) * 2) - 1
whittled = whittle(rand_cloud, tree_polys)
plot_3d(whittled, s=50, show=True)
