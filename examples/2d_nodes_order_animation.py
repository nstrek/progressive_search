import numpy as np

from src.progressive_search.main import ProgressiveGridSearch, Real, Integer, String
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

mpl.use('TkAgg')

MAX_DEGREE = 4

s = ProgressiveGridSearch(func=lambda x: x,
                          params=[Integer('row', left_boundary=0, right_boundary=2 ** MAX_DEGREE - 1),
                                  Integer('column', left_boundary=0, right_boundary=2 ** MAX_DEGREE - 1)],
                          stop_criterion=None,
                          max_resolution_degree=MAX_DEGREE)


def node_order_generator():
    for node_point, node_indexes in s:
        yield node_indexes


image_matrix = np.zeros_like(s.mask).astype(int)
image_matrix2 = np.zeros_like(s.mask).astype(int)


x_left, x_right = 0, 2 ** MAX_DEGREE - 1
y_left, y_right = 0, 2 ** MAX_DEGREE - 1

fig, ax = plt.subplots(ncols=1)
title = plt.suptitle(t='', fontsize=20)
ln, = ax.plot([], [])#axes[0].plot([], [])
# ln, = axes[1].plot([], [])


ax.xaxis.set_tick_params(labelbottom=False)
ax.yaxis.set_tick_params(labelleft=False)

ax.set_xticks([])
ax.set_yticks([])


def init():
    # axes[0].set_xlim(x_left * 1.05, x_right * 1.05)
    # axes[0].set_ylim(y_left * 1.05, y_right * 1.05)

    # axes[1].set_xlim(x_left * 1.05, x_right * 1.05)
    # axes[1].set_ylim(y_left * 1.05, y_right * 1.05)

    return ln,#, ln

xs = []
ys = []


def update(point):
    # image_matrix[point[0], point[1]] += 1
    #
    # a = axes[0].imshow(image_matrix, vmin=0, vmax=10, cmap='hot')

    image_matrix2[point[0], point[1]] = 2
    c = ax.imshow(image_matrix2, vmin=0, vmax=2, cmap='hot')
    image_matrix2[point[0], point[1]] = 1

    return c,


ani = FuncAnimation(fig, update, frames=node_order_generator(), init_func=init, blit=True, repeat=False)
# plt.show()


import pathlib

path = pathlib.Path(__file__).parent.parent.resolve()
print(path)

f = fr"{path}\images\2d_nodes_order_animation_fractal_{MAX_DEGREE}.gif"
ani.save(f, writer='Pillow', fps=30)
