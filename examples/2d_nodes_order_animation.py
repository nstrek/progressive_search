from src.progressive_search.main import ProgressiveGridSearch, Real, Integer, String
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

mpl.use('TkAgg')

MAX_DEGREE = 6

s = ProgressiveGridSearch(func=lambda x: x,
                          params=[Integer('row', left_boundary=-5, right_boundary=10),
                                  Integer('column', left_boundary=10, right_boundary=100)],
                          stop_criterion=None,
                          max_resolution_degree=MAX_DEGREE)


def node_order_generator():
    for node_point, node_indexes in s:
        yield node_indexes


x_left, x_right = -3, 2 ** MAX_DEGREE
y_left, y_right = -3, 2 ** MAX_DEGREE

fig, axes = plt.subplots(ncols=2)
title = plt.suptitle(t='', fontsize=20)
ln, = axes[0].plot([], [])
ln, = axes[1].plot([], [])


def init():
    axes[0].set_xlim(x_left * 1.05, x_right * 1.05)
    axes[0].set_ylim(y_left * 1.05, y_right * 1.05)

    axes[1].set_xlim(x_left * 1.05, x_right * 1.05)
    axes[1].set_ylim(y_left * 1.05, y_right * 1.05)

    return ln, ln

xs = []
ys = []


def update(point):

    # title.set_text(f'{len(frame)}')

    a = axes[0].scatter(point[0], point[1], color='red')

    xs.append(point[0])
    ys.append(point[1])

    b = axes[1].scatter(xs, ys, color='blue')

    return a, b


ani = FuncAnimation(fig, update, frames=node_order_generator(), init_func=init, blit=True, repeat=False)
plt.show()

# f = r"D:\PycharmProjects\progressive_search\for_article\2d_animation_bad_order.gif"
# ani.save(f, writer='Pillow', fps=30)
