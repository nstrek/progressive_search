from src.progressive_search.main import Real
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import itertools

mpl.use('TkAgg')

# nodes_to_update - параметр задающий частоту обновления графика

x_left, x_right = -5.0, 5.0
y_left, y_right = -5.0, 5.0

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


def generator():
    x = Real('x', left_boundary=x_left, right_boundary=x_right)
    y = Real('y', left_boundary=y_left, right_boundary=y_right)

    iter(x), iter(y)

    for degree in range(1, 7):#x.max_resolution_degree):
        x_nodes = next(x)
        y_nodes = next(y)

        print(degree)

        for point in itertools.product(x_nodes, y_nodes):
            yield point

        print(len(xs), len(ys))
xs = []
ys = []

def update(frame):
    # print(frame[0])
    # axes[0].set_title(f'{frame[0]}')
    # fig.suptitle(f'{frame[0]}')

    title.set_text(f'{len(frame)}')

    a = axes[0].scatter(frame[0], frame[1], color='red')

    xs.append(frame[0])
    ys.append(frame[1])

    b = axes[1].scatter(xs, ys, color='blue')

    return a, b


ani = FuncAnimation(fig, update, frames=generator(), init_func=init, blit=True, repeat=False)
# plt.show()

from matplotlib import animation

# f = r"D:\PycharmProjects\progressive_plots\for_article\anim.html"
# writergif = animation.HTMLWriter(fps=30)
# ani.save(f, writer=writergif)
f = r"D:\PycharmProjects\progressive_search\for_article\2d_animation_bad_order.gif"
ani.save(f, writer='Pillow', fps=30)
