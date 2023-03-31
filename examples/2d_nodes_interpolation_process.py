import numpy as np

from src.progressive_search.main import ProgressiveGridSearch, Real, Integer, String
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
from scipy.interpolate import LinearNDInterpolator
import pathlib

path = pathlib.Path(__file__).parent.parent.resolve()

mpl.use('TkAgg')


MAX_DEGREE = 7

img_index = 1

if img_index == 0:

    img = Image.open(rf"{path}\images\Python-logo.png").convert('L')
    img = img.crop((245, 145, 555, 455))

    img = img.resize((2 ** MAX_DEGREE, 2 ** MAX_DEGREE), resample=Image.Resampling.BILINEAR)
    # img.show()
    img_arr = np.array(img)

elif img_index == 1:

    img = Image.open(rf"{path}\images\averaged_women_face.jpg").convert('L')
    # img = img.crop((245, 145, 555, 455))

    img = img.resize((2 ** MAX_DEGREE, 2 ** MAX_DEGREE), resample=Image.Resampling.BILINEAR)
    # img.show()
    img_arr = np.array(img)


print(img_arr.shape, img_index)

# quit()

def func(x, y):
    try:
        return img_arr[x, y]
    except IndexError:
        return 0



optimizer = ProgressiveGridSearch(func=lambda x: x,
                                  params=[Integer('row', left_boundary=0, right_boundary=2 ** MAX_DEGREE - 1),
                                  Integer('column', left_boundary=0, right_boundary=2 ** MAX_DEGREE - 1)],
                                  stop_criterion=None,
                                  max_resolution_degree=MAX_DEGREE)


def node_optimal_order_generator():
    for node_point, node_indexes in optimizer:
        yield node_indexes


def node_bruteforce_order_generator():
    for i in range(2 ** MAX_DEGREE):
        for j in range(2 ** MAX_DEGREE):
            yield np.array([i, j])


def node_generator():
    a = node_optimal_order_generator()
    b = node_bruteforce_order_generator()

    for node_indexes in a:
        yield node_indexes, next(b)


progressive_matrix = np.zeros_like(optimizer.mask).astype(int)
naive_matrix = np.zeros_like(optimizer.mask).astype(int)


fig, axs = plt.subplots(nrows=2, ncols=1, constrained_layout=True)

# clear subplots
for ax in axs:
    ax.remove()

# add subfigure per subplot
gridspec = axs[0].get_subplotspec().get_gridspec()
subfigs = [fig.add_subfigure(gs) for gs in gridspec]

axes = [[None, None], [None, None]]

for row, subfig in enumerate(subfigs):

    # create 1x3 subplots per subfig
    axs = subfig.subplots(nrows=1, ncols=2)
    for col, ax in enumerate(axs):

        axes[row][col] = ax

subfigs[0].suptitle('GridSearch process')

subfigs[1].suptitle('ProgressiveGridSearch process with interpolation')

for i in range(2):
    for j in range(2):
        axes[i][j].xaxis.set_tick_params(labelbottom=False)
        axes[i][j].yaxis.set_tick_params(labelleft=False)

        axes[i][j].set_xticks([])
        axes[i][j].set_yticks([])


def generator():
    node_gen = node_generator()

    naive_image = np.zeros_like(naive_matrix)

    progressive_image = np.zeros_like(progressive_matrix)

    for progressive, naive in node_gen:
        naive_matrix[tuple(naive)] = 2
        # a = axes[0, 0].imshow(naive_matrix, vmin=0, vmax=10, cmap='hot')


        naive_image[tuple(naive)] = func(naive[0], naive[1])

        # b = axes[0, 1].imshow(naive_image)

        progressive_matrix[tuple(progressive)] = 2
        # c = axes[1, 0].imshow(progressive_matrix, vmin=0, vmax=2, cmap='hot')


        # x_arr = np.arange(2 ** MAX_DEGREE)
        # y_arr = np.arange(2 ** MAX_DEGREE) + 2 ** MAX_DEGREE
        # data_arr = np.zeros((2 ** MAX_DEGREE, 2 ** MAX_DEGREE))
        # xs, ys, zs = [], [], []
        #
        # indexes = np.argwhere(optimizer.mask == True)
        #
        # for x_index, y_index in indexes:
        #     xs.append(x_index)
        #     ys.append(y_index)
        #
        #     zs.append(func(x_index, y_index))
        #
        # xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)

        nc = optimizer.number_of_functions_calls

        if 4 <= nc < 32 or \
                (2 ** 5 <= nc < 2 ** 6 and nc % 2 ** 9 == 0) or \
                (2 ** 6 <= nc < 2 ** 7 and nc % 2 ** 10 == 0) or \
                (2 ** 7 <= nc < 2 ** 8 and nc % 2 ** 11 == 0) or \
                (2 ** 8 <= nc < 2 ** 9 and nc % 2 ** 12 == 0) or \
                (2 ** 9 <= nc < 2 ** 10 and nc % 2 ** 13 == 0) or \
                (2 ** 10 <= nc and nc % 2 ** 10):
            print(nc)
            xs, ys, zs = [], [], []

            indexes = np.argwhere(optimizer.mask == True)

            for x_index, y_index in indexes:
                xs.append(x_index)
                ys.append(y_index)

                zs.append(func(x_index, y_index))

            xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)

            interp = LinearNDInterpolator(list(zip(xs, ys)), zs)

            for i in range(progressive_image.shape[0]):
                for j in range(progressive_image.shape[1]):
                    progressive_image[i, j] = interp(np.array([i, j]))

        yield naive_matrix, naive_image, progressive_matrix, progressive_image

        naive_matrix[tuple(naive)] = 1
        progressive_matrix[tuple(progressive)] = 1


def update(frame):
    fig.suptitle(f'{optimizer.number_of_functions_calls=}')#, horizontalalignment=(0, 0))

    if optimizer.number_of_functions_calls % 50 == 0:
        print(optimizer.number_of_functions_calls)

    naive_matrix, naive_image, progressive_matrix, progressive_image = frame

    a = axes[0][0].imshow(naive_matrix, vmin=0, vmax=2, cmap='hot')
    # a = subfigs[0][0].imshow(naive_matrix, vmin=0, vmax=2, cmap='hot')

    b = axes[0][1].imshow(naive_image, cmap='gray', vmin=0, vmax=255)
    # b = subfigs[0][1].imshow(naive_image, cmap='gray', vmin=0, vmax=255)

    c = axes[1][0].imshow(progressive_matrix, vmin=0, vmax=2, cmap='hot')
    # c = subfigs[1][0].imshow(progressive_matrix, vmin=0, vmax=2, cmap='hot')

    d = axes[1][1].imshow(progressive_image, cmap='gray', vmin=0, vmax=255)
    # d = subfigs[1][1].imshow(progressive_image, cmap='gray', vmin=0, vmax=255)

    return a, b, c, d


ani = FuncAnimation(fig, update, frames=generator(), blit=True, repeat=False)
# plt.show()

f = fr"{path}\images\python_logo_processes_comparison_max_degree_{MAX_DEGREE}_{img_index}.gif"
ani.save(f, writer='Pillow', fps=30)
