from typing import Union, List, Callable, Dict
import numpy as np
import itertools


class Varianta:  # or Variable
    def __init__(self, name: str, var_type: str,
                 necessary_values: Union[List[int], List[float], List[str], None] = None,
                 left_boundary: Union[int, float, None] = None, right_boundary: Union[int, float, None] = None):
        raise NotImplementedError()

    def __next__(self):
        raise NotImplementedError()


class Real(Varianta):
    def __init__(self, name: str, left_boundary: float, right_boundary: float, max_resolution_degree: int = 10):

        if not (right_boundary - left_boundary > 0):
            raise ValueError(
                f'right_boundary is required to be greater than the left_boundary, but your input is {left_boundary=} and {right_boundary=}')

        if not (max_resolution_degree > 0):
            raise ValueError(
                f'max_resolution_degree is required to be greater than zero, but your input is {max_resolution_degree=}')

        self.name = name
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary
        self.max_resolution_degree = max_resolution_degree

        self.prev_resolution_nodes_pairs = None
        self.curr_resolution_nodes_pairs = None
        self.curr_resolution_degree = None

    def __len__(self):
        return 2 ** self.max_resolution_degree

    def __next__(self):
        # Наследники Varianta выдают узлы в нужном порядке.
        # Мб в __iter__ происходит повышение разрешения, это я размышляю над многомерным циклом в ProgressiveGridSearch

        self.curr_resolution_degree += 1

        if self.curr_resolution_degree == 1:
            # return {'curr_resolution_degree': self.curr_resolution_degree,
            #         'nodes': [self.left_boundary, self.right_boundary]}
            return [self.left_boundary, self.right_boundary]

        if self.curr_resolution_degree > self.max_resolution_degree:
            raise StopIteration

        nodes = []

        for node in self.prev_resolution_nodes_pairs:
            c = (node[0] + node[1]) / 2
            nodes.append(c)
            self.curr_resolution_nodes_pairs.extend([(node[0], c), (c, node[1])])

        self.prev_resolution_nodes_pairs = self.curr_resolution_nodes_pairs
        self.curr_resolution_nodes_pairs = []
        # return {'curr_resolution_degree': self.curr_resolution_degree, 'nodes': nodes}
        return nodes

    def __iter__(self):
        self.prev_resolution_nodes_pairs = [(self.left_boundary, self.right_boundary)]  # Как их прокачать через next?
        self.curr_resolution_nodes_pairs = []
        self.curr_resolution_degree = 0

        return self
        # self.prev_resolution_nodes


class Integer(Varianta):
    def __init__(self, name: str, left_boundary: int, right_boundary: int):
        if not (right_boundary - left_boundary > 0):
            raise ValueError(
                f'right_boundary is required to be greater than the left_boundary, but your input is {left_boundary=} and {right_boundary=}')

        self.name = name
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary

        degree = 1
        while (right_boundary - left_boundary) + 1 > 2 ** degree:
            degree += 1

        self.max_resolution_degree = degree

        self.array = np.arange(start=self.left_boundary, stop=self.right_boundary + 1, dtype=np.int)

    def __next__(self):
        pass


class String(Varianta):
    def __init__(self, name: str, necessary_values: List[str]):
        pass


class ProgressiveGridSearch:
    def __init__(self, func: Callable, stop_criterion: Union[str, Callable], params: List[Varianta],
                 max_resolution_degree: int = 10):
        # self.t0  # Время запуска чтобы можно было останавливать по времени.
        pass
        # для целых переменных брать степень такую, что 2^(degree+1) > (right - left). То есть не будем перебирать все значения, а можно флаг поставить режима
        # И этот флаг ставить в Varianta, как и max_resolution_degree для вещественных (там можно сменить на язык чисел после запятой, а можно и не менять)
        # Нет, все-таки придется брать degree: 2^degree > (right - left) и фильтровать узлы больше right


# Критерии останова:
# 1) По времени
# 2) По изменению значения меньше, чем на дельту за итерацию
# 3) По количеству вычислений

def f(alpha, degree):  # Обязательно именованные аргументы
    return alpha ** degree


# timer_criterion = lambda t0: t0 > 60 * 60
#
# solution = ProgressiveGridSearch(func=f, stop_criterion=timer_criterion,
#                                  params=[Real('alpha', left_boundary=-5.5, right_boundary=5.7),
#                                          Real('beta', left_boundary=3.0, right_boundary=15.98),
#                                          Integer('degree', left_boundary=-3, right_boundary=9),
#                                          String('model', necessary_values=['direct', 'reverse'])])


if __name__ == '__main__':
    r = Real('rrr', left_boundary=-5.0, right_boundary=5.0)
    t = Real('ttt', left_boundary=0.0, right_boundary=16.0)

    variables = [r, t]

    # for node in r:
    #     print(r)

    # for var in variables:
    #
    #
    # for vec in itertools.product(variables)

    for elem in r:
        print(elem)

    print('\n\n\n')

    iter(t)
    iter(r)

    for _ in range(4):
        r_nodes = next(r)
        t_nodes = next(t)

        for vec in itertools.product(*[r_nodes, t_nodes]):
            print(vec)


