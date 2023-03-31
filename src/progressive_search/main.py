from random import shuffle
import numpy as np
from typing import Callable, List, Union


class Hypercube:
    def __init__(self, resolution_degree: int, start_point: np.array, parent=None):
        self.resolution_degree = resolution_degree
        self.start_indexes = start_point
        self.parent = parent

        self.side = 2 ** resolution_degree - 1
        self.dim = self.start_indexes.shape[0]

        self.node_order_graph = None
        self.set_node_order_graph()

    def set_node_order_graph(self):
        self.node_order_graph = []

        schema_string = '{0:0' + str(self.dim) + 'b}'

        for node_number in range(2 ** self.dim):
            binary_mask = schema_string.format(node_number)

            point = np.array(self.start_indexes)

            for index, flag in enumerate(binary_mask):
                # print(f'{index=} {flag=}')
                if flag == '1':
                    point[index] += self.side

            self.node_order_graph.append(point)

    def split(self):
        for node in self.node_order_graph:
            yield Hypercube(resolution_degree=self.resolution_degree - 1,
                            start_point=np.around((node + self.start_indexes) / 2).astype(int),
                            parent=self)

    def compute_curr_depth(self):
        res = 0
        if self.parent is None:
            return res
        else:
            parent = self.parent
            while parent is not None:
                res += 1
                parent = parent.parent

            return res


# TODO: посчитать число невычисленных точек, а не только количество коллизий

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

        self.array = np.arange(start=self.left_boundary, stop=self.right_boundary + 1).astype(int)

    def __next__(self):
        pass


class String(Varianta):
    def __init__(self, name: str, necessary_values: List[str]):
        pass


class ProgressiveGridSearch:
    def __init__(self, func: Callable, params: List, stop_criterion: Callable, max_resolution_degree: int = 10):
        self.func = func
        self.params = params
        self.stop_criterion = stop_criterion
        self.max_resolution_degree = max_resolution_degree

        self.number_of_nodes = 2 ** max_resolution_degree

        self.dim = 0
        self.grids = []

        for param in self.params:
            if type(param) is Real:
                self.dim += 1
                grid = np.linspace(param.left_boundary, param.right_boundary, num=self.number_of_nodes)
                self.grids.append(grid)
            elif type(param) is Integer:
                self.dim += 1
                grid = np.arange(param.left_boundary, param.left_boundary + self.number_of_nodes)
                # Временно игнорирую right_boundary, мб буду if'ом фильтровать при вычислении
                self.grids.append(grid)

        self.mask = np.zeros([self.number_of_nodes] * self.dim).astype(bool)
        self.values = np.full_like(self.mask, fill_value=np.nan).astype(float)

        self.nodes_queue = None
        self.curr_hypercube = None
        self.number_of_functions_calls = None

        # number_of_last_level_hypercubes = 2 ** (self.dim * self.max_resolution_degree)
        #
        # self.meta_optimizer = ProgressiveGridSearch(func=self.func,
        #                                             params=[Integer('index', left_boundary=0, right_boundary=number_of_last_level_hypercubes - 1)],
        #                                             stop_criterion=None,
        #                                             max_resolution_degree=self.max_resolution_degree)

    def __iter__(self):
        hypercube = Hypercube(resolution_degree=self.max_resolution_degree, start_point=np.zeros(self.dim).astype(int))

        self.generator = (curr_hc.node_order_graph for curr_hc in self.append_generator(hypercube))
        self.number_of_functions_calls = 0

        self.nodes_queue = iter([])
        return self

    def __next__(self):
        try:
            node_indexes = next(self.nodes_queue)
            if self.mask[tuple(node_indexes)].item() is True:
                return next(self)

            node_point = []

            for k, index in enumerate(node_indexes):
                node_point.append(self.grids[k][index])

            node_point = np.array(node_point)

            if self.curr_hypercube.resolution_degree != self.max_resolution_degree and self.curr_hypercube.resolution_degree != 1:
                if np.any(node_indexes % 2 == 0) and np.sum(node_indexes) % 2 == 0 and np.sum(node_point) != 0:
                    return next(self)

                b = node_indexes - self.curr_hypercube.start_indexes

                if np.any(b % 2 == 0) and np.all(node_indexes != 0):
                    return next(self)

            self.mask[tuple(node_indexes)] = True
            self.number_of_functions_calls += 1

            return node_point, node_indexes
        except StopIteration:
            self.nodes_queue = iter(next(self.generator))
            return next(self)

    def append_generator(self, hypercube: Hypercube):
        lst = [hypercube]

        while lst:

            self.curr_hypercube = lst.pop(0)
            yield self.curr_hypercube

            if self.curr_hypercube.resolution_degree == 1:
                continue

            for sub_elem in self.curr_hypercube.split():
                lst.append(sub_elem)

    # Здесь будет расположен сам цикл поиска минимума/максимума функции на сетке с критериями останова
    def optimize(self):
        for point in self:
            raise NotImplementedError


# TODO: Счетчик коллизий и пропущенных узлов


if __name__ == '__main__':
    s = ProgressiveGridSearch(lambda x: x,
                              params=[Integer('k', left_boundary=-5, right_boundary=10)],
                              stop_criterion=None)

    s.optimize()
