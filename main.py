import math
import os
import random
import multiprocessing as mp
import numpy


PRINT_SLICE_INFO = False
PRINT_ITERATION_NO = True
PRINT_ITERATION_BEST_ANSWER = True
PRINT_ITERATION_BEST_ANSWER_DETAILS = False
PRINT_ITERATION_ALL_ANSWERS = True


class TSP:
    EDGE_WEIGHT_TYPE_EXPLICIT = 'EXPLICIT'
    EDGE_WEIGHT_TYPE_GEO = 'GEO'
    EDGE_WEIGHT_TYPE_EUC_2D = 'EUC_2D'

    def __init__(self, path: str):
        self.dimension = None
        self.edge_with_type = None
        self.distances = None

        with open(path) as f:
            lines = f.readlines()
        lines = list(map(lambda line: line.strip(), lines))
        self.__parse(lines)

    def __parse(self, lines: list):
        def get_value(key):
            line = next(line for line in lines if line.startswith(f'{key}'))
            return line[line.find(':') + 1:].strip()

        self.dimension = int(get_value('DIMENSION'))
        self.edge_with_type = get_value('EDGE_WEIGHT_TYPE')

        if self.edge_with_type == self.EDGE_WEIGHT_TYPE_EXPLICIT:
            self.__parse_explicit(lines)
        elif self.edge_with_type == self.EDGE_WEIGHT_TYPE_GEO:
            self.__parse_geo(lines)
        elif self.edge_with_type == self.EDGE_WEIGHT_TYPE_EUC_2D:
            self.__parse_euc_2d(lines)

    def __parse_explicit(self, lines: list):
        start = next(i for i in range(len(lines)) if lines[i] == 'EDGE_WEIGHT_SECTION') + 1
        raw_distances = list(map(lambda line: line.split(), lines[start: start + self.dimension - 1]))
        self.distances = []
        for i in range(self.dimension):
            row = []
            for j in range(i):
                row.append(self.distances[j][i])
            row.append(0)
            for j in range(i, self.dimension - 1):
                row.append(int(raw_distances[i][j - i]))
            self.distances.append(row)

    def __parse_geo(self, lines: list):
        self.__parse_euc_2d(lines)  # It's not correct bt anyway, it is what it is

    def __parse_euc_2d(self, lines: list):
        start = next(i for i in range(len(lines)) if lines[i] == 'NODE_COORD_SECTION') + 1
        raw_nodes = list(map(lambda line: list(map(lambda x: math.floor(float(x)), line.split())),
                         lines[start: start + self.dimension]))
        nodes = [None] * self.dimension
        for raw_node in raw_nodes:
            nodes[raw_node[0] - 1] = (raw_node[1], raw_node[2])
        self.distances = [[None for x in range(self.dimension)] for y in range(self.dimension)]
        for i in range(self.dimension):
            for j in range(self.dimension):
                self.distances[i][j] = math.floor(math.sqrt(
                    (nodes[i][0] - nodes[j][0]) ** 2 +
                    (nodes[i][1] - nodes[j][1]) ** 2
                ))


class Chromosome:
    def __init__(self):
        global tsp
        self.__data = list(range(tsp.dimension))
        random.shuffle(self.__data)
        self.__cached_cost = 0
        self.__cache_is_valid = False

    def __str__(self):
        return self.__data.__str__() + ': ' + str(self.cost())

    def __lt__(self, other):
        return self.cost() > other.cost()

    def __mul__(self, other):
        global tsp
        (side1, side2) = random.sample(range(tsp.dimension + 1), 2)

        start = min(side1, side2)
        end = max(side1, side2)
        if PRINT_SLICE_INFO:
            print(start, end)

        first_child = Chromosome()
        first_child.__data = self.__crossover(self.__data, other.__data, start, end)

        second_child = Chromosome()
        second_child.__data = self.__crossover(other.__data, self.__data, start, end)

        return [first_child, second_child]

    def __invert__(self):
        global tsp

        result = Chromosome()
        result.__data = self.__data

        count = random.randint(0, MUTATION_DEGREE)
        for _ in range(count):
            (src, dst) = random.sample(range(tsp.dimension), 2)
            if PRINT_SLICE_INFO:
                print(src, dst)
            v = result.__data[src]
            result.__data = result.__data[:src] + result.__data[src + 1:]
            result.__data = result.__data[:dst] + [v] + result.__data[dst:]

        return result

    def __sub__(self, other):
        return other.cost() - self.cost()

    def cost(self):
        global tsp
        if not self.__cache_is_valid:
            self.__cached_cost = 0
            for i in range(len(self.__data)):
                self.__cached_cost += tsp.distances[self.__data[i - 1]][self.__data[i]]
            self.__cache_is_valid = True
        return self.__cached_cost

    @staticmethod
    def __crossover(mother_data: list, father_data: list, start: int, end: int):
        dimension = len(mother_data)
        data = [None] * dimension
        data[start:end] = mother_data[start:end]
        i = end
        for v in father_data[end:] + father_data[:end]:
            if v not in data:
                if i == start:
                    i = end
                if i == dimension:
                    i = 0
                data[i] = v
                i += 1
        return data


class Population:
    def __init__(self, countOrData):
        if type(countOrData) == int:
            self.__data = [Chromosome() for _ in range(countOrData)]
        elif type(countOrData) == list:
            self.__data = countOrData
        else:
            raise Exception()
        self.__data.sort()

    def iterate(self):
        children = self.__crossover()
        children.__mutate()
        self.__replacement(children)

    def __choose(self):
        n = len(self.__data)
        roulette = sum([[i] * (i + 1) for i in range(n)], [])
        turning = random.randint(0, n)
        roulette = roulette[turning:] + roulette[:turning]
        pointers = range(0, len(roulette), math.ceil(len(roulette) / n))

        choices = []
        for pointer in pointers:
            choices.append(self.__data[roulette[pointer]])

        return choices

    def __crossover(self):
        parents = self.__choose()
        random.shuffle(parents)

        P_COUNT = os.cpu_count()

        def pair_chunk_calculator(i, pair_chunk, rd):
            rd[i] = (sum([pair[0] * pair[1] for pair in pair_chunk], []))

        pair_chunks = numpy.array_split([[parents[i], parents[i + 1]] for i in range(0, len(parents) - 1, 2)], P_COUNT)
        manager = mp.Manager()
        rd = manager.dict()
        processes = [mp.Process(
            target=pair_chunk_calculator,
            args=(i, pair_chunks[i], rd)
        ) for i in range(P_COUNT)]

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        return Population(sum(rd.values(), []))

    def __mutate(self):
        for child in self.__data:
            child = ~child

    def __replacement(self, children):
        n = len(children.__data)
        best_children_count = math.floor(REPLACEMENT[0] * n)
        other_children_count = math.floor(REPLACEMENT[1] * n)
        other_parents_count = math.floor(REPLACEMENT[2] * n)
        best_parents_count = n - best_children_count - other_children_count - other_parents_count
        self.__data = (
            children.__data[-best_children_count:] +
            random.sample(children.__data[:(n - best_children_count)], other_children_count) +
            random.sample(self.__data[:(n - best_parents_count)], other_parents_count) +
            self.__data[-best_parents_count:]
        )
        self.__data.sort()

    def escape_stagnancy(self, proportion: float):
        count = math.floor(len(self.__data) * proportion)
        self.__data[:count] = [Chromosome() for _ in range(count)]
        self.__data.sort()

    def answer(self) -> Chromosome:
        return self.__data[-1]

    def answers(self) -> list:
        return list(map(lambda c: c.cost(), self.__data))


BAYG29 = 'testcase.bayg29.tsp'
GR229 = 'testcase.gr229.tsp'
PR1002 = 'testcase.pr1002.tsp'

CHOSEN = PR1002

tsp = TSP(CHOSEN)

if CHOSEN == BAYG29:
    N = 240
    MUTATION_DEGREE = 9
    REPLACEMENT = [.1, .4, .4]
    STAGNANCY_ESCAPE_DEGREE = 2
    STAGNANCY_ESCAPE_PROPORTION = 0.9
    IMPROVEMENT_THRESHOLD = 1
    STAGNANCY_THRESHOLD = 40
elif CHOSEN == GR229:
    N = 240
    MUTATION_DEGREE = 9
    REPLACEMENT = [.1, .4, .4]
    STAGNANCY_ESCAPE_DEGREE = 2
    STAGNANCY_ESCAPE_PROPORTION = 0.9
    IMPROVEMENT_THRESHOLD = 1
    STAGNANCY_THRESHOLD = 40
elif CHOSEN == PR1002:
    N = 300
    MUTATION_DEGREE = 9
    REPLACEMENT = [.1, .4, .4]
    STAGNANCY_ESCAPE_DEGREE = 2
    STAGNANCY_ESCAPE_PROPORTION = 0.9
    IMPROVEMENT_THRESHOLD = 1_000
    STAGNANCY_THRESHOLD = 8

ac = 1_000_000
while ac >= 1_000_000:
    population = Population(N)
    answer = population.answer()
    stagnancy = 0
    i = 0
    while True:
        population.iterate()
        improvement = population.answer() - answer
        answer = population.answer()

        if PRINT_ITERATION_NO:
            print(f"Iteration: {i}")
        if PRINT_ITERATION_BEST_ANSWER:
            print(f"Best Answer: {population.answer().cost()}")
        if PRINT_ITERATION_BEST_ANSWER_DETAILS:
            print(population.answer())
        if PRINT_ITERATION_ALL_ANSWERS:
            print(f"All Answers: {population.answers()}")

        if improvement < IMPROVEMENT_THRESHOLD:
            stagnancy += 1
            if stagnancy == math.floor(STAGNANCY_THRESHOLD / 2):
                MUTATION_DEGREE = math.floor(MUTATION_DEGREE * STAGNANCY_ESCAPE_DEGREE)
            if stagnancy >= math.floor(STAGNANCY_THRESHOLD / 2):
                population.escape_stagnancy(STAGNANCY_ESCAPE_PROPORTION)
            if stagnancy >= STAGNANCY_THRESHOLD:
                break
        else:
            stagnancy = 0

        i += 1

    print(population.answer())
    ac = population.answer().cost()
