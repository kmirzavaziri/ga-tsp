import math
import random

VERBOSE_LEVEL = 1


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
        def get_value(key): return next(
            line for line in lines if line.startswith(f'{key}:'))[len(f'{key}:'):].strip()

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
        pass

    def __parse_euc_2d(self, lines: list):
        pass


class Chromosome:
    def __init__(self, tsp: TSP):
        self.tsp = tsp
        self.__data = list(range(tsp.dimension))
        random.shuffle(self.__data)
        self.__cached_cost = 0
        self.__cache_is_valid = False

    def __str__(self):
        return self.__data.__str__() + ': ' + str(self.cost())

    def __lt__(self, other):
        return self.cost() > other.cost()

    def __mul__(self, other):
        if (type(other) != Chromosome):
            raise Exception('Cannot crossover a with a non-chromosome.')

        if (self.tsp != other.tsp):
            raise Exception('Cannot crossover a with a chromosome of another tsp.')

        (side1, side2) = random.sample(range(self.tsp.dimension + 1), 2)

        start = min(side1, side2)
        end = max(side1, side2)
        if VERBOSE_LEVEL > 1:
            print(start, end)

        first_child = Chromosome(self.tsp)
        first_child.__data = self.__crossover(self.__data, other.__data, start, end)

        second_child = Chromosome(self.tsp)
        second_child.__data = self.__crossover(other.__data, self.__data, start, end)

        return [first_child, second_child]

    def __invert__(self):
        (src, dst) = random.sample(range(self.tsp.dimension), 2)
        if VERBOSE_LEVEL > 1:
            print(src, dst)

        result = Chromosome(self.tsp)
        result.__data = self.__data
        v = result.__data[src]
        result.__data = result.__data[:src] + result.__data[src + 1:]
        result.__data = result.__data[:dst] + [v] + result.__data[dst:]

        return result

    def __sub__(self, other):
        return other.cost() - self.cost()

    def cost(self):
        if not self.__cache_is_valid:
            self.__cached_cost = 0
            for i in range(len(self.__data)):
                self.__cached_cost += self.tsp.distances[self.__data[i - 1]][self.__data[i]]
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


class Population(list):
    def __init__(self, tsp: TSP, countOrData):
        if type(countOrData) == int:
            self.__data = [Chromosome(tsp) for i in range(countOrData)]
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
        children = []
        for i in range(0, len(parents) - 1, 2):
            children += parents[i] * parents[i + 1]
        return Population(None, children)

    def __mutate(self):
        for child in self.__data:
            if random.random() < MUTATION_PROBABILITY:
                child = ~child

    def __replacement(self, children):
        n = len(children.__data)
        children_count = math.floor(REPLACEMENT_CHILDREN_PROPORTION * n)
        parents_count = n - children_count
        self.__data = children.__data[-children_count:] + self.__data[-parents_count:]
        self.__data.sort()

    def answer(self) -> Chromosome:
        return self.__data[-1]

    def answers(self) -> list:
        return list(map(lambda c: c.cost(), self.__data))


BAYG29 = 'testcase.bayg29.tsp'

tsp = TSP(BAYG29)

N = 500
MUTATION_PROBABILITY = .8
REPLACEMENT_CHILDREN_PROPORTION = .2

IMPROVEMENT_THRESHOLD = 10
STAGNANCY_THRESHOLD = 10

population = Population(tsp, N)
answer = population.answer()
stagnancy = 0
i = 0
while True:
    population.iterate()
    improvement = population.answer() - answer
    answer = population.answer()

    if VERBOSE_LEVEL > 0:
        print(f"Iteration: {i}")
        print(f"Best Answer: {population.answer()}")
    if VERBOSE_LEVEL > 1:
        print(f"All Answers: {population.answers()}")

    if improvement < IMPROVEMENT_THRESHOLD:
        stagnancy += 1
        if stagnancy >= STAGNANCY_THRESHOLD:
            break
    else:
        stagnancy = 0

    i += 1

if VERBOSE_LEVEL == 0:
    print(population.answer())
