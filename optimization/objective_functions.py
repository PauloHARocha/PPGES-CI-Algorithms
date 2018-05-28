import math


class Sphere:
    def __init__(self):
        self.function = sphere
        self.minf = -100
        self.maxf = 100

    def __str__(self):
        return "{} {} {}".format("Sphere", self.minf, self.maxf)

class Rastrigin:
    def __init__(self):
        self.function = rastrigin
        self.minf = -5.12
        self.maxf = 5.12

    def __str__(self):
        return "{} {} {}".format("Rastrigin", self.minf, self.maxf)


class Rosenbrock:
    def __init__(self):
        self.function = rosenbrock
        self.minf = -30
        self.maxf = 30

    def __str__(self):
        return "{} {} {}".format("Rosenbrock", self.minf, self.maxf)


def sphere(x):# Funcao Sphere
    return sum(i ** 2 for i in x)


def rastrigin(x):# Funcao Rastrigin
    return 10 * len(x) + sum((i ** 2) - 10 * math.cos(2 * math.pi * i) for i in x)


def rosenbrock(x):# Funcao Rosenbrock
    return sum(100 * ((x[i+1] - x[i] ** 2) ** 2) + (x[i] - 1) ** 2 for i in range(len(x)-1))
