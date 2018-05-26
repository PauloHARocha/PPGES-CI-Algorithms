import numpy as np
from optimization.objective_functions import sphere, rastrigin, rosenbrock


# This code was based on in the following references:
# [1] "On clarifying misconceptions when comparing variants of the Artificial FoodSource Colony Algorithm by offering a new
# implementation" published in 2015 by Marjan Mernik, Shih-Hsi Liu, Dervis Karaboga and Matej Crepinsek
# [2] "A modified Artificial FoodSource Colony algorithm for real-parameter optimization" published in 2010 by
# Bahriye Akay and Dervis Karaboga

# This implementation consider #_employed = #_onlookers = #_food_sources = colony_size / 2

class FoodSource(object):
    def __init__(self, dim):
        nan = float('nan')
        self.pos = [nan for _ in range(dim)]
        self.cost = np.inf
        self.fitness = 0.0
        self.prob = 0.0
        self.trials = 0


class ABC(object):
    def __init__(self, objective_function, n_iter=1000, colony_size=30, trials_limit=100):

        self.objective_function = objective_function

        self.dim = 30
        self.minf = -100
        self.maxf = 100
        self.n_iter = n_iter

        self.gbest = None
        self.optimum_cost_tracking_iter = []

        self.num_fs = int(colony_size / 2)
        self.trials_limit = trials_limit
        self.food_sources = []

        self.optimum_cost_tracking_eval = []
        self.optimum_cost_tracking_iter = []

    @staticmethod
    def calculate_fitness(cost):
        if cost >= 0:
            result = 1.0 / (1.0 + cost)
        else:
            result = 1.0 + abs(cost)
        return result

    def calculate_probabilities(self):
        sum_fit = 0.0
        for fs in range(self.num_fs):
            sum_fit += self.food_sources[fs].fitness

        for fs in range(self.num_fs):
            self.food_sources[fs].prob = (self.food_sources[fs].fitness / sum_fit)

    def update_best_solution(self):
        for fs in self.food_sources:
            if fs.cost < self.gbest.cost:
                self.gbest.pos = fs.pos
                self.gbest.cost = fs.cost

    def init_fs(self, pos):
        fs = FoodSource(self.dim)
        fs.pos = pos
        fs.cost = self.objective_function(fs.pos)
        self.optimum_cost_tracking_eval.append(self.gbest.cost)
        fs.fitness = self.calculate_fitness(fs.cost)
        return fs

    def init_colony(self):
        self.gbest = FoodSource(self.dim)
        self.gbest.cost = np.inf

        for i in range(self.num_fs):
            rand = np.random.random(self.dim)

            fs = self.init_fs(self.minf + rand * (self.maxf - self.minf))
            self.food_sources.append(fs)

            if fs.cost < self.gbest.cost:
                self.gbest.pos = fs.pos
                self.gbest.cost = fs.cost

    def employed_bee_phase(self):
        for fs in range(self.num_fs):
            k = range(self.num_fs)
            k = np.random.choice(np.array(k))
            j = np.random.choice(range(self.dim))
            phi = np.random.uniform(-1, 1)

            new_pos = np.copy(self.food_sources[fs].pos)
            new_pos[j] = self.food_sources[fs].pos[j] + phi * (
            self.food_sources[fs].pos[j] - self.food_sources[k].pos[j])

            if new_pos[j] < self.minf:
                new_pos[j] = self.minf
            elif new_pos[j] > self.maxf:
                new_pos[j] = self.maxf
            cost = self.objective_function(new_pos)
            self.optimum_cost_tracking_eval.append(self.gbest.cost)
            fit = self.calculate_fitness(cost)

            if fit > self.food_sources[fs].fitness:
                self.food_sources[fs].pos = new_pos
                self.food_sources[fs].cost = cost
                self.food_sources[fs].fitness = fit
                self.food_sources[fs].trials = 0
            else:
                self.food_sources[fs].trials += 1

    def onlooker_bee_phase(self):
        t = s = 0
        while t < self.num_fs:
            s = (s + 1) % self.num_fs
            r = np.random.uniform()

            if r < self.food_sources[s].prob:
                t += 1

                k = range(self.num_fs)
                # k = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14]
                # k.remove(s)
                k = np.random.choice(np.array(k))
                j = np.random.choice(range(self.dim))
                phi = np.random.uniform(-1, 1)

                new_pos = np.copy(self.food_sources[s].pos)
                new_pos[j] = new_pos[j] + phi * (new_pos[j] - self.food_sources[k].pos[j])

                if new_pos[j] < self.minf:
                    new_pos[j] = self.minf
                elif new_pos[j] > self.maxf:
                    new_pos[j] = self.maxf
                cost = self.objective_function(new_pos)
                self.optimum_cost_tracking_eval.append(self.gbest.cost)
                fit = self.calculate_fitness(cost)

                if fit > self.food_sources[s].fitness and (self.food_sources[s].cost - cost) >= 0.0001:
                    self.food_sources[s].pos = new_pos
                    self.food_sources[s].cost = cost
                    self.food_sources[s].fitness = fit
                    self.food_sources[s].trials = 0
                else:
                    self.food_sources[s].trials += 1

    def get_max_trial(self):
        max_ = 0
        for fs in range(self.num_fs):
            if self.food_sources[fs].trials > self.food_sources[max_].trials:
                max_ = fs
        return max_

    def scout_bee_phase(self):
        max_ = self.get_max_trial()

        if self.food_sources[max_].trials >= self.trials_limit:
            print('scout')
            rand = np.random.random(self.dim)
            pos = self.minf + rand * (self.maxf - self.minf)
            self.food_sources[max_] = self.init_fs(pos)

    def optimize(self):
        self.optimum_cost_tracking_eval = []
        self.optimum_cost_tracking_iter = []

        self.init_colony()
        self.update_best_solution()

        for i in range(self.n_iter):
            self.employed_bee_phase()
            self.calculate_probabilities()
            self.onlooker_bee_phase()
            self.update_best_solution()
            self.scout_bee_phase()
            self.optimum_cost_tracking_iter.append(self.gbest.cost)
            print("Iteration: ", i, " Cost: ", self.gbest.cost)
            print(self.optimum_cost_tracking_eval.__len__())


if __name__ == '__main__':
    ABC(objective_function=rosenbrock, n_iter=5000).optimize()
