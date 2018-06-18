import numpy as np


class Particle:
    def __init__(self, dim):
        nan = float('nan')  # not a number Nan
        self.pos = [nan for _ in range(dim)]
        self.speed = [nan for _ in range(dim)]
        self.cost = np.inf
        self.pbest_pos = self.pos
        self.pbest_cost = self.cost


class PSO:
    def __init__(self, objective_function, dim=30, swarm_size=30, n_iter=10000, n_eval=None, lo_w=0.4, up_w=0.9, c1=2.05,
                 c2=2.05,
                 v_max=10):
        self.dim = dim
        self.min_p_range = objective_function.minf
        self.max_p_range = objective_function.maxf

        self.swarm_size = swarm_size
        self.n_iter = n_iter
        self.n_eval = n_eval

        self.objective_function = objective_function.function
        self.optimum_cost_tracking_iter = []
        self.optimum_cost_tracking_eval = []

        self.swarm = []

        self.gbest = Particle(self.dim)
        self.gbest.cost = np.inf

        self.w = up_w
        self.up_w = up_w
        self.lo_w = lo_w

        self.c1 = c1
        self.c2 = c2
        self.v_max = min(v_max, 100000)

    def __str__(self):
        return 'PSO'

    def __init_swarm(self):

        self.gbest = Particle(self.dim)
        self.gbest.cost = np.inf

        for i in range(self.swarm_size):
            p = Particle(self.dim)
            p.pos = np.random.uniform(self.min_p_range, self.max_p_range, self.dim)
            p.speed = np.zeros(self.dim)
            p.cost = self.objective_function(p.pos)

            p.pbest_pos = p.pos
            p.pbest_cost = p.cost
            if p.pbest_cost < self.gbest.cost:
                self.gbest.pos = p.pbest_pos
                self.gbest.cost = p.pbest_cost

            self.optimum_cost_tracking_eval.append(self.gbest.cost)
            self.swarm.append(p)

        self.optimum_cost_tracking_iter.append(self.gbest.cost)

    # Restart the PSO
    def _init_pso(self):
        self.w = self.up_w
        self.swarm = []
        self.optimum_cost_tracking_iter = []
        self.optimum_cost_tracking_eval = []

    def optimize(self):
        self._init_pso()
        self.__init_swarm()

        range_sim = self.n_iter
        tracking = self.optimum_cost_tracking_iter

        if self.n_eval is not None:
            range_sim = self.n_eval
            tracking = self.optimum_cost_tracking_eval

        while tracking.__len__() < range_sim:

            for p in self.swarm:
                r1 = np.random.random(len(p.speed))
                r2 = np.random.random(len(p.speed))
                p.speed = self.w * np.array(p.speed) + self.c1 * r1 * (p.pbest_pos - p.pos) + self.c1 * r2 * (self.gbest.pos - p.pos)

                # Limit the velocity of the particle
                p.speed = np.sign(p.speed) * np.minimum(np.absolute(p.speed), np.ones(self.dim) * self.v_max)

                p.pos = p.pos + p.speed

                # Confinement of the particle in the search space
                if (p.pos < self.min_p_range).any() or (p.pos > self.max_p_range).any():
                    p.speed[p.pos < self.min_p_range] = -1 * p.speed[p.pos < self.min_p_range]
                    p.speed[p.pos > self.max_p_range] = -1 * p.speed[p.pos > self.max_p_range]
                    p.pos[p.pos > self.max_p_range] = self.max_p_range
                    p.pos[p.pos < self.min_p_range] = self.min_p_range

                p.cost = self.objective_function(p.pos)
                self.optimum_cost_tracking_eval.append(self.gbest.cost)

                if p.cost < p.pbest_cost:
                    p.pbest_pos = p.pos
                    p.pbest_cost = p.cost

                if p.pbest_cost < self.gbest.cost:
                    self.gbest.pos = p.pbest_pos
                    self.gbest.cost = p.pbest_cost

                self.w = self.up_w - (float(tracking.__len__()) / range_sim) * (self.up_w - self.lo_w)

            self.optimum_cost_tracking_iter.append(self.gbest.cost)
            print('{} - {} - {}'.format(self.optimum_cost_tracking_iter.__len__(),
                                            self.optimum_cost_tracking_eval.__len__(),
                                            self.gbest.cost))


from optimization.objective_functions import Sphere, Rastrigin, Rosenbrock

if __name__ == '__main__':
    # PSO(objective_function=Rosenbrock(), n_iter=10000).optimize()
    PSO(objective_function=Rastrigin(), n_iter=10000).optimize()

