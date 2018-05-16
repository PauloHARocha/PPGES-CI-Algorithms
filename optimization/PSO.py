import numpy as np


class Particle:
    def __init__(self, dim):
        nan = float('nan') #not a number Nan
        self.pos = [nan for _ in range(dim)]
        self.speed = [nan for _ in range(dim)]
        self.cost = nan
        self.pbest_pos = self.pos
        self.pbest_cost = self.cost


class PSO:
    def __init__(self, dim, objective_func, swarm_size=30, n_iter=10000, lo_w=0.4, up_w=0.9, c1=2.05, c2=2.05,
                 v_max=100, min_p_range=-100, max_p_range=100):
        self.dim = dim
        self.min_p_range = min_p_range
        self.max_p_range = max_p_range

        self.swarm_size = swarm_size
        self.n_iter = n_iter

        self.objective_func = objective_func
        self.optimum_cost_tracking_iter = []
        self.optimum_cost_tracking_eval = []

        self.swarm = []

        self.gbest_p = Particle(self.dim)
        self.gbest_p.cost = np.inf

        self.w = up_w
        self.up_w = up_w
        self.lo_w = lo_w

        self.c1 = c1
        self.c2 = c2
        self.v_max = min(v_max, 100000)

    def __init_swarm(self):

        self.gbest_p = Particle(self.dim)
        self.gbest_p.cost = np.inf

        for i in range(self.swarm_size):
            p = Particle(self.dim)
            p.pos = np.random.uniform(self.min_p_range, self.max_p_range, self.dim)
            p.speed = np.zeros(self.dim)
            p.cost = self.objective_func(p.pos)

            p.pbest_pos = p.pos
            p.pbest_cost = p.cost
            if p.pbest_cost < self.gbest_p.cost:
                self.gbest_p.pos = p.pbest_pos
                self.gbest_p.cost = p.pbest_cost

            self.optimum_cost_tracking_eval.append(self.gbest_p.cost)
            self.swarm.append(p)

        self.optimum_cost_tracking_iter.append(self.gbest_p.cost)


    # Restart the PSO
    def _init_pso(self):
        self.w = self.up_w
        self.optimum_cost_tracking_iter = []
        self.optimum_cost_tracking_eval = []

    def optmize(self):
        self._init_pso()
        self.__init_swarm()

        for i in range(self.n_iter):
            # swarm_pos = []
            for p in self.swarm:
                r1 = np.random.random(len(p.speed))
                r2 = np.random.random(len(p.speed))
                p.speed = self.w * p.speed + self.c1 * r1 * (p.pbest_pos - p.pos) \
                          + self.c1 * r2 * (self.gbest_p.pos - p.pos)

                # Limit the velocity of the particle
                p.speed = np.sign(p.speed) * np.minimum(np.absolute(p.speed), np.ones(self.dim) * self.v_max)

                p.pos = p.pos + p.speed

                # Confinement of the particle in the search space
                if (p.pos < self.min_p_range).any() or (p.pos > self.max_p_range).any():
                    p.speed[p.pos < self.min_p_range] = -1 * p.speed[p.pos < self.min_p_range]
                    p.speed[p.pos > self.max_p_range] = -1 * p.speed[p.pos > self.max_p_range]
                    p.pos[p.pos > self.max_p_range] = self.max_p_range
                    p.pos[p.pos < self.min_p_range] = self.min_p_range

                p.cost = self.objective_func(p.pos)
                self.optimum_cost_tracking_eval.append(self.gbest_p.cost)

                if p.cost <= p.pbest_cost:
                    p.pbest_pos = p.pos
                    p.pbest_cost = p.cost

                if p.pbest_cost <= self.gbest_p.cost:
                    self.gbest_p.pos = p.pbest_pos
                    self.gbest_p.cost = p.pbest_cost

                self.w = self.up_w - (float(i) / self.n_iter) * (self.up_w - self.lo_w)

                self.optimum_cost_tracking_iter.append(self.gbest_p.cost)


