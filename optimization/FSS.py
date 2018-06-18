import numpy as np


class Fish:
    def __init__(self, dim):
        nan = float('nan')
        self.pos = [nan for _ in range(dim)]
        self.cost = np.inf
        self.weight = 0
        self.delta_cost = 0
        self.delta_pos = [nan for _ in range(dim)]

    def __str__(self):
        return '{} - {} - {}'.format(self.pos, self.cost, self.weight)


class FSS(object):
    def __init__(self, objective_function, n_iter=10000, n_eval=None, school_size=30, dim=30, weight_min=1,
                 step_i_init=0.1, step_i_end=0.001, step_v_init=0.01, step_v_end=0.001):

        self.objective_function = objective_function

        self.dim = dim
        self.minf = objective_function.minf
        self.maxf = objective_function.maxf
        self.n_iter = n_iter
        self.n_eval = n_eval
        self.school_size = school_size

        self.weight_min = weight_min
        self.weight_scale = self.n_iter

        self.prev_weight_school = 0.0
        self.curr_weight_school = 0.0

        self.step_i_init = step_i_init
        self.step_i_end = step_i_end

        self.step_v_init = step_v_init
        self.step_v_end = step_v_end

        self.step_i = self.step_i_init
        self.step_v = self.step_v_init

        self.gbest = None
        self.optimum_cost_tracking_eval = []
        self.optimum_cost_tracking_iter = []

        self.school_fish = []
        self.test = 0

    def __str__(self):
        return 'FSS'

    def calculate_barycenter(self):
        barycenter = np.zeros(self.dim)
        density = 0.0

        for f in self.school_fish:
            density += f.weight
            barycenter += f.pos * f.weight

        barycenter = barycenter / density

        return barycenter

    def _init_fish_school(self):

        self.school_fish = []
        self.gbest = Fish(self.dim)
        self.curr_weight_school = 0.0
        self.prev_weight_school = 0.0

        for f in range(self.school_size):
            f = Fish(self.dim)

            f.pos = np.random.uniform(self.minf, self.maxf, self.dim)
            f.weight = self.weight_scale / 2.0

            f.cost = self.objective_function.function(f.pos)
            self.optimum_cost_tracking_eval.append(self.gbest.cost)

            self.curr_weight_school += f.weight

            if f.cost < self.gbest.cost:
                self.gbest.cost = f.cost
                self.gbest.pos = f.pos

            self.school_fish.append(f)

        self.prev_weight_school = self.curr_weight_school

    def _mov_ind(self):
        max_delta_cost = 0
        for f in self.school_fish:
            rand = np.random.uniform(-1, 1, self.dim)

            n_pos = f.pos + (self.step_i * rand)
            if (n_pos < self.minf).any():
                n_pos[n_pos < self.minf] = self.minf
            elif (n_pos > self.maxf).any():
                n_pos[n_pos > self.maxf] = self.maxf

            n_cost = self.objective_function.function(n_pos)
            self.optimum_cost_tracking_eval.append(self.gbest.cost)

            if n_cost < f.cost:
                f.delta_cost = abs(n_cost - f.cost)
                f.cost = n_cost
                f.delta_pos = n_pos - f.pos
                f.pos = n_pos
            else:
                f.delta_pos = np.zeros(self.dim)
                f.delta_cost = 0

            if max_delta_cost < f.delta_cost:
                max_delta_cost = f.delta_cost

            if f.cost < self.gbest.cost:
                self.gbest.cost = f.cost
                self.gbest.pos = f.pos

        self.prev_weight_school = self.curr_weight_school
        self.curr_weight_school = 0.0

        for f in self.school_fish:  # feeding
            if max_delta_cost:
                f.weight = f.weight + (f.delta_cost / max_delta_cost)
                if f.weight < self.weight_min:
                    f.weight = self.weight_min

            self.curr_weight_school += f.weight  # update school weight

    def _mov_col_ins(self):
        cost_eval_enhanced = np.zeros(self.dim)
        density = 0.0

        for f in self.school_fish:
            density += f.delta_cost
            cost_eval_enhanced += f.delta_pos * f.delta_cost

        if density != 0:
            cost_eval_enhanced = cost_eval_enhanced / density

        for f in self.school_fish:
            new_pos = f.pos + cost_eval_enhanced

            if (new_pos < self.minf).any():
                new_pos[new_pos < self.minf] = self.minf
            elif (new_pos > self.maxf).any():
                new_pos[new_pos > self.maxf] = self.maxf

            f.pos = new_pos

    def _mov_col_vol(self):
        barycenter = self.calculate_barycenter()

        if self.curr_weight_school < self.prev_weight_school:
            self.test += 1

        for f in self.school_fish:
            rand = np.random.uniform(0, 1, self.dim)
            if self.curr_weight_school > self.prev_weight_school:
                new_pos = f.pos - (self.step_v * rand * (f.pos - barycenter))
            else:
                new_pos = f.pos + (self.step_v * rand * (f.pos - barycenter))

            if (new_pos < self.minf).any():
                new_pos[new_pos < self.minf] = self.minf
            elif (new_pos > self.maxf).any():
                new_pos[new_pos > self.maxf] = self.maxf

            f.cost = self.objective_function.function(new_pos)
            self.optimum_cost_tracking_eval.append(self.gbest.cost)
            f.pos = new_pos

            if f.cost < self.gbest.cost:
                self.gbest.cost = f.cost
                self.gbest.pos = f.pos

    def _update_step(self, curr_i, total_i):
        self.step_i = self.step_i_init - curr_i * (self.step_i_init - self.step_i_end) / total_i
        self.step_v = self.step_v_init - curr_i * (self.step_v_init - self.step_v_end) / total_i

    def optimize(self):
        self.optimum_cost_tracking_eval = []
        self.optimum_cost_tracking_iter = []

        self._init_fish_school()

        range_sim = self.n_iter
        tracking = self.optimum_cost_tracking_iter

        if self.n_eval is not None:
            range_sim = self.n_eval
            tracking = self.optimum_cost_tracking_eval

        while tracking.__len__() < range_sim:
            self._mov_ind()
            self._mov_col_ins()
            self._mov_col_vol()
            self._update_step(tracking.__len__(), range_sim)

            self.optimum_cost_tracking_iter.append(self.gbest.cost)
            print('{} - {} - {}'.format(self.optimum_cost_tracking_iter.__len__(),
                                            self.optimum_cost_tracking_eval.__len__(),
                                            self.gbest.cost))


from optimization.objective_functions import Sphere, Rastrigin, Rosenbrock

if __name__ == '__main__':
    FSS(objective_function=Sphere(), n_eval=500000).optimize()