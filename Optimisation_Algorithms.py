#Contributors: Michael Starr ( 20236596), Rishi Gupta ( 20231111 )


import numpy as np
import copy
import sys
import random
import itertools
import cma
import math
from scipy.stats import multivariate_normal as MVN
import pandas as pd
import matplotlib.pyplot as plt
import itertools as it


# # Objective Function

def rastrigin(x):
    return np.sum(x * x - 10 * np.cos(2 * np.pi * x)) + 10 * np.size(x)


# Random Search
# Citation : https://github.com/angelgaspar/randomsearch/blob/master/randomsearch/randomsearch.py
def Random_Search(function, dimensions, lower_boundary, upper_boundary, max_iter, maximize=False):
    best_solution = np.array([float()] * dimensions)
    history = []
    print(max_iter)
    for i in range(dimensions):
        best_solution[i] = random.uniform(lower_boundary[i], upper_boundary[i])
    for _ in range(max_iter):

        solution1 = function(best_solution)

        new_solution = [lower_boundary[d] + random.random() * (upper_boundary[d] - lower_boundary[d]) for d in
                        range(dimensions)]
        new_solution = np.array(new_solution)
        if np.greater_equal(new_solution, lower_boundary).all() and np.less_equal(new_solution, upper_boundary).all():
            solution2 = function(new_solution)
        elif maximize:
            solution2 = -100000.0
        else:
            solution2 = 100000.0

        if solution2 > solution1 and maximize:
            best_solution = new_solution
        elif solution2 < solution1 and not maximize:
            best_solution = new_solution
        tp = (function(best_solution), best_solution)
        history.append(tp)
    best_fitness = function(best_solution)

    return best_fitness, best_solution, history


# # GA Algorithm

def GA(f, init, nbr, crossover, select, popsize, ngens, pmut):
    # make initial population, evaluate fitness, print stats

    history = []
    pop = [init() for _ in range(popsize)]
    popfit = [f(x) for x in pop]
    history.append(stats(0, popfit))
    gen_best = []
    tp = ()
    for gen in range(1, ngens):
        # make an empty new population
        newpop = []
        # elitism
        bestidx = min(range(popsize), key=lambda i: popfit[i])
        best = pop[bestidx]
        newpop.append(best)
        while len(newpop) < popsize:
            # select and crossover
            p1 = select(pop, popfit)
            p2 = select(pop, popfit)
            c1, c2 = crossover(p1, p2)
            # apply mutation to only a fraction of individuals
            if np.random.random() < pmut:
                c1 = nbr(c1)
            if np.random.random() < pmut:
                c2 = nbr(c2)
            # add the new individuals to the population
            newpop.append(c1)
            newpop.append(c2)
        # overwrite old population with new, evaluate, do stats
        pop = newpop
        popfit = [f(x) for x in pop]
        history.append(stats(gen, popfit))
        best_indx = popfit.index(min(popfit))
        tp = (popfit[best_indx], pop[best_indx])
        gen_best.append(tp)
    bestidx = popfit.index(min(popfit))

    return pop[bestidx], popfit[bestidx], gen_best


def stats(gen, popfit):
    return gen, (gen + 1) * len(popfit), np.min(popfit), np.mean(popfit), np.median(popfit), np.max(popfit), np.std(
        popfit)


def tournament_select(pop, popfit, size):
    # To avoid re-calculating f for same individual multiple times, we
    # put fitness evaluation in the main loop and store the result in
    # popfit. We pass that in here.  Now the candidates are just
    # indices, representing individuals in the population.
    candidates = random.sample(list(range(len(pop))), size)

    # The winner is the index of the individual with min fitness.
    winner = min(candidates, key=lambda c: popfit[c])
    return pop[winner]


def real_init(n):
    # Initialise the individual somewhere within bounds.
    return np.random.uniform(low=-5.12, high=5.12, size=(n,))
    # return np.random.random(n)


def real_nbr(x):
    ub = 5.12
    lb = -5.12
    delta = 0.1
    x = x.copy()
    # draw from a Gaussian
    # x = x + delta * np.random.randn(len(x))
    x = x + delta * np.random.randn(len(x))

    # if outside of bounds then mirror the individual back into bounds by the amount they were going outside by.
    if ((x > ub).any() | (x < lb).any()):
        for idx, elem in enumerate(x):
            if (elem < lb):
                # Need to mirror only the part of the step that is out of bounds
                elem += 2 * np.abs(np.abs(elem) - np.abs(lb))
            elif (elem > ub):
                elem -= 2 * np.abs(np.abs(elem) - np.abs(ub))
        x[idx] = elem
    return x


def uniform_crossover(p1, p2):
    c1, c2 = [], []
    for i in range(len(p1)):
        if np.random.random() < 0.5:
            c1.append(p1[i]);
            c2.append(p2[i])
        else:
            c1.append(p2[i]);
            c2.append(p1[i])
    return np.array(c1), np.array(c2)


def order1_crossover(p1, p2):
    k = np.random.randint(0, len(p1) - 1)

    c1 = p1
    c2 = p2
    for i in range(k, len(p1)):
        c1[i], c2[i] = c2[i], c1[i]
    return np.array(c1), np.array(c2)


# # PSO Algorithm

# Cite - https://gist.github.com/VinACE/37cbd8e9392c2391f19a8b2d01979602

class Particle:
    def __init__(self, f, dim, minx, maxx):
        self.position = np.array([0.0 for i in range(dim)])
        self.velocity = np.array([0.0 for i in range(dim)])
        self.best_part_pos = np.array([0.0 for i in range(dim)])

        for i in range(dim):
            self.position[i] = ((maxx - minx) * np.random.random() + minx)
            self.velocity[i] = ((maxx - minx) * np.random.random() + minx)

        self.f = f(self.position)  # curr f
        self.best_part_pos = copy.copy(self.position)
        self.best_part_err = self.f  # best f


def PSO(f, max_epochs, n, dim, minx, maxx, c1, c2, w):
    # create n random particles
    #     options = {'c1':c1,'c2':c2,'w':w}
    swarm = [Particle(f, dim, minx, maxx) for i in range(n)]
    best_swarm_pos = [0.0 for i in range(dim)]  # not necess.
    best_swarm_err = sys.float_info.max  # swarm best
    for i in range(n):  # check each particle
        if swarm[i].f < best_swarm_err:
            best_swarm_err = swarm[i].f
            best_swarm_pos = copy.copy(swarm[i].position)

    epoch = 0

    #     w = options['w']  # inertia
    #     c1 = options['c1']  # cognitive (particle)
    #     c2 = options['c2']  # social (swarm)
    best_swarm = []
    while epoch < max_epochs:

        for i in range(n):  # process each particle

            # compute new velocity of curr particle
            for k in range(dim):
                r1 = np.random.random()  # randomizations
                r2 = np.random.random()

                swarm[i].velocity[k] = ((w * swarm[i].velocity[k]) +
                                        (c1 * r1 * (swarm[i].best_part_pos[k] -
                                                    swarm[i].position[k])) +
                                        (c2 * r2 * (best_swarm_pos[k] -
                                                    swarm[i].position[k])))

                if swarm[i].velocity[k] < minx:
                    swarm[i].velocity[k] = minx
                elif swarm[i].velocity[k] > maxx:
                    swarm[i].velocity[k] = maxx

            # compute new position using new velocity
            for k in range(dim):
                swarm[i].position[k] += swarm[i].velocity[k]

            # compute f of new position
            swarm[i].f = f(swarm[i].position)

            # is new position a new best for the particle?
            if swarm[i].f < swarm[i].best_part_err:
                swarm[i].best_part_err = swarm[i].f
                swarm[i].best_part_pos = copy.copy(swarm[i].position)

            # is new position a new best overall?
            if swarm[i].f < best_swarm_err:
                best_swarm_err = swarm[i].f
                best_swarm_pos = copy.copy(swarm[i].position)
        best_tup = (best_swarm_err, best_swarm_pos)
        best_swarm.append(best_tup)
        # for-each particle
        epoch += 1
    # while
    return best_swarm_err, best_swarm_pos, best_swarm


# # SA

def SA(f, init, nbr, T, alpha, maxits):
    """Simulated annealing. Assume we are minimising.
    Return the best ever x and its f-value.

    Pass in initial temperature T and decay factor alpha.

    T decays by T *= alpha at each step.
    """
    x = init()  # generate an initial random solution
    fx = f(x)
    bestx = x
    bestfx = fx
    history = []
    for i in range(1, maxits):
        xnew = nbr(x)  # generate a neighbour of x
        fxnew = f(xnew)

        # "accept" xnew if it is better, OR even if worse, with a
        # small probability related to *how much worse* it is. assume
        # we are minimising, not maximising.
        if fxnew < fx or random.random() < math.exp((fx - fxnew) / T):
            x = xnew
            fx = fxnew

            # make sure we keep the best ever x also
            if fxnew < bestfx:
                bestx = x
                bestfx = fx
        tp = (bestfx, bestx)
        history.append(tp)
        T *= alpha  # temperature decreases
    return bestfx, bestx, history


# # Late Acceptance Hill-Climbing

# L is history length
# n is the budget of evaluations
# C is the cost function
# init is a function that creates an initial individual
# nbr is a neighbourhood function

def LAHC(L, n, C, init, nbr):
    list_of_best = []
    s = init()  # initial solution
    Cs = C(s)  # cost of current solution
    best = s  # best-ever solution
    Cbest = Cs  # cost of best-ever
    f = [Cs] * L  # initial history
    # print(0, Cbest, best)
    for I in range(1, n):  # number of iterations
        s_ = nbr(s)  # candidate solution
        Cs_ = C(s_)  # cost of candidate
        if Cs_ < Cbest:  # minimising
            best = s_  # update best-ever
            Cbest = Cs_
            # print(I, Cbest, best)
        v = I % L  # v indexes I circularly
        if Cs_ <= f[v] or Cs_ <= Cs:
            s = s_  # accept candidate
            Cs = Cs_  # (otherwise reject)
        f[v] = Cs  # update circular history
        list_of_best.append((Cbest, best))

    return best, Cbest, list_of_best


# our standard bitstring init. use lambda: init(n) to give a function
# that takes no parameters.
def init(n):
    return [np.random.randrange(2) for _ in range(n)]


######I think this needs to be changed to something more applicable to our problem.#####
# our usual bitstring nbr
def nbr(x):
    x = x.copy()
    i = np.random.randrange(len(x))
    x[i] = 1 - x[i]
    return x


# # CMA - Covariance Matrix Adaptation

def CMA(f, init, popsize, n, ngens, sigma, seed):
    initial = real_init(n)
    maxits = ngens * popsize
    es = cma.CMAEvolutionStrategy(initial, sigma, {'popsize': popsize, 'bounds': [-5.12, 5.12], 'seed': seed})
    es.optimize(f, iterations=maxits / es.popsize)
    fbest = es.result.fbest
    xbest = es.result.xbest
    #     es.logger.plot()
    return fbest, xbest


# # Main function:


def main(**kwargs):
    # LAHC params:
    algo = kwargs.get('algo')
    ngens = int(kwargs.get('ngens'))
    n = int(kwargs.get('dimensions'))
    fitness_budget = 50000
    popsize = int(fitness_budget / ngens)
    # history length
    # CMA params:
    f = rastrigin

    max_bound = 5.12 * np.ones(n)
    min_bound = - max_bound
    # bounds = (min_bound, max_bound)
    plt.figure()
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost of function")
    final_costs = []
    for seed in range(1, 6):
        print("\nseed = ", seed)
        random.seed(seed)
        np.random.seed(seed)
        if algo == 'GA':
            # GA
            winner, cost, h = GA(f,
                                 lambda: real_init(n),
                                 real_nbr,
                                 eval(kwargs.get('crossover_type')),
                                 lambda pop, popfit: tournament_select(pop, popfit, int(kwargs.get('tsize'))),
                                 popsize,
                                 ngens,
                                 float(kwargs.get('pmut'))
                                 )
            final_costs.append(cost)
            print()
            print("Best solution from GA for seed = {} :".format(str(seed)))
            print(cost, winner)

            # plt.figure()
            # plt.title("GA")
            # plt.plot([n[0] for n in h])
        elif algo == 'PSO':
            # PSO
            cost, winner, h = PSO(f, ngens, popsize, n, -5.12, 5.12, float(kwargs.get('c1')), float(kwargs.get('c2')),
                                  float(kwargs.get('w')))
            final_costs.append(cost)
            print("Best solution from PSO for seed = {} :".format(str(seed)))
            print(cost, winner)
            # plt.figure()
            # plt.title("PSO")
            # plt.plot([n[0] for n in h])
        elif algo == 'LAHC':
            # Late Acceptance Hill-Climbing
            # def LAHC(L, n, C, init, nbr)

            # L is history length
            # n is the budget of evaluations
            # C is the cost function
            # init is a function that creates an initial individual
            # nbr is a neighbourhood function

            # Replaced nbr with a more real-valued scale, not just binary bitstrings.

            winner, cost, list_of_best_costs = LAHC(int(kwargs.get('L')), popsize * ngens, f, lambda: real_init(n),
                                                    real_nbr)
            final_costs.append(cost)
            print("Best solution from LAHC [best_cost,best] for seed {} :".format(str(seed)))
            print(cost, winner)
            print()
            # plt.figure()
            # plt.title("LAHC")
            # plt.plot([element[0] for element in list_of_best_costs])
        elif algo == "CMA":
            cost, winner = CMA(f, lambda: real_init(n), popsize, n, ngens, float(kwargs.get('sigma')), seed)
            final_costs.append(cost)
            print()
            print("Best solution from CMA for seed = {} :".format(str(seed)))
            print(cost, winner)
            # plt.figure()
            # plt.title("CMA")
            # plt.plot(h)
            # plt.savefig("CMA_Cost_VS_Iterations.png")
        elif algo == "RS":
            cost, winner, h = Random_Search(f, n, min_bound, max_bound, ngens * popsize, maximize=False)
            final_costs.append(cost)
            print("Best solution from Random Search for seed = {} :".format(str(seed)))
            print(cost, winner)
            plt.title("RS")
            plt.plot([n[0] for n in h])
            plt.savefig("RS_Cost_VS_Iterations.png")

        else:
            print('Wrong Algorithm chosen...')
            return None, None

    # Write mean, std_dev etc into a dat file.
    mean = np.mean(final_costs)
    std = np.std(final_costs)
    # print("Mean and Standard Deviation for {} is Mean={} and STD={}".format(algo, str(mean), str(std)))
    return mean, std


if __name__ == "__main__":

    algo = sys.argv[1]
    kwargs = {}
    data = []
    # Params used by all algs:
    params = []
    # generations = np.linspace(50,200,4)
    generations = [10, 100, 1000, 5000]
    dimensions = [6]
    params.append(generations)
    params.append(dimensions)

    if algo == "GA": # Factorial Experiment for GA
        #### GA
        crossover_type = ["uniform_crossover", "order1_crossover"]
        tsize = [2, 4, 6, 8]
        pmut = np.linspace(0.1, 1, 4)
        params.append(crossover_type)
        params.append(tsize)
        params.append(pmut)

        for configs in it.product(*params):
            kwargs['algo'] = algo
            kwargs['ngens'] = configs[0]
            kwargs['dimensions'] = configs[1]
            kwargs['crossover_type'] = configs[2]
            kwargs['tsize'] = configs[3]
            kwargs['pmut'] = configs[4]
            print(kwargs)
            mean, std = main(**kwargs)
            data.append((configs[0], configs[1], configs[2], configs[3], configs[4], mean, std))
        cols = ['ngens', 'dimension', 'crossover_type', 'tsize', 'pmut', 'mean', 'std']
        result = pd.DataFrame(data, columns=cols)
        result.to_csv("GA_Factorical_Experiment.csv", index=False)

    elif algo == "PSO":# Factorial Experiment for PSO
        #### PSO
        # Get a reasonably even distribution across the param space.
        c1 = [0.125, 0.375, 0.625, 0.875]
        c2 = [0.125, 0.375, 0.625, 0.875]
        w = [0.125, 0.375, 0.625, 0.875]
        params.append(c1)
        params.append(c2)
        params.append(w)

        for configs in it.product(*params):
            kwargs['algo'] = algo
            kwargs['ngens'] = configs[0]
            kwargs['dimensions'] = configs[1]
            kwargs['c1'] = configs[2]
            kwargs['c2'] = configs[3]
            kwargs['w'] = configs[4]
            print(kwargs)
            mean, std = main(**kwargs)
            data.append((configs[0], configs[1], configs[2], configs[3], configs[4], mean, std))
        cols = ['ngens', 'dimension', 'c1', 'c2', 'w', 'mean', 'std']

        result = pd.DataFrame(data, columns=cols)
        result.to_csv("PSO_Factorical_Experiment.csv", index=False)

    elif algo == "LAHC":# Factorial Experiment for LAHC
        #### LAHC
        L = np.linspace(2, 10, 5)
        params.append(L)

        for configs in it.product(*params):
            kwargs['algo'] = algo
            kwargs['ngens'] = configs[0]
            kwargs['dimensions'] = configs[1]
            kwargs['L'] = configs[2]
            print(kwargs)
            mean, std = main(**kwargs)
            data.append((configs[0], configs[1], configs[2], mean, std))
        cols = ['ngens', 'dimension', 'L', 'mean', 'std']

        result = pd.DataFrame(data, columns=cols)
        result.to_csv("LAHC_Factorical_Experiment.csv", index=False)

    elif algo == "CMA":# Factorial Experiment for CMA
        #### CMA
        sigma = np.linspace(0.1,1,10)
        params.append(sigma)

        for configs in it.product(*params):
            kwargs['algo'] = algo
            kwargs['ngens'] = configs[0]
            kwargs['dimensions'] = configs[1]
            kwargs['sigma'] = configs[2]
            print(kwargs)
            mean, std = main(**kwargs)
            data.append((configs[0], configs[1], configs[2], mean, std))
        cols = ['ngens', 'dimension', 'sigma', 'mean', 'std']

        result = pd.DataFrame(data, columns=cols)
        result.to_csv("CMA1_Factorical_Experiment.csv", index=False)

    elif algo == "RS":# Factorial Experiment for RS
        #### SA

        for configs in it.product(*params):
            kwargs['algo'] = algo
            kwargs['ngens'] = configs[0]
            kwargs['dimensions'] = configs[1]
            print(kwargs)
            mean, std = main(**kwargs)
            data.append((configs[0], configs[1], mean, std))
        cols = ['ngens', 'dimension', 'mean', 'std']

        result = pd.DataFrame(data, columns=cols)
        result.to_csv("RS_Factorical_Experiment.csv", index=False)

    else:
        print("Algo not found")

#################################################################################
#################################################################################
#################################################################################
#################################################################################
