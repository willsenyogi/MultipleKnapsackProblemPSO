import cplex
from cplex.exceptions import CplexSolverError
from docplex.mp.model import Model
import numpy as np
import pandas as pd
import random
from typing import List

# DATASET IMPORT
item_data = pd.read_csv('item dataset.csv')
warehouse_data = pd.read_csv('FMCG_data.csv')

# RANDOM DATA
filtered_item_data = item_data[item_data['Profit'] > 0]
random_item_data = filtered_item_data.sample(n=100)
random_warehouse_data = warehouse_data.sample(n=50)

# CPLEX PARAMETERS
n = 60 #banyak barang
m = 10 #banyak knapsack
w = random_item_data['Sales'].round().tolist() #bobot barang
p = random_item_data['Profit'].round().tolist() #profit barang
c = (random_warehouse_data['retail_shop_num'] * 100).tolist() #kapasitas knapsack

# CPLEX MODEL
def cplexMkp():
    try:
        mdl = Model(name='Warehouse Distribution MKP')

        # Decision variable
        x = np.empty((n, m), dtype=object)
        for i in range(n):
            for j in range(m):
                x[i][j] = mdl.binary_var(name='x_{}_{}'.format(i, j))

        # Model objective
        mdl.maximize(sum(sum(p[i] * x[i][j] for i in range(n)) for j in range(m))
    )

    # Constraints
        # Constraint untuk Kapasitas Knapsack 
        for j in range(m):
            mdl.add_constraint(sum(w[i] * x[i][j] for i in range(n)) <= c[j])

        # Constraint untuk setiap barang i diletakkan di setiap knapsack j 
        for i in range(n):
            mdl.add_constraint(sum(x[i][j] for j in range(m)) <= 1)

        s = mdl.solve()
        mdl.print_solution() #Print solution

    except CplexSolverError as exc:
        if exc.args[2] == cplex.exceptions.error_codes.CPXERR_NO_MEMORY:
            print("insufficient memory.")
        else:
            raise   

# PSO FUNCTIONS
class Item:
    def __init__(self, name, weight, value):
        self.name = name
        self.weight = weight
        self.value = value


class Individual:
    lbest = 0
    velocity = 0

    def __init__(self, bits: List[int]):
        self.bits = bits

    def __str__(self):
        return repr(self.bits)

    def __hash__(self):
        return hash(str(self.bits))

    def fitness(self) -> int:
        total_value = []
        total_weight = []
        for i in range(M):
            total_value.append(sum(item.value for item, bit in zip(items, self.bits) if bit == (i+1)))
            total_weight.append(sum(item.weight for item, bit in zip(items, self.bits) if bit == (i+1)))
            if total_weight[i] > MAX_KNAPSACK_WEIGHT[i]:
                return 0


        profit = sum(total_value)
        if self.lbest < profit:
            self.lbest = profit

        return profit

# PSO PARAMETERS
MAX_KNAPSACK_WEIGHT = c
c1 = c2 = 2.0
w_inertia = 0.8
M = m
items = [Item(str(i + 1), w[i], p[i]) for i in range(n)]

gbest = 0
gbest_obj = 0

def generate_initial_population(count=150) -> List[Individual]:
    global gbest

    population = set()

    # generate initial population having `count` individuals
    while len(population) != count:

        bits = [
            #random.choice([0, 1])
            random.choice(range(M+1))
            for _ in items
        ]
        population.add(Individual(bits))
    
    #set gbest particle and its fitness score
    gbest = max(population, key = lambda x: x.fitness())
    gbest_obj = gbest.fitness()

    return list(population)

def update(population: List[Individual]) -> List[Individual]:
    global gbest, gbest_obj

    # update velocity and position (allocation of knapsack for each item)
    for individual in population:
        r1 = random.SystemRandom().uniform(0, 2)
        r2 = random.SystemRandom().uniform(0, 2)
        individual.velocity = w_inertia * individual.velocity + c1*r1*(individual.lbest - individual.fitness()) + c2*r2*(gbest_obj - individual.fitness())
        for i in range(len(individual.bits)):
            if random.SystemRandom().uniform(-3, 3) <= individual.velocity:
                # Flip the bit
                while (True):
                    newbit = random.choice(range(M+1))
                    if(newbit != individual.bits[i]): break
                individual.bits[i] = newbit
        
    #set gbest particle and its fitness score
    gbest = max(population, key = lambda x: x.fitness())
    gbest_obj = gbest.fitness()
    return population

def print_generation(population: List[Individual]):
    for individual in population:
        print(individual.bits, " p:", individual.fitness(), " v:", individual.velocity, " best:", individual.lbest)
    print("Global Best Particle = ", gbest, " : ", gbest.fitness())

def pso():
    population = generate_initial_population()
    print("Initial Population:")
    for i in range(6):
        print(population[i], " : ", population[i].fitness(), " best = ", population[i].lbest)
    
    prevGBest_obj = gbest.fitness()
    print("Global Best Particle = ", gbest, " : ", prevGBest_obj)

    sameGBestObj = 0
    for i in range(100):
        population = update(population)
        if(gbest_obj == prevGBest_obj): sameGBestObj += 1
        else: prevGBest_obj = gbest_obj
        if(sameGBestObj == 5): break
    print("Done in ", i+1, " iterations.")

if __name__ == '__main__':
    print("CPLEX SOLUTION :")
    cplexMkp()
    print("=================================================================================== \nPSO SOLUTION :")
    for i in range(10):
        pso()
        print("Solution iteration-", i+1, ": ", gbest, gbest.fitness(), "\n")