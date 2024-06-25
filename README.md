### About this repo
This simulation implements Operations Research solutions for optimizing item distribution across multiple warehouses, considering budget constraints. It uses both CPLEX and Particle Swarm Optimization (PSO) algorithms to solve this distribution problem.

### Description
This project aims to optimize the distribution of items across multiple warehouses while maximizing profit and respecting budget constraints. It models the problem as a variant of the Multiple Knapsack Problem (MKP), where items represent products, knapsacks represent warehouses, and the constraints are based on retail shops handled by each warehouses.

### Features

- CPLEX solver implementation for exact solutions
- Particle Swarm Optimization (PSO) implementation for approximate solutions
- Random data sampling from kaggle datasets
- Comparison between CPLEX and PSO solutions
- Consideration of item sales, profit, and warehouse capacities

### Requirements
- Python 3.8 or 3.7
- cplex
- docplex
- numpy
- pandas

### Usage
- Ensure you have the required datasets in your project directory:
  - item dataset.csv: Contains information about items (sales, profit)
  - FMCG_data.csv: Contains information about warehouses (capacity)

- Install required package
  - ```pip install cplex docplex numpy pandas```
 
- Run python script

Data Source :
- [item dataset.csv](https://www.kaggle.com/datasets/abiodunonadeji/united-state-superstore-sales), Taken from Kaggle
- [FMCG_data](https://www.kaggle.com/datasets/suraj9727/supply-chain-optimization-for-a-fmcg-company), Taken from Kaggle
