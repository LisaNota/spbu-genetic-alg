# Genetic Algorithm Implementation

## Project Objective:
The objective of this project is to explore two main ways of encoding chromosomes' genotype in a genetic algorithm and to evaluate their effectiveness.

## Project Components:
The project consists of the following components:

### `Genetic` Class:
- **Attributes**:
  - `num_generations`: Number of generations for the genetic algorithm.
  - `population_size`: Size of the population.
  - `mutation_rate`: Probability of mutation.
  - `low`: Lower bound for gene values.
  - `high`: Upper bound for gene values.
- **Methods**:
  - `fitness_function(x, y)`: Defines the fitness function.
  - `initialize_population()`: Initializes the population with random individuals.
  - `enforce_bounds(individual)`: Enforces the bounds for gene values.
  - `crossover(parent1, parent2)`: Performs crossover between two parents.
  - `mutate(individual)`: Mutates an individual.
  - `select_parents(population, fitness_values)`: Selects parents using tournament selection.
  - `run_genetic_algorithm()`: Runs the genetic algorithm and returns the best individual and its fitness.

### User Interface:
The user interface is implemented using Tkinter, providing options to set parameters for the genetic algorithm and visualize the results.

## Usage:
1. Set the parameters in the GUI, including mutation rate, population size, number of generations, and gene value bounds.
2. Click on the "Calculate" button to run the genetic algorithm.
3. The results, including the best solution and its coordinates, will be displayed in the interface.

## Dependencies:
- Python 3.x
- NumPy
- Tkinter (standard Python library)

## How to Run:
- Ensure Python 3.x and the required dependencies are installed.
- Run the Python script `genetic_algorithm.py`.
- Adjust the parameters as needed in the GUI and click on the "Calculate" button to run the algorithm.
