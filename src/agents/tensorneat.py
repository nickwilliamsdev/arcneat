import torch
import torch.nn as nn

class TensorNeatRunner:
    """
    Runner for PyTorch-native TensorNEAT integration.
    Allows evolutionary optimization of the MuZero/DeltaNet weights
    across arckit task iterations.
    
    Designed to be PyTorch compilation (torch.compile) friendly.
    """
    def __init__(self, model_class, model_kwargs, population_size=10):
        self.population_size = population_size
        
        # Instantiate a population of models
        # In a true TensorNEAT, nodes and edges are dynamically structured,
        # but for DeltaNets we can optimize their weights or connectivity using NEAT.
        self.population = [model_class(**model_kwargs) for _ in range(population_size)]
        
        # To be friendly to torch.compile, we can stack parameters
        # or use vmap/functorch, but for simplicity we keep them as module lists for now.

    # @torch.compile (can be enabled over time once compilation bottlenecks are resolved locally)
    def evaluate_individual(self, individual_idx, grid_inputs):
        """
        Evaluate a single individual on a batch of ARC grids.
        """
        model = self.population[individual_idx]
        policy_logits, value, _ = model(grid_inputs)
        return policy_logits, value
        
    def mutate(self):
        """
        PyTorch-native mutation logic. Apply mutations to the whole population.
        """
        for model in self.population:
            self._mutate_single(model)

    def _mutate_single(self, model):
        with torch.no_grad():
            for param in model.parameters():
                # Simple Gaussian mutation for demonstration
                mutation_mask = torch.rand_like(param) < 0.1
                noise = torch.randn_like(param) * 0.05
                param.add_(mutation_mask * noise)

    def select_and_reproduce(self, fitness_scores):
        """
        Evolutionary selection step to preserve top-performing networks doing ARC reasoning.
        """
        import copy
        
        # Sort individuals by fitness (descending)
        sorted_indices = torch.argsort(torch.tensor(fitness_scores), descending=True).tolist()
        
        # Keep top 20% (elitism)
        num_elites = max(1, self.population_size // 5)
        elites = [self.population[i] for i in sorted_indices[:num_elites]]

        new_population = []
        
        # Transfer elites unmodified into the next generation
        for e in elites:
            new_population.append(copy.deepcopy(e))

        # Fill the rest with randomly mutated elites
        while len(new_population) < self.population_size:
            parent_idx = torch.randint(0, num_elites, (1,)).item()
            parent = elites[parent_idx]
            child = copy.deepcopy(parent)
            self._mutate_single(child)
            new_population.append(child)

        self.population = new_population
