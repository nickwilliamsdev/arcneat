import torch
import torch.nn as nn

class NEATDeltaNetExecution(nn.Module):
    def __init__(self, num_nodes, plastic=False, beta=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.plastic = plastic
        self.beta = beta
        
        # Fast weight state matrix: Stores the topology/weights of the NEAT genome
        self.register_buffer("W_initial", torch.zeros(num_nodes, num_nodes))
        
    def load_neat_genome(self, genome, config):
        """
        Parses a pytorch-neat/neat-python genome into the base weight matrix.
        """
        self.W_initial.zero_()
        # Extract connection genes
        for cg in genome.connections.values():
            if cg.enabled:
                # In NEAT, connection goes from cg.key[0] -> cg.key[1]
                # In standard matrix math (y = Wx), row = output, col = input
                in_node, out_node = cg.key
                self.W_initial[out_node, in_node] = cg.weight

    def forward(self, inputs, steps=5):
        """
        Args:
            inputs: Tensor of shape (batch_size, num_nodes) containing initial activations
                    (inputs clamped to input node indices, others zero).
            steps: Number of recurrent time-steps to relax/propagate through the graph.
        """
        batch_size = inputs.size(0)
        
        # Initialize fast weight state for the batch
        W_t = self.W_initial.clone().unsqueeze(0).repeat(batch_size, 1, 1)
        x_t = inputs.clone()
        
        for _ in range(steps):
            # 1. Readout via current associative memory (Standard FWP matrix-vector multiplication)
            # x_next = W_t * x_t
            x_next = torch.bmm(W_t, x_t.unsqueeze(-1)).squeeze(-1)
            x_next = torch.tanh(x_next)  # Standard NEAT node activation
            
            # Re-clamp raw environmental inputs if evaluating in an open loop
            x_next[:, :inputs.size(1)] = inputs[:, :inputs.size(1)]
            
            # 2. Delta Rule Fast Weight Update (Plasticity / Lifetime Adaptation)
            if self.plastic:
                # k_t is the current state, v_t is the target state (the updated prediction)
                k_t = x_t.unsqueeze(-1)
                v_t = x_next.unsqueeze(-1)
                
                # Delta Rule error term: (v_t - W_t * k_t)
                prediction_error = v_t - torch.bmm(W_t, k_t)
                
                # Outer product update: W_t = W_t + beta * (error (times) k_t^T)
                delta_W = torch.bmm(prediction_error, k_t.transpose(1, 2))
                W_t = W_t + self.beta * delta_W
                
            x_t = x_next
            
        return x_t