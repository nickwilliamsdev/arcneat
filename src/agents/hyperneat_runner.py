import torch
import torch.nn as nn
import neat
from src.pytorch_neat.cppn import create_cppn
from src.models.deltanet import DeltaNetARCModel
from src.env.arc_env import ARCEnv

class HyperNEATDeltaNetBuilder:
    def __init__(self, config, vocab_size=10, d_model=64, dim_head=64, action_space=9000, device="cpu"):
        self.config = config
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.dim_head = dim_head
        self.action_space = action_space
        self.device = device
        
        # Precompute coordinate tensors to query the CPPN for the DeltaNet cell structures
        # We need weight matrices:
        # to_q, to_k, to_v : (dim_head, d_model)
        # to_beta: (1, d_model)
        # out_proj: (d_model, dim_head)
        
        # We'll just define basic grid coordinates for each target
        self._qkv_coords = self._make_coords(self.dim_head, self.d_model)
        self._beta_coords = self._make_coords(1, self.d_model)
        self._out_proj_coords = self._make_coords(self.d_model, self.dim_head)
        
    def _make_coords(self, out_dim, in_dim):
        """ Creates an (out_dim * in_dim, 4) tensor of (x_in, y_in, x_out, y_out) scaled [-1, 1] """
        y_in = torch.linspace(-1, 1, in_dim)
        x_in = torch.zeros(in_dim)  # simplify: 1D input essentially
        
        y_out = torch.linspace(-1, 1, out_dim)
        x_out = torch.zeros(out_dim) # simplify: 1D output essentially
        
        # Meshgrid
        Y_out, Y_in = torch.meshgrid(y_out, y_in, indexing='ij')
        X_out, X_in = torch.meshgrid(x_out, x_in, indexing='ij')
        
        # Stack into (N, 4)
        c = torch.stack([X_in.flatten(), Y_in.flatten(), X_out.flatten(), Y_out.flatten()], dim=1)
        return c.to(self.device)

    def create_model(self, genome, config):
        """
        Takes a NEAT genome, builds a CPPN, and queries it to generate weights for a new DeltaNetARCModel.
        """
        # Leaf names are our inputs to CPPN, node names are outputs
        leaf_names = ["x_in", "y_in", "x_out", "y_out"]
        node_names = ["w_q", "w_k", "w_v", "w_beta", "w_out"]
        
        cppn_nodes = create_cppn(genome, config, leaf_names, node_names)
        
        # Initialize a blank model
        model = DeltaNetARCModel(self.vocab_size, self.d_model, self.action_space).to(self.device)
        
        # cppn_nodes returns a list: [w_q, w_k, w_v, w_beta, w_out]
        node_q, node_k, node_v, node_beta, node_out = cppn_nodes
        
        # Generate weights
        with torch.no_grad():
            w_q = node_q(x_in=self._qkv_coords[:,0], y_in=self._qkv_coords[:,1], x_out=self._qkv_coords[:,2], y_out=self._qkv_coords[:,3])
            w_k = node_k(x_in=self._qkv_coords[:,0], y_in=self._qkv_coords[:,1], x_out=self._qkv_coords[:,2], y_out=self._qkv_coords[:,3])
            w_v = node_v(x_in=self._qkv_coords[:,0], y_in=self._qkv_coords[:,1], x_out=self._qkv_coords[:,2], y_out=self._qkv_coords[:,3])
            
            # Update to_q, to_k, to_v
            model.deltanet.to_q.weight.copy_(w_q.view(self.dim_head, self.d_model))
            model.deltanet.to_k.weight.copy_(w_k.view(self.dim_head, self.d_model))
            model.deltanet.to_v.weight.copy_(w_v.view(self.dim_head, self.d_model))
            
            # Update beta
            w_beta = node_beta(x_in=self._beta_coords[:,0], y_in=self._beta_coords[:,1], x_out=self._beta_coords[:,2], y_out=self._beta_coords[:,3])
            model.deltanet.to_beta.weight.copy_(w_beta.view(1, self.d_model))
            
            # Update out_proj
            w_out = node_out(x_in=self._out_proj_coords[:,0], y_in=self._out_proj_coords[:,1], x_out=self._out_proj_coords[:,2], y_out=self._out_proj_coords[:,3])
            model.deltanet.out_proj.weight.copy_(w_out.view(self.d_model, self.dim_head))
            
        return model
