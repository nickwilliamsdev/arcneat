import torch
import torch.nn as nn
import neat
from src.pytorch_neat.cppn import create_cppn
from src.models.deltanet import DeltaNetARCModel

# Semantic z-axis values for each DeltaNet weight matrix.
# These are fixed constants that give the CPPN a stable signal per matrix role.
_WEIGHT_ROLES = {
    "q":    -1.00,
    "k":    -0.50,
    "v":     0.00,
    "beta":  0.50,
    "out":   1.00,
}


class HyperNEATDeltaNetBuilder:
    def __init__(self, config, vocab_size=10, d_model=64, dim_head=64, action_space=9000, device="cpu"):
        self.config = config
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.dim_head = dim_head
        self.action_space = action_space
        self.device = device

        # Precompute 2D coordinate tensors (used by create_model)
        self._qkv_coords   = self._make_coords(self.dim_head, self.d_model)
        self._beta_coords  = self._make_coords(1,             self.d_model)
        self._out_proj_coords = self._make_coords(self.d_model, self.dim_head)

    # ------------------------------------------------------------------
    # 2D substrate helpers (original)
    # ------------------------------------------------------------------

    def _make_coords(self, out_dim, in_dim):
        """(out_dim * in_dim, 4) tensor: (x_in, y_in, x_out, y_out) in [-1, 1]"""
        y_in  = torch.linspace(-1, 1, in_dim)
        x_in  = torch.zeros(in_dim)
        y_out = torch.linspace(-1, 1, out_dim)
        x_out = torch.zeros(out_dim)

        Y_out, Y_in = torch.meshgrid(y_out, y_in, indexing='ij')
        X_out, X_in = torch.meshgrid(x_out, x_in, indexing='ij')

        c = torch.stack([X_in.flatten(), Y_in.flatten(),
                         X_out.flatten(), Y_out.flatten()], dim=1)
        return c.to(self.device)

    def create_model(self, genome, config):
        """Build a DeltaNetARCModel from a 4-input CPPN genome (2D substrate)."""
        leaf_names = ["x_in", "y_in", "x_out", "y_out"]
        node_names = ["w_q", "w_k", "w_v", "w_beta", "w_out"]

        cppn_nodes = create_cppn(genome, config, leaf_names, node_names)
        node_q, node_k, node_v, node_beta, node_out = cppn_nodes

        model = DeltaNetARCModel(self.vocab_size, self.d_model, self.action_space).to(self.device)

        def _q2(node, coords):
            return node(x_in=coords[:, 0], y_in=coords[:, 1],
                        x_out=coords[:, 2], y_out=coords[:, 3])

        with torch.no_grad():
            model.deltanet.to_q.weight.copy_(
                _q2(node_q,    self._qkv_coords).view(self.dim_head, self.d_model))
            model.deltanet.to_k.weight.copy_(
                _q2(node_k,    self._qkv_coords).view(self.dim_head, self.d_model))
            model.deltanet.to_v.weight.copy_(
                _q2(node_v,    self._qkv_coords).view(self.dim_head, self.d_model))
            model.deltanet.to_beta.weight.copy_(
                _q2(node_beta, self._beta_coords).view(1, self.d_model))
            model.deltanet.out_proj.weight.copy_(
                _q2(node_out,  self._out_proj_coords).view(self.d_model, self.dim_head))

        return model

    # ------------------------------------------------------------------
    # 3D substrate helpers
    # ------------------------------------------------------------------

    def _make_coords_3d(self, out_dim, in_dim, z_in_val=0.0, z_out_val=0.0):
        """
        (out_dim * in_dim, 6) tensor: (x_in, y_in, z_in, x_out, y_out, z_out).

        x/y axes encode position within the weight matrix in [-1, 1].
        z_in_val / z_out_val encode the semantic role of the weight matrix
        so a single shared CPPN can generate differentiated weights per layer.
        """
        y_in  = torch.linspace(-1, 1, in_dim)
        x_in  = torch.zeros(in_dim)
        y_out = torch.linspace(-1, 1, out_dim)
        x_out = torch.zeros(out_dim)

        Y_out, Y_in = torch.meshgrid(y_out, y_in, indexing='ij')
        X_out, X_in = torch.meshgrid(x_out, x_in, indexing='ij')

        N    = out_dim * in_dim
        Z_in  = torch.full((N,), z_in_val)
        Z_out = torch.full((N,), z_out_val)

        c = torch.stack([X_in.flatten(), Y_in.flatten(), Z_in,
                         X_out.flatten(), Y_out.flatten(), Z_out], dim=1)
        return c.to(self.device)

    def create_model_3d(self, genome, config):
        """
        Build a DeltaNetARCModel from a 6-input CPPN genome (3D substrate).

        Requires a NEAT config with num_inputs = 6 (e.g. neat-config-3d.cfg).

        The z coordinate encodes the semantic role of each weight matrix,
        allowing a single evolved CPPN to produce differentiated weights for
        Q, K, V, beta, and out_proj simultaneously.
        """
        leaf_names = ["x_in", "y_in", "z_in", "x_out", "y_out", "z_out"]
        node_names = ["w_q", "w_k", "w_v", "w_beta", "w_out"]

        cppn_nodes = create_cppn(genome, config, leaf_names, node_names)
        node_q, node_k, node_v, node_beta, node_out = cppn_nodes

        # Each weight matrix gets its own z value so the CPPN sees a distinct
        # role signal per matrix type.
        coords_q    = self._make_coords_3d(self.dim_head, self.d_model,
                                           z_out_val=_WEIGHT_ROLES["q"])
        coords_k    = self._make_coords_3d(self.dim_head, self.d_model,
                                           z_out_val=_WEIGHT_ROLES["k"])
        coords_v    = self._make_coords_3d(self.dim_head, self.d_model,
                                           z_out_val=_WEIGHT_ROLES["v"])
        coords_beta = self._make_coords_3d(1,             self.d_model,
                                           z_out_val=_WEIGHT_ROLES["beta"])
        coords_out  = self._make_coords_3d(self.d_model,  self.dim_head,
                                           z_out_val=_WEIGHT_ROLES["out"])

        model = DeltaNetARCModel(self.vocab_size, self.d_model, self.action_space).to(self.device)

        def _q3(node, coords):
            return node(x_in=coords[:, 0], y_in=coords[:, 1], z_in=coords[:, 2],
                        x_out=coords[:, 3], y_out=coords[:, 4], z_out=coords[:, 5])

        with torch.no_grad():
            model.deltanet.to_q.weight.copy_(
                _q3(node_q,    coords_q).view(self.dim_head, self.d_model))
            model.deltanet.to_k.weight.copy_(
                _q3(node_k,    coords_k).view(self.dim_head, self.d_model))
            model.deltanet.to_v.weight.copy_(
                _q3(node_v,    coords_v).view(self.dim_head, self.d_model))
            model.deltanet.to_beta.weight.copy_(
                _q3(node_beta, coords_beta).view(1, self.d_model))
            model.deltanet.out_proj.weight.copy_(
                _q3(node_out,  coords_out).view(self.d_model, self.dim_head))

        return model
    
    def create_meta_model_3d(self, genome, config):
        """
        Build a DeltaNetARCModel from a 6-input CPPN genome (3D substrate).

        Requires a NEAT config with num_inputs = 6 (e.g. neat-config-3d.cfg).

        The z coordinate encodes the semantic role of each weight matrix,
        allowing a single evolved CPPN to produce differentiated weights for
        Q, K, V, beta, and out_proj simultaneously.
        """
        leaf_names = ["x_in", "y_in", "z_in", "x_out", "y_out", "z_out"]
        node_names = ["w_in", "w_q", "w_k", "w_v", "w_beta", "w_out"]

        cppn_nodes = create_cppn(genome, config, leaf_names, node_names)
        node_in, node_q, node_k, node_v, node_beta, node_out = cppn_nodes

        # Each weight matrix gets its own z value so the CPPN sees a distinct
        # role signal per matrix type.
        coords_q    = self._make_coords_3d(self.dim_head, self.d_model,
                                           z_out_val=_WEIGHT_ROLES["q"])
        coords_k    = self._make_coords_3d(self.dim_head, self.d_model,
                                           z_out_val=_WEIGHT_ROLES["k"])
        coords_v    = self._make_coords_3d(self.dim_head, self.d_model,
                                           z_out_val=_WEIGHT_ROLES["v"])
        coords_beta = self._make_coords_3d(1,             self.d_model,
                                           z_out_val=_WEIGHT_ROLES["beta"])
        coords_out  = self._make_coords_3d(self.d_model,  self.dim_head,
                                           z_out_val=_WEIGHT_ROLES["out"])

        model = DeltaNetARCModel(self.vocab_size, self.d_model, self.action_space).to(self.device)

        def _q3(node, coords):
            return node(x_in=coords[:, 0], y_in=coords[:, 1], z_in=coords[:, 2],
                        x_out=coords[:, 3], y_out=coords[:, 4], z_out=coords[:, 5])

        with torch.no_grad():
            model.deltanet.to_q.weight.copy_(
                _q3(node_q,    coords_q).view(self.dim_head, self.d_model))
            model.deltanet.to_k.weight.copy_(
                _q3(node_k,    coords_k).view(self.dim_head, self.d_model))
            model.deltanet.to_v.weight.copy_(
                _q3(node_v,    coords_v).view(self.dim_head, self.d_model))
            model.deltanet.to_beta.weight.copy_(
                _q3(node_beta, coords_beta).view(1, self.d_model))
            model.deltanet.out_proj.weight.copy_(
                _q3(node_out,  coords_out).view(self.d_model, self.dim_head))

        return model