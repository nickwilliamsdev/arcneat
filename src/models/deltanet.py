import torch
import torch.nn as nn
import torch.nn.functional as F

class DeltaNetCell(nn.Module):
    """
    A simplified single step PyTorch implementation of the DeltaNet cell.
    Implements a linear-time RNN relying on the Delta rule for memory updates.
    """
    def __init__(self, d_model, dim_head=64):
        super().__init__()
        self.d_model = d_model
        self.dim_head = dim_head

        # Projections expected for a standard Fast Weight Programmer / DeltaNet
        self.to_q = nn.Linear(d_model, dim_head, bias=False)
        self.to_k = nn.Linear(d_model, dim_head, bias=False)
        self.to_v = nn.Linear(d_model, dim_head, bias=False)
        self.to_beta = nn.Linear(d_model, 1, bias=False)

        self.out_proj = nn.Linear(dim_head, d_model, bias=False)

    def forward(self, x, state=None):
        """
        x: (batch, seq_len, d_model)
        state: Tuple (S, P) where S is the fast weight matrix and P is the normalization factor.
        """
        batch, seq_len, _ = x.shape
        
        q = self.to_q(x) # (B, L, D)
        k = self.to_k(x)
        v = self.to_v(x)
        
        # Sigmoid applied to beta controls the update rate of the memory
        beta = torch.sigmoid(self.to_beta(x)) # (B, L, 1)

        # Normalize keys/queries (e.g., L2 norm) as typical in DeltaNets
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)

        if state is None:
            S = torch.zeros(batch, self.dim_head, self.dim_head, device=x.device)
            P = torch.zeros(batch, self.dim_head, self.dim_head, device=x.device)
        else:
            S, P = state

        outputs = []
        for t in range(seq_len):
            qt = q[:, t, :] # (B, D)
            kt = k[:, t, :]
            vt = v[:, t, :]
            bt = beta[:, t, :]

            # Retrieve from memory: y_t = S_{t-1} q_t
            # (B, D)
            yt = torch.bmm(S, qt.unsqueeze(-1)).squeeze(-1)
            
            # Predict the error using previous fast weights
            v_hat = torch.bmm(S, kt.unsqueeze(-1)).squeeze(-1)
            error = vt - v_hat

            # Update memory using delta rule
            # S_t = S_{t-1} + beta_t * error_t * seq_len
            S = S + bt.unsqueeze(-1) * torch.bmm(error.unsqueeze(-1), kt.unsqueeze(-1).transpose(1, 2))
            
            outputs.append(yt)

        outputs = torch.stack(outputs, dim=1) # (B, L, D)
        out = self.out_proj(outputs)
        
        return out, (S, P)


class DeltaNetARCModel(nn.Module):
    """
    DeltaNet-based architecture for processing ARC grids and producing action values.
    """
    def __init__(self, vocab_size=10, d_model=128, action_space=9000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.deltanet = DeltaNetCell(d_model)
        
        # Outputs policy logits and value estimate for MuZero/RL workflows
        self.policy_head = nn.Linear(d_model, action_space)
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, grid_seq, state=None):
        # grid_seq: (B, L) where L is flattened grid size (e.g. 30x30 = 900)
        x = self.embedding(grid_seq) # (B, L, d_model)
        features, next_state = self.deltanet(x, state)
        
        # Use final sequence element feature for prediction
        final_feature = features[:, -1, :] 
        
        policy_logits = self.policy_head(final_feature)
        value = self.value_head(final_feature)
        
        return policy_logits, value, next_state
